import argparse
import csv
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from sglang.benchmark.datasets import get_dataset
from sglang.benchmark.datasets.autobench import (
    sample_autobench_requests,
    serialize_dataset_row_to_autobench,
)
from sglang.benchmark.utils import get_tokenizer

SUPPORTED_DATASETS = {
    "sharegpt",
    "custom",
    "random",
    "generated-shared-prefix",
}

FLAG_ALIASES = {
    "tp": "tp_size",
    "pp": "pp_size",
    "dp": "dp_size",
    "ep": "ep_size",
}

OOM_HINT = "Candidate likely OOMed. Increase GPU count or use GPUs with larger memory."


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def canonical_flag_name(name: str) -> str:
    return FLAG_ALIASES.get(name, name)


def canonicalize_flags(flags: Dict[str, Any]) -> Dict[str, Any]:
    return {canonical_flag_name(key): value for key, value in flags.items()}


def flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


def tail_text(path: str, limit: int = 4000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[-limit:]


def cli_args(flags: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in flags.items():
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
        else:
            args.extend([flag, str(value)])
    return args


def classify_failure(message: str) -> Tuple[Optional[str], Optional[str]]:
    lower = message.lower()
    oom_markers = (
        "out of memory",
        "cuda out of memory",
        "hip out of memory",
        "cudnn_status_alloc_failed",
        "std::bad_alloc",
        "memoryerror",
        "memory allocation",
        "no available memory",
    )
    if any(marker in lower for marker in oom_markers):
        return "oom", OOM_HINT
    return None, None


def prompt_kind(prompt: Any) -> str:
    if isinstance(prompt, str):
        return "prompt"
    if isinstance(prompt, list) and prompt:
        if isinstance(prompt[0], dict):
            return "messages"
        if isinstance(prompt[0], str):
            return "multi_turn"
        if isinstance(prompt[0], int):
            return "token_ids"
    return "unknown"


def summarize_rows(rows: Sequence[Any]) -> Dict[str, Any]:
    kinds: Dict[str, int] = {}
    output_lens = [row.output_len for row in rows]
    for row in rows:
        kind = prompt_kind(row.prompt)
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "num_requests": len(rows),
        "prompt_kinds": kinds,
        "output_len_min": min(output_lens) if output_lens else 0,
        "output_len_max": max(output_lens) if output_lens else 0,
        "output_len_avg": (
            round(sum(output_lens) / len(output_lens), 2) if output_lens else 0.0
        ),
    }


def infer_backend(backend: str, rows: Sequence[Any]) -> str:
    if backend != "auto":
        return backend

    kinds = {prompt_kind(row.prompt) for row in rows}
    if kinds <= {"messages", "multi_turn"}:
        return "sglang-oai-chat"
    if kinds <= {"prompt"}:
        return "sglang-oai"
    if kinds <= {"token_ids"}:
        return "sglang"
    raise ValueError(
        f"Cannot infer backend for mixed prompt kinds: {sorted(kinds)}. "
        "Set benchmark.backend explicitly."
    )


def looks_like_autobench(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(row, dict) and any(
                key in row for key in ("prompt", "messages", "prompt_origin", "system")
            )
    return False


def write_autobench_jsonl(
    path: str, rows: Sequence[Any], metadata: Optional[Dict[str, Any]] = None
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            record = serialize_dataset_row_to_autobench(row, metadata=metadata)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_dataset_cfg(
    dataset_cfg: Optional[Dict[str, Any]], benchmark_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    raw = {} if dataset_cfg is None else dataset_cfg
    if isinstance(raw, str):
        raw = {"kind": raw}
    cfg = dict(raw)

    if "kind" not in cfg and cfg.get("path") in SUPPORTED_DATASETS:
        cfg["kind"] = cfg["path"]
        cfg["path"] = ""

    if "kind" not in cfg and benchmark_cfg.get("dataset_path"):
        cfg["kind"] = "custom"
        cfg["path"] = benchmark_cfg["dataset_path"]

    if "num_prompts" not in cfg and benchmark_cfg.get("num_prompts") is not None:
        cfg["num_prompts"] = benchmark_cfg["num_prompts"]

    cfg["kind"] = cfg.get("kind", "custom")
    if cfg["kind"] == "autobench":
        cfg["kind"] = "custom"
    if cfg["kind"] not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset kind: {cfg['kind']}. "
            f"Supported: {sorted(SUPPORTED_DATASETS)}"
        )
    return cfg


def build_dataset_args(
    dataset_cfg: Dict[str, Any], tokenizer_path: str, model: Optional[str]
) -> SimpleNamespace:
    dataset_path = dataset_cfg.get("path", "")
    if dataset_cfg["kind"] == "sharegpt" and dataset_path in ("", None, "sharegpt"):
        dataset_path = ""

    return SimpleNamespace(
        dataset_name=dataset_cfg["kind"],
        dataset_path=dataset_path,
        tokenizer=tokenizer_path,
        model=model,
        num_prompts=int(dataset_cfg.get("num_prompts", 1000)),
        sharegpt_output_len=dataset_cfg.get("output_len"),
        sharegpt_context_len=dataset_cfg.get("context_len"),
        random_input_len=int(dataset_cfg.get("random_input_len", 1024)),
        random_output_len=int(dataset_cfg.get("random_output_len", 256)),
        random_range_ratio=float(dataset_cfg.get("random_range_ratio", 0.0)),
        prompt_suffix=dataset_cfg.get("prompt_suffix", ""),
        apply_chat_template=bool(dataset_cfg.get("apply_chat_template", False)),
        gsp_num_groups=int(dataset_cfg.get("gsp_num_groups", 64)),
        gsp_prompts_per_group=int(dataset_cfg.get("gsp_prompts_per_group", 16)),
        gsp_system_prompt_len=int(dataset_cfg.get("gsp_system_prompt_len", 2048)),
        gsp_question_len=int(dataset_cfg.get("gsp_question_len", 128)),
        gsp_output_len=int(dataset_cfg.get("gsp_output_len", 256)),
        gsp_range_ratio=float(dataset_cfg.get("gsp_range_ratio", 1.0)),
        gsp_fast_prepare=bool(dataset_cfg.get("gsp_fast_prepare", False)),
        gsp_send_routing_key=bool(dataset_cfg.get("gsp_send_routing_key", False)),
        gsp_num_turns=int(dataset_cfg.get("gsp_num_turns", 1)),
        gsp_ordered=bool(dataset_cfg.get("gsp_ordered", False)),
        seed=int(dataset_cfg.get("seed", 1)),
    )


def load_autobench_rows(
    dataset_path: str,
    tokenizer_path: str,
    num_prompts: int = 0,
    output_len: Optional[int] = None,
) -> List[Any]:
    return sample_autobench_requests(
        dataset_path=dataset_path,
        num_requests=num_prompts,
        tokenizer=get_tokenizer(tokenizer_path),
        fixed_output_len=output_len,
    )


def prepare_dataset(
    dataset_cfg: Dict[str, Any],
    tokenizer_path: str,
    model: Optional[str],
    output_path: str,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    if dataset_cfg["kind"] == "custom" and looks_like_autobench(
        dataset_cfg.get("path", "")
    ):
        rows = load_autobench_rows(
            dataset_path=dataset_cfg["path"],
            tokenizer_path=tokenizer_path,
            num_prompts=int(dataset_cfg.get("num_prompts", 0)),
            output_len=dataset_cfg.get("output_len"),
        )
    else:
        tokenizer = get_tokenizer(tokenizer_path)
        dataset_args = build_dataset_args(dataset_cfg, tokenizer_path, model)
        rows = get_dataset(dataset_args, tokenizer=tokenizer, model_id=model)

    if not rows:
        raise ValueError("Prepared dataset is empty.")

    write_autobench_jsonl(
        output_path,
        rows,
        metadata={
            "source_dataset_name": dataset_cfg["kind"],
            "source_dataset_path": dataset_cfg.get("path") or dataset_cfg["kind"],
        },
    )
    return output_path, rows, summarize_rows(rows)


def infer_total_gpus(server_cfg: Dict[str, Any]) -> Optional[int]:
    parallel_cfg = server_cfg.get("parallel", {})
    for key in ("gpu_count",):
        value = parallel_cfg.get(key, server_cfg.get(key))
        if value is not None:
            return int(value)

    env = server_cfg.get("env", {})
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        value = env.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if not value or value.lower() in {"all", "none", "void"}:
            continue
        return len([item for item in value.split(",") if item.strip()])
    return None


def resolve_parallelism(
    server_cfg: Dict[str, Any], flags: Dict[str, Any], parallel_requested: bool
) -> Dict[str, Any]:
    flags = canonicalize_flags(flags)
    if not parallel_requested:
        return flags

    tp_size = int(flags.get("tp_size", 1))
    pp_size = int(flags.get("pp_size", 1))
    if "dp_size" in flags:
        return flags

    total_gpus = infer_total_gpus(server_cfg)
    if total_gpus is None:
        raise ValueError(
            "Cannot infer total GPU count for parallel search. "
            "Set server.parallel.gpu_count or server.env.CUDA_VISIBLE_DEVICES."
        )

    shard_size = tp_size * pp_size
    if shard_size <= 0 or total_gpus % shard_size != 0:
        raise ValueError(
            f"Cannot derive dp_size: total_gpus={total_gpus}, "
            f"tp_size={tp_size}, pp_size={pp_size}."
        )

    flags["dp_size"] = total_gpus // shard_size
    return flags


def build_server_candidates(
    server_cfg: Dict[str, Any], tier: int, max_candidates: Optional[int]
) -> List[Dict[str, Any]]:
    base_flags = canonicalize_flags(deepcopy(server_cfg.get("base_flags", {})))
    search_space = canonicalize_flags(deepcopy(server_cfg.get("search_space", {})))
    parallel_cfg = canonicalize_flags(deepcopy(server_cfg.get("parallel", {})))
    parallel_requested = bool(parallel_cfg)
    for key, value in parallel_cfg.items():
        if key == "gpu_count":
            continue
        values = as_list(value)
        if values:
            base_flags.setdefault(key, values[0])
    search_space.update(
        {key: value for key, value in parallel_cfg.items() if key != "gpu_count"}
    )

    candidates = build_candidates(
        base_flags=base_flags,
        search_space=search_space,
        tier=tier,
        max_candidates=max_candidates,
    )
    return [
        resolve_parallelism(server_cfg, candidate, parallel_requested)
        for candidate in candidates
    ]


def build_candidates(
    base_flags: Dict[str, Any],
    search_space: Dict[str, Sequence[Any]],
    tier: int,
    max_candidates: Optional[int],
) -> List[Dict[str, Any]]:
    base_flags = canonicalize_flags(base_flags)
    search_space = canonicalize_flags(search_space)
    items = [(key, as_list(values)) for key, values in search_space.items()]
    if tier == 1:
        items = [(k, v[:2]) for k, v in items[:6]]
    elif tier == 2:
        items = [(k, v[:3]) for k, v in items[:8]]

    candidates = [deepcopy(base_flags)]
    if tier == 1:
        for key, values in items:
            for value in values:
                candidates.append(deepcopy(base_flags) | {key: value})
    elif tier == 2 and items:
        head, tail = items[:3], items[3:]
        for combo in itertools.product(*[values for _, values in head]):
            candidate = deepcopy(base_flags)
            for (key, _), value in zip(head, combo):
                candidate[key] = value
            candidates.append(candidate)
        for key, values in tail:
            for value in values:
                candidates.append(deepcopy(base_flags) | {key: value})
    elif tier == 3 and items:
        for combo in itertools.product(*[values for _, values in items]):
            candidate = deepcopy(base_flags)
            for (key, _), value in zip(items, combo):
                candidate[key] = value
            candidates.append(candidate)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if max_candidates and len(deduped) >= max_candidates:
            break
    return deduped


def build_qps_plan(benchmark_cfg: Dict[str, Any]) -> Tuple[str, List[float], float]:
    qps_cfg = benchmark_cfg.get("qps", benchmark_cfg.get("request_rate"))
    if isinstance(qps_cfg, list):
        return "fixed", [float(value) for value in qps_cfg], 0.0
    if isinstance(qps_cfg, dict) and "values" in qps_cfg:
        return "fixed", [float(value) for value in qps_cfg["values"]], 0.0
    if isinstance(qps_cfg, dict) and {"lower", "upper"} <= set(qps_cfg):
        return (
            "search",
            [float(qps_cfg["lower"]), float(qps_cfg["upper"])],
            float(qps_cfg.get("tolerance", 0.1)),
        )
    raise ValueError("benchmark.qps must be a list or a {lower, upper, tolerance} map.")


def meets_sla(result: Dict[str, Any], benchmark_cfg: Dict[str, Any]) -> bool:
    sla = benchmark_cfg.get("sla", {})
    max_ttft_ms = sla.get("max_ttft_ms")
    max_tpot_ms = sla.get("max_tpot_ms")
    if (
        max_ttft_ms is not None
        and result.get("mean_ttft_ms", float("inf")) > max_ttft_ms
    ):
        return False
    if (
        max_tpot_ms is not None
        and result.get("mean_tpot_ms", float("inf")) > max_tpot_ms
    ):
        return False
    return True


def result_sort_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        1 if record.get("sla_passed") else 0,
        record.get("requested_qps", 0.0),
        record.get("metrics", {}).get("output_throughput", 0.0),
        -record.get("metrics", {}).get("mean_ttft_ms", float("inf")),
        -record.get("metrics", {}).get("mean_tpot_ms", float("inf")),
    )


def launch_server(
    server_cfg: Dict[str, Any], server_flags: Dict[str, Any], log_path: str
) -> subprocess.Popen:
    command_prefix = server_cfg.get("command_prefix")
    if command_prefix is None:
        command = [sys.executable, "-m", "sglang.launch_server"]
    elif isinstance(command_prefix, str):
        command = shlex.split(command_prefix)
    else:
        command = [str(item) for item in command_prefix]

    command.extend(cli_args(server_flags))
    command.extend(str(item) for item in server_cfg.get("extra_args", []))

    env = os.environ.copy()
    env.update({key: str(value) for key, value in server_cfg.get("env", {}).items()})
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    process._autobench_log_file = log_file  # type: ignore[attr-defined]
    return process


def stop_server(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
    finally:
        log_file = getattr(process, "_autobench_log_file", None)
        if log_file is not None:
            log_file.close()


def build_bench_command(
    benchmark_cfg: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    backend: str,
    base_url: str,
    dataset_path: str,
    tokenizer_path: str,
    request_rate: float,
    max_concurrency: Optional[int],
    output_file: str,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        backend,
        "--base-url",
        base_url,
        "--dataset-name",
        "autobench",
        "--dataset-path",
        dataset_path,
        "--tokenizer",
        tokenizer_path,
        "--num-prompts",
        str(dataset_summary["num_requests"]),
        "--request-rate",
        str(request_rate),
        "--output-file",
        output_file,
        "--seed",
        str(int(benchmark_cfg.get("seed", 1))),
        "--ready-check-timeout-sec",
        str(int(benchmark_cfg.get("ready_check_timeout_sec", 600))),
    ]
    if benchmark_cfg.get("model"):
        command.extend(["--model", str(benchmark_cfg["model"])])
    if benchmark_cfg.get("served_model_name"):
        command.extend(["--served-model-name", str(benchmark_cfg["served_model_name"])])
    if benchmark_cfg.get("disable_tqdm", True):
        command.append("--disable-tqdm")
    if benchmark_cfg.get("output_details"):
        command.append("--output-details")
    if benchmark_cfg.get("disable_stream"):
        command.append("--disable-stream")
    if benchmark_cfg.get("disable_ignore_eos"):
        command.append("--disable-ignore-eos")
    if benchmark_cfg.get("pd_separated"):
        command.append("--pd-separated")
    if benchmark_cfg.get("flush_cache"):
        command.append("--flush-cache")
    if benchmark_cfg.get("tag"):
        command.extend(["--tag", str(benchmark_cfg["tag"])])
    if max_concurrency is not None:
        command.extend(["--max-concurrency", str(max_concurrency)])
    if benchmark_cfg.get("warmup_requests") is not None:
        command.extend(
            ["--warmup-requests", str(int(benchmark_cfg["warmup_requests"]))]
        )
    if benchmark_cfg.get("extra_request_body") is not None:
        command.extend(
            [
                "--extra-request-body",
                json.dumps(benchmark_cfg["extra_request_body"]),
            ]
        )
    return command


def run_bench_command(command: List[str]) -> Dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-4000:])

    output_file = command[command.index("--output-file") + 1]
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError("bench_serving produced no JSONL output")
    return json.loads(lines[-1])


def run_trial(
    stage_name: str,
    candidate_id: int,
    server_cfg: Dict[str, Any],
    benchmark_cfg: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    backend: str,
    dataset_path: str,
    tokenizer_path: str,
    server_flags: Dict[str, Any],
    output_dir: str,
    request_rate: float,
    max_concurrency: Optional[int],
) -> Dict[str, Any]:
    process = None
    log_path = os.path.join(
        output_dir,
        f"server_{stage_name}_cand{candidate_id}_mc{max_concurrency}_q{request_rate}.log",
    )
    bench_path = os.path.join(
        output_dir,
        f"bench_{stage_name}_cand{candidate_id}_mc{max_concurrency}_q{request_rate}.jsonl",
    )
    host = server_cfg.get("host", "127.0.0.1")
    port = int(server_flags.get("port", server_cfg.get("port", 30000)))
    base_url = benchmark_cfg.get("base_url", f"http://{host}:{port}")
    record = {
        "stage": stage_name,
        "candidate_id": candidate_id,
        "requested_qps": request_rate,
        "max_concurrency": max_concurrency,
        "server_flags": deepcopy(server_flags),
        "sla_passed": False,
    }

    try:
        if server_cfg.get("launch", True):
            process = launch_server(server_cfg, server_flags, log_path)
        metrics = run_bench_command(
            build_bench_command(
                benchmark_cfg=benchmark_cfg,
                dataset_summary=dataset_summary,
                backend=backend,
                base_url=base_url,
                dataset_path=dataset_path,
                tokenizer_path=tokenizer_path,
                request_rate=request_rate,
                max_concurrency=max_concurrency,
                output_file=bench_path,
            )
        )
        record["sla_passed"] = meets_sla(metrics, benchmark_cfg)
        record["metrics"] = metrics
    except Exception as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        diagnosis, hint = classify_failure(
            "\n".join(part for part in [repr(exc), tail_text(log_path)] if part)
        )
        if diagnosis:
            record["diagnosis"] = diagnosis
        if hint:
            record["hint"] = hint
    finally:
        stop_server(process)
    return record


def merge_host_port(
    server_cfg: Dict[str, Any], flags: Dict[str, Any]
) -> Dict[str, Any]:
    merged = canonicalize_flags(deepcopy(flags))
    if server_cfg.get("host") is not None and "host" not in merged:
        merged["host"] = server_cfg["host"]
    if server_cfg.get("port") is not None and "port" not in merged:
        merged["port"] = server_cfg["port"]
    return merged


def run_candidate(
    stage_name: str,
    candidate_id: int,
    server_cfg: Dict[str, Any],
    benchmark_cfg: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    backend: str,
    dataset_path: str,
    tokenizer_path: str,
    server_flags: Dict[str, Any],
    output_dir: str,
) -> List[Dict[str, Any]]:
    mode, values, tolerance = build_qps_plan(benchmark_cfg)
    max_concurrency_values = as_list(benchmark_cfg.get("max_concurrency", [None]))
    records: List[Dict[str, Any]] = []

    def one_trial(
        request_rate: float, max_concurrency: Optional[int]
    ) -> Dict[str, Any]:
        return run_trial(
            stage_name=stage_name,
            candidate_id=candidate_id,
            server_cfg=server_cfg,
            benchmark_cfg=benchmark_cfg,
            dataset_summary=dataset_summary,
            backend=backend,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            server_flags=server_flags,
            output_dir=output_dir,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
        )

    for max_concurrency in max_concurrency_values:
        if mode == "fixed":
            records.extend(one_trial(qps, max_concurrency) for qps in values)
            continue

        lower, upper = values
        best: Optional[Dict[str, Any]] = None
        while upper - lower > tolerance:
            qps = round((lower + upper) / 2, 4)
            record = one_trial(qps, max_concurrency)
            records.append(record)
            if record.get("metrics") and record["sla_passed"]:
                lower = qps
                best = record
            else:
                upper = qps
        if best is not None:
            best["best_for_candidate"] = True

    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    rows = [flatten(record) for record in records]
    headers = sorted({header for row in rows for header in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_stage(
    stage_name: str,
    candidates: Sequence[Dict[str, Any]],
    server_cfg: Dict[str, Any],
    benchmark_cfg: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    backend: str,
    dataset_path: str,
    tokenizer_path: str,
    output_dir: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    best_record: Optional[Dict[str, Any]] = None

    for candidate_id, candidate_flags in enumerate(candidates):
        merged = merge_host_port(server_cfg, candidate_flags)
        print(
            f"[{stage_name}] candidate {candidate_id + 1}/{len(candidates)}: "
            f"{json.dumps(merged, ensure_ascii=False)}"
        )
        candidate_records = run_candidate(
            stage_name=stage_name,
            candidate_id=candidate_id,
            server_cfg=server_cfg,
            benchmark_cfg=benchmark_cfg,
            dataset_summary=dataset_summary,
            backend=backend,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            server_flags=merged,
            output_dir=output_dir,
        )
        records.extend(candidate_records)

        for record in candidate_records:
            if not record.get("metrics"):
                continue
            if best_record is None or result_sort_key(record) > result_sort_key(
                best_record
            ):
                best_record = record

    return records, best_record


def run_auto_benchmark(config_path: str) -> str:
    config = load_yaml(config_path)
    server_cfg = config["server"]
    benchmark_cfg = config["benchmark"]
    search_cfg = config.get("search", {})

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = benchmark_cfg.get("output_dir") or os.path.join(
        os.getcwd(), "auto_benchmark_results", timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_path = benchmark_cfg.get("tokenizer") or server_cfg.get(
        "base_flags", {}
    ).get("model_path")
    model = benchmark_cfg.get("model") or server_cfg.get("base_flags", {}).get(
        "model_path"
    )
    if tokenizer_path is None:
        raise ValueError(
            "benchmark.tokenizer or server.base_flags.model_path is required."
        )

    dataset_cfg = normalize_dataset_cfg(config.get("dataset"), benchmark_cfg)
    prepared_dataset_path, rows, dataset_summary = prepare_dataset(
        dataset_cfg=dataset_cfg,
        tokenizer_path=tokenizer_path,
        model=model,
        output_path=os.path.join(output_dir, "prepared_dataset.jsonl"),
    )
    backend = infer_backend(benchmark_cfg.get("backend", "auto"), rows)
    print(f"prepared_dataset={prepared_dataset_path}")
    print(f"dataset_summary={json.dumps(dataset_summary, ensure_ascii=False)}")
    print(f"selected_backend={backend}")

    tier = int(search_cfg.get("tier", 1))
    max_candidates = search_cfg.get("max_candidates")
    base_candidates = build_server_candidates(server_cfg, tier, max_candidates)
    all_records, best_base = run_stage(
        stage_name="base",
        candidates=base_candidates,
        server_cfg=server_cfg,
        benchmark_cfg=benchmark_cfg,
        dataset_summary=dataset_summary,
        backend=backend,
        dataset_path=prepared_dataset_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
    )

    speculative_cfg = config.get("speculative", {})
    if speculative_cfg.get("enabled"):
        if best_base is None:
            raise ValueError(
                "Speculative search requires at least one successful base run."
            )
        if not speculative_cfg.get("draft_model_path"):
            raise ValueError("speculative.draft_model_path is required.")

        spec_base_flags = deepcopy(best_base["server_flags"])
        spec_base_flags.update(deepcopy(speculative_cfg.get("base_flags", {})))
        spec_base_flags["speculative_algorithm"] = speculative_cfg.get(
            "algorithm", "EAGLE"
        )
        spec_base_flags["speculative_draft_model_path"] = speculative_cfg[
            "draft_model_path"
        ]
        spec_candidates = build_candidates(
            base_flags=canonicalize_flags(spec_base_flags),
            search_space=deepcopy(speculative_cfg.get("search_space", {})),
            tier=tier,
            max_candidates=max_candidates,
        )
        spec_records, _ = run_stage(
            stage_name="speculative",
            candidates=spec_candidates,
            server_cfg=server_cfg,
            benchmark_cfg=benchmark_cfg,
            dataset_summary=dataset_summary,
            backend=backend,
            dataset_path=prepared_dataset_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
        )
        all_records.extend(spec_records)

    results_jsonl = os.path.join(output_dir, "results.jsonl")
    results_csv = os.path.join(output_dir, "results.csv")
    write_jsonl(results_jsonl, all_records)
    write_csv(results_csv, all_records)
    print(f"results_jsonl={results_jsonl}")
    print(f"results_csv={results_csv}")
    return output_dir


def convert_dataset(args: argparse.Namespace) -> None:
    dataset_cfg = normalize_dataset_cfg(
        {
            key: value
            for key, value in vars(args).items()
            if key not in {"command", "output", "tokenizer", "model"}
        },
        {},
    )
    output_path, rows, summary = prepare_dataset(
        dataset_cfg=dataset_cfg,
        tokenizer_path=args.tokenizer,
        model=args.model,
        output_path=args.output,
    )
    print(f"prepared_dataset={output_path}")
    print(f"rows={len(rows)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_dataset(args: argparse.Namespace) -> None:
    rows = load_autobench_rows(args.dataset_path, args.tokenizer, num_prompts=0)
    print(json.dumps(summarize_rows(rows), ensure_ascii=False, indent=2))
