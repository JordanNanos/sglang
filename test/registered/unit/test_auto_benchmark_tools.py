import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

sys.modules.setdefault("zmq", types.SimpleNamespace())

from sglang.auto_benchmark_lib import (
    build_candidates,
    build_server_candidates,
    classify_failure,
    infer_backend,
    prepare_dataset,
)
from sglang.benchmark.datasets.autobench import sample_autobench_requests
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def create_lightweight_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({f"tok_{i}": i + 4 for i in range(4096)})

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return hf_tokenizer


class TestAutoBenchmarkTools(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        self.tokenizer = create_lightweight_tokenizer()
        self.tokenizer_dir = self.tmpdir_path / "tok"
        self.tokenizer.save_pretrained(self.tokenizer_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_autobench_jsonl(self) -> str:
        rows = [
            {"prompt": "tok_1 tok_2 tok_3", "output_len": 32},
            {
                "messages": [{"role": "user", "content": "tok_4 tok_5"}],
                "output_len": 24,
                "extra_request_body": {"temperature": 0.0},
            },
            {
                "system": "tok_6",
                "content": ["tok_7 tok_8", "tok_9", "tok_10 tok_11"],
                "output_len": 16,
            },
        ]
        path = self.tmpdir_path / "sample.autobench.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def _write_sharegpt_json(self) -> str:
        rows = [
            {
                "conversations": [
                    {"value": "tok_1 tok_2 tok_3"},
                    {"value": "tok_4 tok_5"},
                ]
            },
            {
                "conversations": [
                    {"value": "tok_6 tok_7"},
                    {"value": "tok_8 tok_9 tok_10"},
                ]
            },
        ]
        path = self.tmpdir_path / "sharegpt.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        return str(path)

    def test_prepare_custom_autobench_dataset(self):
        dataset_path = self._write_autobench_jsonl()
        output_path = self.tmpdir_path / "prepared.autobench.jsonl"

        prepared_path, rows, summary = prepare_dataset(
            dataset_cfg={
                "kind": "custom",
                "path": dataset_path,
                "num_prompts": 2,
            },
            tokenizer_path=str(self.tokenizer_dir),
            model=None,
            output_path=str(output_path),
        )

        self.assertEqual(prepared_path, str(output_path))
        self.assertEqual(summary["num_requests"], 2)
        self.assertTrue(Path(prepared_path).exists())
        converted_rows = sample_autobench_requests(
            dataset_path=prepared_path,
            num_requests=0,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(converted_rows), 2)

    def test_prepare_sharegpt_dataset(self):
        sharegpt_path = self._write_sharegpt_json()
        output_path = self.tmpdir_path / "sharegpt.autobench.jsonl"

        prepared_path, rows, summary = prepare_dataset(
            dataset_cfg={
                "kind": "sharegpt",
                "path": sharegpt_path,
                "num_prompts": 2,
            },
            tokenizer_path=str(self.tokenizer_dir),
            model=None,
            output_path=str(output_path),
        )

        self.assertEqual(prepared_path, str(output_path))
        self.assertEqual(summary["num_requests"], 2)
        self.assertEqual(len(rows), 2)

    def test_infer_backend(self):
        prompt_rows = [SimpleNamespace(prompt="tok_1 tok_2")]
        chat_rows = [SimpleNamespace(prompt=[{"role": "user", "content": "tok_1"}])]
        token_id_rows = [SimpleNamespace(prompt=[1, 2, 3])]

        self.assertEqual(infer_backend("auto", prompt_rows), "sglang-oai")
        self.assertEqual(infer_backend("auto", chat_rows), "sglang-oai-chat")
        self.assertEqual(infer_backend("auto", token_id_rows), "sglang")

    def test_build_candidates_by_tier(self):
        base_flags = {"model_path": "/model", "tp_size": 4}
        search_space = {
            "prefill_attention_backend": ["fa3", "flashinfer", "triton"],
            "decode_attention_backend": ["fa3", "flashinfer"],
            "chunked_prefill_size": [4096, 8192],
            "max_running_requests": [64, 128],
            "schedule_policy": ["lpm", "fcfs"],
        }

        tier1 = build_candidates(base_flags, search_space, tier=1, max_candidates=None)
        tier2 = build_candidates(base_flags, search_space, tier=2, max_candidates=None)
        tier3 = build_candidates(base_flags, search_space, tier=3, max_candidates=32)

        self.assertGreater(len(tier1), 1)
        self.assertGreater(len(tier2), len(tier1))
        self.assertGreater(len(tier3), len(tier2))
        self.assertEqual(tier1[0]["model_path"], "/model")

    def test_parallel_search_derives_dp_size(self):
        server_cfg = {
            "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            "base_flags": {"model_path": "/model"},
            "parallel": {
                "tp": [4, 2],
                "pp_size": [1],
            },
            "search_space": {},
        }

        candidates = build_server_candidates(server_cfg, tier=2, max_candidates=None)
        tp_dp_pairs = {
            (candidate["tp_size"], candidate["dp_size"]) for candidate in candidates
        }
        self.assertIn((4, 2), tp_dp_pairs)
        self.assertIn((2, 4), tp_dp_pairs)

    def test_ep_alias_and_oom_classification(self):
        server_cfg = {
            "base_flags": {"model_path": "/model", "tp_size": 8},
            "search_space": {"ep": [1, 4]},
        }

        candidates = build_server_candidates(server_cfg, tier=2, max_candidates=None)
        ep_sizes = {candidate.get("ep_size", 1) for candidate in candidates}
        self.assertEqual(ep_sizes, {1, 4})

        diagnosis, hint = classify_failure("RuntimeError: CUDA out of memory")
        self.assertEqual(diagnosis, "oom")
        self.assertIn("Increase GPU count", hint)


if __name__ == "__main__":
    unittest.main()
