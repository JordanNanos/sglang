---
name: sglang-auto-benchmark
description: Run SGLang auto benchmark searches with tiered server-flag sweeps, canonical dataset preparation, ShareGPT auto-download, custom-data conversion/validation, SLA or fixed-QPS benchmarking, CSV export, and optional second-stage speculative/EAGLE tuning. Use when the user wants an AI-operated benchmark workflow rather than a one-off bench_serving command.
---

# SGLang Auto Benchmark

This skill is for repeatable, AI-driven SGLang performance tuning.

The preferred workflow is:
- start from a mostly pure-TP baseline command,
- move the rest of the performance knobs into `search_space`,
- let auto benchmark search and compare candidates under the target SLA.

The implementation lives in:
- `python -m sglang.auto_benchmark`
- canonical dataset loader in `python -m sglang.bench_serving --dataset-name autobench`
- cookbook-derived LLM reference configs in `.claude/skills/sglang-auto-benchmark/references/cookbook-llm/`

## Preconditions

- SGLang can already launch and serve the target model in this environment.
- The model path exists, or the model is otherwise launchable.
- The goal is clear:
  - benchmark a fixed QPS list, or
  - search the maximum QPS that satisfies `max_ttft_ms` / `max_tpot_ms`.

If those are not true yet, fix them before running a large search.

## Most Important Rule

If the user wants the best command for a **real production or real workload scenario**, the benchmark must use **their real request distribution**.

That means:
- real prompt lengths,
- real output lengths,
- real multi-turn patterns,
- real tool / reasoning / sampling settings,
- real prefix-sharing behavior if it exists.

`sharegpt`, `random`, and `generated-shared-prefix` are useful for sanity checks and broad tuning, but they are not a substitute for the user’s real traffic.

## Supported Dataset Kinds

The current implementation intentionally keeps the dataset surface small:

- `sharegpt`
  - Supports auto-download when no file path is provided.
  - Will be prepared into canonical autobench JSONL on disk before benchmarking.
- `custom`
  - Supports two cases:
    - old `bench_serving` custom conversation JSONL,
    - already-converted canonical autobench JSONL.
- `random`
  - Uses SGLang’s existing synthetic/random benchmark path.
- `generated-shared-prefix`
  - Uses SGLang’s existing shared-prefix synthetic generator.

Everything is normalized into one canonical autobench JSONL file before the benchmark loop starts.

## Canonical Dataset Format

Canonical format is JSONL, one request per line.

Minimal rows:

```json
{"prompt": "Write a summary of this document.", "output_len": 256}
{"prompt": [{"role": "user", "content": "Summarize this document."}], "output_len": 256}
{"prompt": ["first turn", "follow-up turn"], "output_len": 128}
```

Optional fields:

```json
{
  "prompt": [{"role": "user", "content": "Use the weather tool."}],
  "output_len": 256,
  "extra_request_body": {"temperature": 0.0, "top_p": 0.95},
  "image_data": ["file:///tmp/example.png"],
  "timestamp": 1710000000,
  "routing_key": "group-a",
  "metadata": {"source": "custom-upload"}
}
```

Compatibility:
- legacy `messages`
- legacy `prompt_origin`
- legacy `param_send`
- legacy `system + content`

## ShareGPT Auto-Prepare

`sharegpt` does not need a full path.

Example:

```bash
python3 -m sglang.auto_benchmark convert \
  --kind sharegpt \
  --tokenizer /path/to/tokenizer \
  --num-prompts 1200 \
  --output /tmp/sharegpt.autobench.jsonl
```

This will:
- auto-download ShareGPT through the existing SGLang cache path when needed,
- convert it into canonical autobench JSONL,
- save it to the requested output path.

## Custom User Data Workflow

When the user uploads custom data:

1. Inspect a few raw rows first.
2. Decide whether the file is:
   - already canonical autobench JSONL,
   - old `bench_serving` custom format,
   - or an unsupported custom schema that must be transformed manually.
3. If manual transformation is needed:
   - map it into canonical JSONL,
   - never hallucinate missing turns or answers,
   - never keep the final assistant answer as part of the benchmark prompt if that answer is the target completion,
   - preserve per-request generation settings in `extra_request_body`.
4. Run:

```bash
python3 -m sglang.auto_benchmark validate \
  --dataset-path /path/to/converted.autobench.jsonl \
  --tokenizer /path/to/tokenizer
```

5. Manually inspect at least 3 converted rows and confirm:
   - prompt shape is correct,
   - final assistant answer was not accidentally left in the prompt,
   - `output_len` is sensible,
   - request extras were preserved.

## Search Tiers

`search.tier` controls search breadth.

- Tier 1
  - Fast sanity sweep.
  - Baseline plus a small one-at-a-time scan.
- Tier 2
  - Good default.
  - Small cartesian search on the first few high-priority keys plus one-at-a-time expansion for the rest.
- Tier 3
  - Largest search.
  - Full cartesian product of the provided search space.
  - Slowest, but best when the search space is intentionally bounded.

YAML key order matters. Put the most important search keys first.

## What Is Tunable

This workflow is not limited to attention backend tuning.

`server.base_flags` and `server.search_space` are passed directly to `sglang.launch_server`, so in practice any valid server CLI flag can be set or searched.

There is also a small convenience layer for parallel search:

- `server.parallel.tp`
- `server.parallel.pp_size`

When `server.parallel` is used and `dp_size` is not set explicitly, the workflow auto-derives:

`dp_size = visible_gpus / (tp_size * pp_size)`

Visible GPU count is inferred from `server.env.CUDA_VISIBLE_DEVICES` by default, or from `server.parallel.gpu_count` if you set it explicitly.

The most important performance-related groups are:

- Kernel / backend
  - `attention_backend`
  - `prefill_attention_backend`
  - `decode_attention_backend`
  - `sampling_backend`
  - `grammar_backend`
- Batching / scheduling
  - `max_running_requests`
  - `max_queued_requests`
  - `chunked_prefill_size`
  - `prefill_max_requests`
  - `max_prefill_tokens`
  - `schedule_policy`
  - `schedule_conservativeness`
  - `num_continuous_decode_steps`
  - `stream_interval`
- Memory / cache
  - `mem_fraction_static`
  - `max_total_tokens`
  - `page_size`
  - `disable_radix_cache`
- Parallel / distributed execution
  - `tp_size`
  - `pp_size`
  - `dp_size`
  - `ep_size`
  - `load_balance_method`
  - `enable_dp_attention`
  - `enable_mixed_chunk`
  - `disable_overlap_schedule`
- Runtime / CUDA graph
  - keep CUDA graph enabled by default for performance benchmarking
  - `cuda_graph_max_bs`
  - `disable_cuda_graph_padding`
  - `enable_cudagraph_gc`
- Optional speculative / EAGLE stage
  - `speculative_num_steps`
  - `speculative_eagle_topk`
  - `speculative_num_draft_tokens`
  - `speculative_attention_mode`
  - `speculative_draft_attention_backend`
  - `speculative_accept_threshold_single`
  - `speculative_accept_threshold_acc`

In practice, `mem_fraction_static` is usually worth including in the search space with a small conservative range, because it directly changes available KV cache capacity and therefore affects throughput, TTFT stability, and whether a candidate starts successfully at all.

Do not put these into the default search space:
- `enable_hierarchical_cache`
- `hicache_ratio`
- `hicache_size`
- `enable_lmcache`

Those features are not treated as standard auto-benchmark sweep knobs in this workflow.

## Base Tuning Before EAGLE

Never start by tuning EAGLE first.

Use this order:

1. Tune the non-speculative base server first.
2. Find the best normal config for the target dataset and SLA.
3. Only if the user explicitly asks for speculative/EAGLE tuning, and provides the required draft model or equivalent assets, run the second-stage speculative search.

Do not put `disable_cuda_graph` into the default search space. For normal performance tuning, CUDA graph should stay enabled unless the user is debugging compatibility issues.

When a candidate OOMs, keep it in the final result table as a failed row and add a hint such as:
- increase GPU count, or
- use GPUs with larger memory.

## Running The Workflow

Prepare a dataset explicitly:

```bash
python3 -m sglang.auto_benchmark convert \
  --kind custom \
  --path /path/to/data.jsonl \
  --tokenizer /path/to/tokenizer \
  --output /tmp/data.autobench.jsonl
```

Run from config:

```bash
python3 -m sglang.auto_benchmark run --config /path/to/config.yaml
```

Outputs:
- prepared canonical dataset JSONL
- per-run `results.jsonl`
- summary `results.csv`
- per-candidate server logs

## Config Template

Use:
- `references/config-example.yaml`
- `references/qwen3-32b.yaml`
- `references/llama3.1-70b-instruct.yaml`
- `references/minimax-m2.5.yaml`

The current reference set keeps the existing Qwen 32B example and adds:
- Llama 3.1
- MiniMax-M2.5

All reference configs intentionally use Hugging Face repo IDs by default.
You can replace them with local model paths and local dataset paths when needed.

## What To Report Back

After a run, summarize:
- which tier was used,
- which dataset kind was used,
- whether the dataset was synthetic or real user traffic,
- best base config,
- best QPS that satisfied SLA,
- whether speculative tuning was skipped or run,
- paths to:
  - prepared dataset JSONL
  - `results.jsonl`
  - `results.csv`
  - key server logs
