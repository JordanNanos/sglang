# Cookbook LLM References

These configs are derived from `sgl-cookbook` autoregressive text-model pages and normalized into the auto-benchmark config format.

Rules used here:
- Keep the baseline as close as possible to a pure-TP launch command.
- Move common performance knobs into `search_space`.
- Add `ep_size` search for relevant MoE pages.
- Keep CUDA graph enabled by default.
- If a candidate OOMs, the result table should recommend increasing GPU count or using GPUs with larger memory.

Excluded from this folder because they are OCR/VL-oriented rather than text-serving benchmark configs:
- DeepSeekOCR / DeepSeekOCR2
- GLMOCR
- GLM45V / GLM46V
- Qwen2.5-VL / Qwen3-VL
- Step3-VL-10B

Configs in this folder:
- `deepseek-v3.2.yaml`
- `deepseek-math-v2.yaml`
- `deepseek-r1-0528.yaml`
- `deepseek-v3.1.yaml`
- `deepseek-v3.yaml`
- `devstral-small-2-24b-instruct-2512.yaml`
- `ernie-4.5-21b-a3b-pt.yaml`
- `glm-4.5.yaml`
- `glm-4.6.yaml`
- `glm-4.7.yaml`
- `glm-4.7-flash.yaml`
- `glm-5-fp8.yaml`
- `gpt-oss-120b.yaml`
- `glyph.yaml`
- `intern-s1.yaml`
- `kimi-k2.5.yaml`
- `kimi-k2-instruct.yaml`
- `kimi-linear-48b-a3b-instruct.yaml`
- `llada2-1-mini.yaml`
- `ling-2.5-1t.yaml`
- `llama-3.1-70b-instruct.yaml`
- `llama-3.3-70b-instruct.yaml`
- `llama-4-scout-17b-16e-instruct.yaml`
- `llama-4-maverick-17b-128e-instruct-fp8.yaml`
- `mimo-v2-flash.yaml`
- `minimax-m2.5.yaml`
- `minimax-m2.1.yaml`
- `ministral-3-8b-instruct-2512.yaml`
- `mistral-small-4-119b-2603.yaml`
- `nemotron-3-nano-30b-a3b-bf16.yaml`
- `nemotron-3-super-120b-a12b-bf16.yaml`
- `qwen35-397b-a17b-fp8.yaml`
- `qwen3-coder-480b-a35b-instruct.yaml`
- `qwen3-coder-next.yaml`
- `qwen3-235b-a22b.yaml`
- `qwen3-next-80b-a3b-instruct.yaml`
- `ring-2.5-1t.yaml`
- `step-3.5-flash.yaml`
