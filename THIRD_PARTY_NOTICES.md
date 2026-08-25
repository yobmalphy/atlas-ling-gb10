# Third-party notices

This file supplements, and does not replace, notices already present in the
source tree.

## Atlas Inference Engine

- Source: <https://github.com/Avarok-Cybersecurity/atlas>
- License: AGPL-3.0-only; see `LICENSE`.
- Base commit: `a046cdfdead4b75dc43f6ec346b04733d136d081`.
- TurboQuant and other kernel citations are preserved in `CITATIONS.md` and in
  the relevant source headers.

## Ling checkpoints

Weights are not distributed by this repository. Obtain them from their model
cards and review the current terms before use:

- Quantized checkpoint:
  <https://huggingface.co/kingjones777/Ling-3.0-flash-NVFP4-SGLang-MTP>
- Base model: <https://huggingface.co/inclusionAI/Ling-3.0-Flash>

At publication time, those model repositories displayed MIT licensing. Their
owners control the model weights and model-card terms; verify them again for a
production deployment.

## Benchmark dataset

- LiveCodeBench: <https://github.com/LiveCodeBench/LiveCodeBench>
- Qwen benchmark checkpoint:
  <https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>
- Ornith benchmark checkpoint:
  <https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4>
- The published charts contain task identifiers and measurements, not copied
  prompts, tests, or model reasoning traces.

Exact revisions and the test-data hash are pinned in
`benchmarks/livecodebench-test6-medium/README.md`.
