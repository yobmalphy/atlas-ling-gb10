# Modifications and provenance

This repository is a modified version of the
[Atlas Inference Engine](https://github.com/Avarok-Cybersecurity/atlas), based
on commit `a046cdfdead4b75dc43f6ec346b04733d136d081` (tag `b306`). It was
modified by Jose Lamboy on 2026-08-25.

The derivative remains licensed as AGPL-3.0-only. The root `LICENSE`, upstream
history, copyright notices, citations, and third-party attributions are
preserved. No model weights are included.

## Ling 3.0 additions

- Parse Bailing Hybrid configuration and resolve a native Ling 3.0 GB10 target.
- Load compressed-tensors NVFP4 weights for KDA, MLA, a 512-expert MoE, and the
  Bailing MTP head.
- Add KDA/MLA execution, Bailing MTP proposal/prefill paths, MoE routing
  compatibility, and required GB10 CUDA kernels.
- Add GLM-4.5-compatible tool-call parsing and bounded grammar handling used by
  Ling tool calls. The shared parser fixes also affect Poolside-format users.
- Add targeted Docker build arguments and model metadata for
  `ling-3.0-flash`/`nvfp4`.
- Add configuration, target-resolution, parser, and loader regression tests,
  plus diagnostic PyTorch oracle scripts.
- Declare Qwen3.6 compatibility metadata needed by the shared model-selection
  path.

The diagnostic `scripts/ling_*_oracle.py` programs are engineering oracles,
not a replacement for the Rust test suite or live GB10 validation.

## Hardware validation

The Ling implementation commit `d1f8f520` was validated on one NVIDIA GB10
system with the quantized checkpoint, TP1,
BF16 KV cache, BF16 MTP projections, MTP enabled, and a 262,144-token server
ceiling. The public benchmark artifacts describe the measured configuration
and its limitations. Rebuilding produces a new image; record its source commit
and image digest for deployment provenance.

## TurboQuant attribution boundary

The base Atlas commit was already public and contains upstream TurboQuant+
code and `CITATIONS.md`. This derivative does not claim that work and does not
add TurboQuant+ modifications. All upstream notices and the prior-art chain in
`CITATIONS.md` remain intact.
