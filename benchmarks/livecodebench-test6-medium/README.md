# Reproduce the second benchmark

This directory contains the complete harness for the **second test only**: the
eight-task LiveCodeBench `test6` medium comparison shown in the published
charts. The earlier mixed 11-task experiment is intentionally not included.

## What is pinned

- Dataset: `livecodebench/code_generation_lite`, file `test6.jsonl`.
- Dataset revision: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
- Dataset SHA-256:
  `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`.
- Official grader: `LiveCodeBench/LiveCodeBench` commit
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Tasks, in order: `abc387_c`, `abc389_d`, `abc390_d`, `abc390_c`,
  `abc394_d`, `abc396_d`, `abc397_c`, `abc398_c`.
- Sampling: deterministic pass@1, temperature `0`, top-p `1`, one generation,
  8,192-token completion allowance, thinking enabled with a 512-token budget.
- Server: TP1, 262,144-token ceiling, BF16 KV, one sequence, batch size one,
  MTP enabled, prefix caching disabled.

The runner downloads the pinned dataset revision and refuses to continue if
its SHA-256 differs.

## Checkpoints used

| Label | Hugging Face checkpoint | Revision | MTP drafts |
|---|---|---|---:|
| Ling 3.0 | `kingjones777/Ling-3.0-flash-NVFP4-SGLang-MTP` | `8d10afa56d671e97d73285708bd29f6014161913` | 1 (K=2) |
| Qwen 3.8 27B | `unsloth/Qwen3.8-27B-NVFP4` | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | 3 (K=4) |
| Ornith 1.5 35B | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | `0f0b1b59b879ccde1353e6ebd0fb10c204d4c544` | 1 (K=2) |

Download those exact revisions rather than each repository's moving `main`.
Model owners control the weight licenses and availability.

## 1. Prepare the grader

Run this on a trusted Linux host. The official grader executes generated model
code; use an isolated machine or disposable VM/container, never a production
host with secrets or valuable writable mounts.

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git /tmp/LiveCodeBench
git -C /tmp/LiveCodeBench checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24

python3 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install --upgrade pip
python -m pip install -e /tmp/LiveCodeBench
```

## 2. Start one model at a time

Use the same Atlas source/image for all three legs. Record both values before
running:

```bash
ATLAS_COMMIT=$(git rev-parse HEAD)
ATLAS_IMAGE=$(docker image inspect atlas-ling-gb10:latest --format '{{.Id}}')
```

The common serve contract is:

```text
--max-seq-len 262144
--max-prefill-tokens 8192
--kv-cache-dtype bf16
--gpu-memory-utilization 0.85
--max-num-seqs 1
--max-batch-size 1
--lm-head-dtype bf16
--ssm-cache-slots 0
--enable-prefix-caching false
--speculative
--mtp-quantization bf16
--mtp-gate force
--request-timeout 1800
--no-auto-swap
--tp-size 1
--no-tui
```

Add `--num-drafts 1` for Ling and Ornith; use `--num-drafts 3` for Qwen.
Ling's exact launch recipe is in [`../../docs/LING3_GB10.md`](../../docs/LING3_GB10.md).
For Qwen select kernel target `qwen3.8-27b`; for Ornith select
`qwen3.6-35b-a3b`. Stop each server before loading the next model so the three
legs do not compete for GB10 memory.

Verify `/v1/models` reports the expected model and `max_model_len: 262144`
before every leg. Save `docker inspect` output alongside your private run
artifacts if you need auditable image and command provenance.

## 3. Capture each leg

Set `ENDPOINT` to localhost or to your private Tailscale address. The examples
use placeholders and do not publish the original private endpoint.

```bash
RUNNER=benchmarks/livecodebench-test6-medium/reproduce.py
ENDPOINT=http://127.0.0.1:8888/v1
RESULTS=benchmark-results/second-test

python "$RUNNER" run \
  --endpoint "$ENDPOINT" \
  --model-label ling-3.0-flash-nvfp4-mtp-k2 \
  --served-model ling-3.0-flash-nvfp4-mtp \
  --checkpoint kingjones777/Ling-3.0-flash-NVFP4-SGLang-MTP \
  --checkpoint-revision 8d10afa56d671e97d73285708bd29f6014161913 \
  --atlas-commit "$ATLAS_COMMIT" --atlas-image "$ATLAS_IMAGE" \
  --mtp-drafts 1 --output "$RESULTS/ling-3.0-flash"

python "$RUNNER" run \
  --endpoint "$ENDPOINT" \
  --model-label qwen3.8-27b-nvfp4-mtp-k4 \
  --served-model qwen3.8-27b-second-place-262k-mtp \
  --checkpoint unsloth/Qwen3.8-27B-NVFP4 \
  --checkpoint-revision 7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108 \
  --atlas-commit "$ATLAS_COMMIT" --atlas-image "$ATLAS_IMAGE" \
  --mtp-drafts 3 --output "$RESULTS/qwen3.8-27b"

python "$RUNNER" run \
  --endpoint "$ENDPOINT" \
  --model-label ornith-1.5-35b-nvfp4-mtp-k2 \
  --served-model ornith-1.5-35b-second-place-262k-mtp \
  --checkpoint ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
  --checkpoint-revision 0f0b1b59b879ccde1353e6ebd0fb10c204d4c544 \
  --atlas-commit "$ATLAS_COMMIT" --atlas-image "$ATLAS_IMAGE" \
  --mtp-drafts 1 --output "$RESULTS/ornith-1.5-35b"
```

Do not run the legs concurrently. Existing result files are skipped; pass
`--force` only when you intentionally want to replace a captured task.

## 4. Grade and report

```bash
for model in ling-3.0-flash qwen3.8-27b ornith-1.5-35b; do
  python "$RUNNER" grade \
    --result-dir "$RESULTS/$model" \
    --lcb-repo /tmp/LiveCodeBench
done

python "$RUNNER" report \
  --results-root "$RESULTS" \
  --output "$RESULTS/REPORT.md"
```

Compare the generated report with
[`../../docs/benchmarks/ling-gb10/THREE_MODEL_REPORT.md`](../../docs/benchmarks/ling-gb10/THREE_MODEL_REPORT.md).
Minor TPS and TTFT variation is normal; correctness is deterministic only if
the checkpoint revisions, Atlas commit/image, prompts, and serving flags match.

## Publication hygiene

Raw result JSON contains complete model output and reasoning text. Inspect and
sanitize it before publishing. The repository publishes only aggregate and
per-task metrics, not raw reasoning, task prompts, hidden tests, local paths,
or private network addresses.
