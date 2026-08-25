# Ling 3.0 Flash NVFP4 + MTP on NVIDIA GB10

This target runs the Bailing Hybrid Ling 3.0 Flash checkpoint natively in Atlas
on a single NVIDIA GB10/DGX Spark. The live validation configuration used TP1,
NVFP4 main weights, BF16 KV cache, BF16 MTP projections, one active sequence,
and a 262,144-token server ceiling.

## Requirements

- NVIDIA GB10 / DGX Spark with its NVIDIA container runtime working.
- Docker and the NVIDIA Container Toolkit.
- Enough free unified memory for the checkpoint, Atlas, KV cache, and runtime
  workspace. Stop other memory-heavy model or ComfyUI workloads first.
- A local copy of
  [`kingjones777/Ling-3.0-flash-NVFP4-SGLang-MTP`](https://huggingface.co/kingjones777/Ling-3.0-flash-NVFP4-SGLang-MTP).

The weights are not part of this repository.

## Build the targeted image

From the repository root:

```bash
docker build \
  --build-arg ATLAS_TARGET_HW=gb10 \
  --build-arg ATLAS_TARGET_MODEL=ling-3.0-flash \
  --build-arg ATLAS_TARGET_QUANT=nvfp4 \
  --build-arg ATLAS_GIT_SHA="$(git rev-parse HEAD)" \
  -f docker/gb10/Dockerfile \
  -t atlas-ling-gb10:latest .
```

Record the output image digest with:

```bash
docker image inspect atlas-ling-gb10:latest --format '{{index .RepoDigests 0}} {{.Id}}'
```

## Run the validated 262K configuration

Set `BIND_ADDR=127.0.0.1` for local-only access. To serve over Tailscale, set it
to the GB10's Tailscale address. Do not bind an unauthenticated Atlas endpoint
to the public internet.

```bash
MODEL_DIR=/absolute/path/to/Ling-3.0-flash-NVFP4-SGLang-MTP
BIND_ADDR=127.0.0.1

docker run --rm --name atlas-ling \
  --network host --gpus all --ipc=host \
  -v "${MODEL_DIR}:/model:ro" \
  atlas-ling-gb10:latest \
  serve --model-from-path /model \
    --kernel-target ling-3.0-flash \
    --model-name ling-3.0-flash-nvfp4-mtp \
    --bind "${BIND_ADDR}" \
    --port 8888 \
    --max-seq-len 262144 \
    --max-prefill-tokens 24576 \
    --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1 \
    --max-batch-size 1 \
    --lm-head-dtype bf16 \
    --speculative \
    --num-drafts 1 \
    --mtp-quantization bf16 \
    --mtp-gate force \
    --request-timeout 1800 \
    --no-auto-swap
```

`--num-drafts 1` gives a two-token verify width (K=2). The 262,144 value is a
server ceiling: real usable prompt length is lower by the requested output
budget and chat-template overhead. Long prompts also have substantial TTFT.

## Verify the endpoint

```bash
curl -fsS "http://${BIND_ADDR}:8888/v1/models"

curl -fsS "http://${BIND_ADDR}:8888/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ling-3.0-flash-nvfp4-mtp",
    "messages": [{"role": "user", "content": "Return only these three words and no explanation: ATLAS LING READY"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Successful startup should identify the native `ling-3.0-flash` kernel target,
load the Bailing MTP head, advertise the configured context ceiling, and return
a coherent completion. A healthy `/v1/models` response alone does not prove
generation; run the completion smoke test too.

## Tailscale clients

Use the same OpenAI-compatible base URL from another Tailscale device:

```text
http://<GB10_TAILSCALE_IP>:8888/v1
```

Keep Atlas bound to the Tailscale address and enforce access with Tailscale ACLs.

## Known scope

- Validated on one GB10 with tensor parallelism 1, not a multi-GPU topology.
- Main weights are NVFP4; the validated quality configuration keeps KV and MTP
  projections in BF16.
- The benchmark is a small, deterministic coding comparison, not a universal
  model ranking. See [the report](benchmarks/ling-gb10/README.md).
- The Mac-side Rust tests can skip CUDA compilation, but a production image and
  generation smoke test must be run on GB10 hardware.
