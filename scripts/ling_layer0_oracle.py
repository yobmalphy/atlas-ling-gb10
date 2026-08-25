#!/usr/bin/env python3
"""Small Torch oracle for Ling 3.0's BF16 dense layers 0 and 1.

This intentionally loads only embeddings and layer-0 tensors. It is a
diagnostic reference for the new Atlas KDA path, not a serving implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


def load_tensors(root: Path, names: list[str], device: str) -> dict[str, torch.Tensor]:
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for filename, file_names in by_file.items():
        with safe_open(root / filename, framework="pt", device=device) as handle:
            for name in file_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized.to(x.dtype) * weight).to(x.dtype)


def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # ShortConvolution is a causal depthwise Conv1d. PyTorch's correlation
    # order maps weight[..., -1] to the current token, matching Atlas' state
    # shift followed by dot(state[0..K], weight[0..K]).
    channels = x.shape[-1]
    x_ch_first = x.transpose(0, 1).unsqueeze(0)
    padded = F.pad(x_ch_first, (weight.shape[-1] - 1, 0))
    out = F.conv1d(padded, weight, groups=channels)
    return F.silu(out.squeeze(0).transpose(0, 1))


def dense_kda_layer(
    root: Path, layer: int, x: torch.Tensor, device: str
) -> torch.Tensor:
    p = f"model.layers.{layer}"
    a = f"{p}.attention"
    names = [
        "model.word_embeddings.weight",
        f"{p}.input_layernorm.weight",
        f"{p}.post_attention_layernorm.weight",
        f"{a}.q_proj.weight",
        f"{a}.k_proj.weight",
        f"{a}.v_proj.weight",
        f"{a}.q_conv1d.weight",
        f"{a}.k_conv1d.weight",
        f"{a}.v_conv1d.weight",
        f"{a}.f_proj.weight",
        f"{a}.b_proj.weight",
        f"{a}.A_log",
        f"{a}.dt_bias",
        f"{a}.g_proj.weight",
        f"{a}.o_norm.weight",
        f"{a}.o_proj.weight",
        f"{p}.mlp.gate_proj.weight",
        f"{p}.mlp.up_proj.weight",
        f"{p}.mlp.down_proj.weight",
    ]
    w = load_tensors(root, names, device)
    residual = x
    normed = rms_norm(x, w[f"{p}.input_layernorm.weight"], 1e-6)

    q = short_conv(F.linear(normed, w[f"{a}.q_proj.weight"]), w[f"{a}.q_conv1d.weight"])
    k = short_conv(F.linear(normed, w[f"{a}.k_proj.weight"]), w[f"{a}.k_conv1d.weight"])
    v = short_conv(F.linear(normed, w[f"{a}.v_proj.weight"]), w[f"{a}.v_conv1d.weight"])
    f_raw = F.linear(normed, w[f"{a}.f_proj.weight"])
    beta = torch.sigmoid(F.linear(normed, w[f"{a}.b_proj.weight"]).float())
    output_gate = F.linear(normed, w[f"{a}.g_proj.weight"])

    tokens = x.shape[0]
    heads = 32
    dim = 128
    q = q.view(tokens, heads, dim).float()
    k = k.view(tokens, heads, dim).float()
    v = v.view(tokens, heads, dim).float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    f_raw = f_raw.view(tokens, heads, dim).float()
    output_gate = output_gate.view(tokens, heads, dim)
    a_rate = w[f"{a}.A_log"].float().exp().view(heads, 1)
    dt_bias = w[f"{a}.dt_bias"].float().view(heads, dim)
    state = torch.zeros((heads, dim, dim), dtype=torch.float32, device=device)
    outputs = []
    for token in range(tokens):
        log_decay = -5.0 * torch.sigmoid(a_rate * (f_raw[token] + dt_bias))
        state.mul_(torch.exp(log_decay).unsqueeze(-1))
        delta = v[token] - torch.einsum("hkv,hk->hv", state, k[token])
        delta.mul_(beta[token].unsqueeze(-1))
        state.add_(k[token].unsqueeze(-1) * delta.unsqueeze(-2))
        outputs.append(torch.einsum("hkv,hk->hv", state, q[token]) / dim**0.5)
    attn = torch.stack(outputs).to(torch.bfloat16)
    attn = rms_norm(attn, w[f"{a}.o_norm.weight"], 1e-6)
    attn = (attn.float() * torch.sigmoid(output_gate.float())).to(torch.bfloat16)
    attn = F.linear(attn.flatten(1), w[f"{a}.o_proj.weight"])

    hidden = (residual + attn).to(torch.bfloat16)
    normed = rms_norm(hidden, w[f"{p}.post_attention_layernorm.weight"], 1e-6)
    gate = F.linear(normed, w[f"{p}.mlp.gate_proj.weight"])
    up = F.linear(normed, w[f"{p}.mlp.up_proj.weight"])
    mlp = F.linear(F.silu(gate) * up, w[f"{p}.mlp.down_proj.weight"])
    hidden = (hidden + mlp).to(torch.bfloat16)
    return hidden


@torch.inference_mode()
def dense_prefix(
    root: Path,
    prompt: str,
    device: str,
    num_layers: int,
    append_tokens: list[int] | None = None,
) -> tuple[list[int], list[torch.Tensor]]:
    if num_layers < 1 or num_layers > 2:
        raise ValueError("Ling's dense prefix contains exactly layers 0 and 1")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if append_tokens:
        ids.extend(append_tokens)
    embeddings = load_tensors(root, ["model.word_embeddings.weight"], device)
    hidden = embeddings["model.word_embeddings.weight"][torch.tensor(ids, device=device)]
    outputs = []
    for layer in range(num_layers):
        hidden = dense_kda_layer(root, layer, hidden, device)
        outputs.append(hidden)
    return ids, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Reply with exactly: ATLAS LING READY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    ids, outputs = dense_prefix(args.model, args.prompt, args.device, args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for layer, hidden in enumerate(outputs):
        output = args.output.with_name(f"{args.output.stem}_L{layer}{args.output.suffix}")
        hidden[-1].float().cpu().numpy().astype("<f4").tofile(output)
        written.append(str(output))
    print(json.dumps({"tokens": len(ids), "ids": ids, "outputs": written}))


if __name__ == "__main__":
    main()
