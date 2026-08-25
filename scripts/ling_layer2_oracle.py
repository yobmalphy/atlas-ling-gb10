#!/usr/bin/env python3
"""Torch oracle for Ling 3.0's first sparse layer (layer 2).

The script evaluates the full dense prefix and layer-2 KDA, but dequantizes
only the eight experts selected for the final prompt token.  That keeps this
diagnostic small enough to run beside Atlas on a GB10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor

from ling_layer0_oracle import dense_prefix, load_tensors, rms_norm, short_conv


def kda_attention(root: Path, layer: int, x: torch.Tensor, device: str) -> torch.Tensor:
    p = f"model.layers.{layer}"
    a = f"{p}.attention"
    names = [
        f"{p}.input_layernorm.weight",
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

    tokens, heads, dim = x.shape[0], 32, 128
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
    return (residual + attn).to(torch.bfloat16)


def dequant_expert_projection(
    root: Path, layer: int, expert: int, projection: str, device: str
) -> torch.Tensor:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
    names = [prefix + suffix for suffix in ("weight_packed", "weight_scale", "weight_global_scale")]
    packed = load_tensors(root, names, device)
    local = {name.removeprefix(prefix): value for name, value in packed.items()}
    return NVFP4PackedCompressor.decompress(local, None)["weight"]


@torch.inference_mode()
def layer2(root: Path, prompt: str, device: str) -> tuple[list[int], torch.Tensor, list[int], list[float]]:
    ids, dense = dense_prefix(root, prompt, device, 2)
    hidden = kda_attention(root, 2, dense[-1], device)
    p = "model.layers.2"
    names = [
        f"{p}.post_attention_layernorm.weight",
        f"{p}.mlp.gate.weight",
        f"{p}.mlp.gate.expert_bias",
        f"{p}.mlp.shared_experts.gate_proj.weight",
        f"{p}.mlp.shared_experts.up_proj.weight",
        f"{p}.mlp.shared_experts.down_proj.weight",
    ]
    w = load_tensors(root, names, device)
    residual = hidden[-1]
    x = rms_norm(residual, w[f"{p}.post_attention_layernorm.weight"], 1e-6)

    logits = F.linear(x.float(), w[f"{p}.mlp.gate.weight"].float())
    scores = torch.sigmoid(logits)
    selection_scores = (scores + w[f"{p}.mlp.gate.expert_bias"]).view(8, 64)
    group_scores = selection_scores.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(4, sorted=False).indices
    mask = torch.zeros(8, dtype=torch.bool, device=device)
    mask[selected_groups] = True
    masked = (scores + w[f"{p}.mlp.gate.expert_bias"]).view(8, 64).masked_fill(
        ~mask[:, None], float("-inf")
    ).flatten()
    experts = masked.topk(8).indices
    weights = scores[experts]
    weights = weights / (weights.sum() + 1e-20) * 2.5

    sg = F.linear(x, w[f"{p}.mlp.shared_experts.gate_proj.weight"])
    su = F.linear(x, w[f"{p}.mlp.shared_experts.up_proj.weight"])
    shared = F.linear(
        F.silu(sg) * su, w[f"{p}.mlp.shared_experts.down_proj.weight"]
    )

    routed = torch.zeros(2560, dtype=torch.float32, device=device)
    for expert, weight in zip(experts.tolist(), weights):
        gate = dequant_expert_projection(root, 2, expert, "gate_proj", device)
        up = dequant_expert_projection(root, 2, expert, "up_proj", device)
        down = dequant_expert_projection(root, 2, expert, "down_proj", device)
        expert_output = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        routed.add_(expert_output.float() * weight)
    moe = routed.to(torch.bfloat16) + shared
    output = (residual + moe).to(torch.bfloat16)
    return ids, output, experts.tolist(), weights.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Reply with exactly: ATLAS LING READY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ids, output, experts, weights = layer2(args.model, args.prompt, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.float().cpu().numpy().astype("<f4").tofile(args.output)
    print(json.dumps({"tokens": len(ids), "ids": ids, "experts": experts, "weights": weights, "output": str(args.output)}))


if __name__ == "__main__":
    main()
