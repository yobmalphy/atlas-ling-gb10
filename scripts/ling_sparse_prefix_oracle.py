#!/usr/bin/env python3
"""Evaluate Ling's dense prefix plus complete sparse KDA layers 2-4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ling_layer0_oracle import dense_prefix, load_tensors, rms_norm
from ling_layer2_oracle import dequant_expert_projection, kda_attention


@torch.inference_mode()
def sparse_moe(root: Path, layer: int, hidden: torch.Tensor, device: str):
    p = f"model.layers.{layer}"
    names = [
        f"{p}.post_attention_layernorm.weight",
        f"{p}.mlp.gate.weight",
        f"{p}.mlp.gate.expert_bias",
        f"{p}.mlp.shared_experts.gate_proj.weight",
        f"{p}.mlp.shared_experts.up_proj.weight",
        f"{p}.mlp.shared_experts.down_proj.weight",
    ]
    w = load_tensors(root, names, device)
    x = rms_norm(hidden, w[f"{p}.post_attention_layernorm.weight"], 1e-6)
    logits = F.linear(x.float(), w[f"{p}.mlp.gate.weight"].float())
    scores = torch.sigmoid(logits)
    biased = scores + w[f"{p}.mlp.gate.expert_bias"]
    group_scores = biased.view(-1, 8, 64).topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(4, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    masked = biased.view(-1, 8, 64).masked_fill(~group_mask[:, :, None], float("-inf")).flatten(1)
    experts = masked.topk(8, dim=-1).indices
    weights = scores.gather(1, experts)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * 2.5

    routed = torch.zeros_like(x, dtype=torch.float32)
    for expert in experts.unique().tolist():
        occurrences = (experts == expert).nonzero(as_tuple=False)
        token_indices = occurrences[:, 0]
        slots = occurrences[:, 1]
        gate = dequant_expert_projection(root, layer, expert, "gate_proj", device)
        up = dequant_expert_projection(root, layer, expert, "up_proj", device)
        down = dequant_expert_projection(root, layer, expert, "down_proj", device)
        selected_x = x[token_indices]
        expert_output = F.linear(
            F.silu(F.linear(selected_x, gate)) * F.linear(selected_x, up), down
        )
        routed.index_add_(
            0, token_indices, expert_output.float() * weights[token_indices, slots, None]
        )
        del gate, up, down, selected_x, expert_output

    sg = F.linear(x, w[f"{p}.mlp.shared_experts.gate_proj.weight"])
    su = F.linear(x, w[f"{p}.mlp.shared_experts.up_proj.weight"])
    shared = F.linear(
        F.silu(sg) * su, w[f"{p}.mlp.shared_experts.down_proj.weight"]
    )
    output = (hidden + routed.to(torch.bfloat16) + shared).to(torch.bfloat16)
    return output, experts, weights


@torch.inference_mode()
def sparse_prefix(
    root: Path,
    prompt: str,
    device: str,
    last_layer: int,
    append_tokens: list[int] | None = None,
):
    ids, dense = dense_prefix(root, prompt, device, 2, append_tokens)
    hidden = dense[-1]
    outputs = []
    routing = {}
    for layer in range(2, last_layer + 1):
        hidden = kda_attention(root, layer, hidden, device)
        hidden, experts, weights = sparse_moe(root, layer, hidden, device)
        outputs.append(hidden)
        routing[layer] = {
            "final_experts": experts[-1].tolist(),
            "final_weights": weights[-1].tolist(),
            "unique_experts": experts.unique().numel(),
        }
    return ids, outputs, routing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Reply with exactly: ATLAS LING READY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-layer", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ids, outputs, routing = sparse_prefix(args.model, args.prompt, args.device, args.last_layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for layer, hidden in enumerate(outputs, start=2):
        output = args.output_dir / f"oracle_L{layer}.bin"
        hidden[-1].float().cpu().numpy().astype("<f4").tofile(output)
        written.append(str(output))
    print(json.dumps({"tokens": len(ids), "ids": ids, "routing": routing, "outputs": written}))


if __name__ == "__main__":
    main()
