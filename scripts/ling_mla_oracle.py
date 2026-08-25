#!/usr/bin/env python3
"""Torch oracle through Ling's first MLA layer (layer 5)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ling_layer0_oracle import load_tensors, rms_norm
from ling_sparse_prefix_oracle import sparse_moe, sparse_prefix


def interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Checkpoint weights store adjacent rotary pairs.  The reference rearranges
    # [even, odd] pairs into [all-even, all-odd] before rotate_half.
    shape = x.shape
    x = x.view(*shape[:-1], shape[-1] // 2, 2).transpose(-1, -2).reshape(shape)
    half = shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


@torch.inference_mode()
def mla_attention(root: Path, layer: int, hidden: torch.Tensor, device: str) -> torch.Tensor:
    p = f"model.layers.{layer}"
    a = f"{p}.attention"
    names = [
        f"{p}.input_layernorm.weight",
        f"{a}.q_proj.weight",
        f"{a}.kv_a_proj_with_mqa.weight",
        f"{a}.kv_a_layernorm.weight",
        f"{a}.kv_b_proj.weight",
        f"{a}.g_proj.weight",
        f"{a}.dense.weight",
    ]
    w = load_tensors(root, names, device)
    residual = hidden
    x = rms_norm(hidden, w[f"{p}.input_layernorm.weight"], 1e-6)
    tokens = x.shape[0]
    q = F.linear(x, w[f"{a}.q_proj.weight"]).view(tokens, 32, 192)
    q_nope, q_rope = q.split((128, 64), dim=-1)

    compressed = F.linear(x, w[f"{a}.kv_a_proj_with_mqa.weight"])
    kv_latent, k_rope = compressed.split((512, 64), dim=-1)
    kv_latent = rms_norm(kv_latent, w[f"{a}.kv_a_layernorm.weight"], 1e-6)
    expanded = F.linear(kv_latent, w[f"{a}.kv_b_proj.weight"]).view(tokens, 32, 256)
    k_nope, value = expanded.split((128, 128), dim=-1)

    positions = torch.arange(tokens, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (
        6_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)[:, None, :]
    sin = emb.sin().to(x.dtype)[:, None, :]
    q_rope = interleaved_rope(q_rope, cos, sin)
    k_rope = interleaved_rope(k_rope[:, None, :], cos, sin).expand(-1, 32, -1)
    query = torch.cat((q_nope, q_rope), dim=-1).transpose(0, 1)
    key = torch.cat((k_nope, k_rope), dim=-1).transpose(0, 1)
    value = value.transpose(0, 1)

    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(192)
    causal = torch.ones((tokens, tokens), dtype=torch.bool, device=device).triu(1)
    scores.masked_fill_(causal, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probs, value).transpose(0, 1)
    gate_logits = F.linear(x, w[f"{a}.g_proj.weight"])
    gate = torch.sigmoid(gate_logits.float()).to(x.dtype)
    output = output * gate[:, :, None]
    output = F.linear(output.flatten(1), w[f"{a}.dense.weight"])
    if os.environ.get("LING_MLA_DIAG") == "1":
        kv_b = w[f"{a}.kv_b_proj.weight"].view(32, 256, 512)
        w_k = kv_b[:, :128, :]
        w_v = kv_b[:, 128:, :]
        q_abs_nope = torch.einsum(
            "thp,hpl->thl", q_nope.float(), w_k.float()
        ).to(torch.bfloat16)
        q_absorbed = torch.cat((q_abs_nope, q_rope), dim=-1).transpose(0, 1)
        compressed_k = torch.cat(
            (kv_latent[:, None, :].expand(-1, 32, -1), k_rope), dim=-1
        ).transpose(0, 1)
        compressed_v = torch.cat(
            (kv_latent, torch.zeros((tokens, 64), dtype=kv_latent.dtype, device=device)),
            dim=-1,
        )[None, :, :].expand(32, -1, -1)
        absorbed_scores = torch.matmul(q_absorbed, compressed_k.transpose(-1, -2)) / math.sqrt(192)
        absorbed_scores.masked_fill_(causal, float("-inf"))
        absorbed_probs = F.softmax(absorbed_scores, dim=-1, dtype=torch.float32).to(q_absorbed.dtype)
        attn_latent = torch.matmul(absorbed_probs, compressed_v)
        v_extracted = torch.einsum(
            "htl,hvl->htv", attn_latent[..., :512].float(), w_v.float()
        ).to(torch.bfloat16).transpose(0, 1)
        o_absorbed = F.linear(
            (v_extracted * gate[:, :, None]).flatten(1), w[f"{a}.dense.weight"]
        )
        last_token_norm = lambda tensor: float(tensor[-1].float().norm())
        print(json.dumps({
            "mla_diag": {
                "q_full": last_token_norm(q),
                "q_absorbed_no_rope": last_token_norm(q_abs_nope),
                "q_absorbed_rope": last_token_norm(q_absorbed.transpose(0, 1)),
                "kv_latent": last_token_norm(kv_latent),
                "k_rope": last_token_norm(k_rope[:, 0]),
                "attn_latent": last_token_norm(attn_latent.transpose(0, 1)),
                "v_extracted": last_token_norm(v_extracted),
                "head_gate_logits": last_token_norm(gate_logits),
                "head_gate_sigmoid": last_token_norm(gate),
                "v_gated": last_token_norm(v_extracted * gate[:, :, None]),
                "o_out": last_token_norm(o_absorbed),
                "direct_o_out": last_token_norm(output),
                "absorbed_direct_cos": float(F.cosine_similarity(
                    o_absorbed[-1].float(), output[-1].float(), dim=0
                )),
            }
        }))
    return (residual + output).to(torch.bfloat16)


@torch.inference_mode()
def layer5(root: Path, prompt: str, device: str, append_tokens: list[int] | None = None):
    ids, sparse, _ = sparse_prefix(root, prompt, device, 4, append_tokens)
    after_attention = mla_attention(root, 5, sparse[-1], device)
    output, experts, weights = sparse_moe(root, 5, after_attention[-1:], device)
    return ids, output[-1], experts[-1].tolist(), weights[-1].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Reply with exactly: ATLAS LING READY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--append-token", type=int)
    args = parser.parse_args()
    append_tokens = [args.append_token] if args.append_token is not None else None
    ids, output, experts, weights = layer5(
        args.model, args.prompt, args.device, append_tokens
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.float().cpu().numpy().astype("<f4").tofile(args.output)
    print(json.dumps({"tokens": len(ids), "ids": ids, "experts": experts, "weights": weights, "output": str(args.output)}))


if __name__ == "__main__":
    main()
