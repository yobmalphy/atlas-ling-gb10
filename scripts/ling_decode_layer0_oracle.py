#!/usr/bin/env python3
"""Reference layer-0 activation for the first autoregressive decode token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ling_layer0_oracle import dense_kda_layer, load_tensors


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Reply with exactly: ATLAS LING READY")
    parser.add_argument("--append-token", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(rendered, add_special_tokens=False).input_ids + [args.append_token]
    embedding = load_tensors(args.model, ["model.word_embeddings.weight"], args.device)
    hidden = embedding["model.word_embeddings.weight"][torch.tensor(ids, device=args.device)]
    hidden = dense_kda_layer(args.model, 0, hidden, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hidden[-1].float().cpu().numpy().astype("<f4").tofile(args.output)
    print(json.dumps({"tokens": len(ids), "ids": ids, "output": str(args.output)}))


if __name__ == "__main__":
    main()
