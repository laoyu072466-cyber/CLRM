#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clrm.data import HiddenStateDataset
from clrm.models import build_model
from clrm.train import TrainConfig, train_pairwise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["md_state", "single_state", "multi_endpoint", "multi_diff", "residual_only", "concat", "weighted"])
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--depths", required=True, nargs="+", type=int)
    p.add_argument("--output", required=True, help="Checkpoint path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pair-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    train_data = HiddenStateDataset(args.train_cache)
    val_data = HiddenStateDataset(args.val_cache)
    if train_data.hidden_size != val_data.hidden_size:
        raise ValueError("train/val hidden sizes differ")

    model = build_model(args.model, train_data.hidden_size, args.depths)
    cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        pair_batch_size=args.pair_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )
    result = train_pairwise(model, train_data, val_data, args.depths, args.output, cfg, args.device)

    ckpt = torch.load(args.output, map_location="cpu", weights_only=False)
    ckpt.update(
        {
            "model_name": args.model,
            "hidden_size": train_data.hidden_size,
            "depths": list(args.depths),
        }
    )
    torch.save(ckpt, args.output)

    result_path = Path(args.output).with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
