#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from clrm.data import HiddenStateDataset
from clrm.eval import best_at_k, candidate_auroc, load_subset_manifest, make_random_subsets, pair_macro
from clrm.models import build_model
from clrm.train import score_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--subset-manifest", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--random-subset-repeats", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = HiddenStateDataset(args.cache)
    model = build_model(ckpt["model_name"], int(ckpt["hidden_size"]), ckpt["depths"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scores = score_dataset(model.to(device), dataset, ckpt["depths"], device)

    macro, micro = pair_macro(dataset.metadata, scores)
    subsets = (
        load_subset_manifest(args.subset_manifest)
        if args.subset_manifest
        else make_random_subsets(dataset.metadata, args.k, args.random_subset_repeats)
    )
    metrics = {
        f"best_at_{args.k}": best_at_k(dataset.metadata, scores, subsets),
        "pair_macro": macro,
        "pair_micro": micro,
        "candidate_auroc": candidate_auroc(dataset.metadata, scores),
        "subset_source": "fixed_manifest" if args.subset_manifest else "generated_for_exploration",
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
