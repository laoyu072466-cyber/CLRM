from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import HiddenStateDataset, build_pairs, collate_candidates
from .eval import pair_macro


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pair_batch_size: int = 32
    max_epochs: int = 5
    patience: int = 2
    grad_clip: float = 1.0
    seed: int = 42
    score_batch_size: int = 32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_dataset(
    model: nn.Module,
    dataset: HiddenStateDataset,
    depths: Sequence[int],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    out = np.empty(len(dataset), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            end = min(len(dataset), start + batch_size)
            hs, mask = collate_candidates(dataset, list(range(start, end)), depths)
            hs = {d: x.to(device=device, non_blocking=True) for d, x in hs.items()}
            values = model(hs, mask.to(device=device, non_blocking=True))["score"]
            out[start:end] = values.detach().cpu().double().numpy()
    return out


def train_pairwise(
    model: nn.Module,
    train_data: HiddenStateDataset,
    val_data: HiddenStateDataset,
    depths: Sequence[int],
    output_path: str | Path,
    config: TrainConfig,
    device: str | torch.device = "cuda",
) -> dict:
    """Train with pairwise logistic ranking loss and select by validation Pair-Macro."""
    seed_everything(config.seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_pairs = build_pairs(train_data.metadata)

    best_macro = -math.inf
    best_epoch = 0
    stale = 0
    history = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        shuffled = list(train_pairs)
        random.Random(config.seed + epoch * 1_000_003).shuffle(shuffled)
        model.train()
        running_loss = 0.0
        seen = 0

        for start in range(0, len(shuffled), config.pair_batch_size):
            batch = shuffled[start : start + config.pair_batch_size]
            pos_rows = [p for p, _, _ in batch]
            neg_rows = [n for _, n, _ in batch]
            rows = pos_rows + neg_rows
            hs, mask = collate_candidates(train_data, rows, depths)
            hs = {d: x.to(device=device, non_blocking=True) for d, x in hs.items()}
            mask = mask.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            scores = model(hs, mask)["score"]
            n = len(batch)
            loss = F.softplus(-(scores[:n] - scores[n:])).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * n
            seen += n

        val_scores = score_dataset(model, val_data, depths, device, config.score_batch_size)
        val_macro, val_micro = pair_macro(val_data.metadata, val_scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_pair_macro": val_macro,
                "val_pair_micro": val_micro,
            }
        )

        if val_macro > best_macro:
            best_macro = val_macro
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_pair_macro": val_macro,
                    "depths": [int(d) for d in depths],
                    "train_config": vars(config),
                },
                output_path,
            )
        else:
            stale += 1
        if stale >= config.patience:
            break

    return {
        "best_epoch": best_epoch,
        "best_val_pair_macro": best_macro,
        "epochs_completed": len(history),
        "history": history,
        "checkpoint": str(output_path),
    }
