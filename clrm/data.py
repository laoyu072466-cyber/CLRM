from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch


class HiddenStateDataset:
    """Lazy dataset for extracted response-token hidden states.

    Directory layout::

        cache_dir/
          metadata.jsonl
          states/
            00000000.pt
            00000001.pt
            ...

    Each state file stores a dict ``{depth: Tensor[T, D]}``.
    """

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        metadata_path = self.root / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        if not self.metadata:
            raise ValueError("empty metadata")

    def __len__(self) -> int:
        return len(self.metadata)

    def load_states(self, row: int) -> dict[int, torch.Tensor]:
        path = self.root / "states" / f"{row:08d}.pt"
        raw = torch.load(path, map_location="cpu", weights_only=False)
        states = {int(k): v.to(torch.float32) for k, v in raw.items()}
        if not states:
            raise ValueError(f"empty state file: {path}")
        return states

    @property
    def hidden_size(self) -> int:
        states = self.load_states(0)
        return int(next(iter(states.values())).shape[-1])


def build_problem_groups(metadata: Sequence[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[str(row["problem_id"])].append(i)
    return dict(groups)


def build_pairs(metadata: Sequence[dict]) -> list[tuple[int, int, str]]:
    """Build all within-problem positive/negative candidate pairs."""
    pairs: list[tuple[int, int, str]] = []
    for problem_id, rows in build_problem_groups(metadata).items():
        pos = [i for i in rows if int(metadata[i]["label"]) == 1]
        neg = [i for i in rows if int(metadata[i]["label"]) == 0]
        pairs.extend((p, n, problem_id) for p in pos for n in neg)
    if not pairs:
        raise ValueError("no mixed-label problems available for pairwise training")
    return pairs


def collate_candidates(
    dataset: HiddenStateDataset,
    rows: Sequence[int],
    depths: Sequence[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    examples = [dataset.load_states(i) for i in rows]
    lengths = []
    for ex in examples:
        missing = [d for d in depths if int(d) not in ex]
        if missing:
            raise KeyError(f"missing depths {missing}")
        lengths.append(int(ex[int(depths[0])].shape[0]))
    if min(lengths) < 1:
        raise ValueError("empty response sequence")
    max_len = max(lengths)
    hidden_size = int(examples[0][int(depths[0])].shape[-1])
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    hs = {
        int(d): torch.zeros((len(rows), max_len, hidden_size), dtype=torch.float32)
        for d in depths
    }
    for b, (ex, n) in enumerate(zip(examples, lengths)):
        mask[b, :n] = True
        for d in depths:
            value = ex[int(d)]
            if value.shape[0] != n or value.shape[-1] != hidden_size:
                raise ValueError("inconsistent hidden-state shape across depths")
            hs[int(d)][b, :n] = value
    return hs, mask
