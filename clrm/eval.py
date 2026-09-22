from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _groups(metadata: Sequence[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        out[str(row["problem_id"])].append(i)
    return dict(out)


def tie_expected(labels: np.ndarray, scores: np.ndarray, rows: Sequence[int]) -> float:
    idx = np.asarray(rows, dtype=np.int64)
    s = scores[idx]
    best = np.max(s)
    winners = idx[np.isclose(s, best, rtol=0.0, atol=0.0)]
    return float(labels[winners].mean())


def pair_macro(metadata: Sequence[dict], scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray([int(x["label"]) for x in metadata], dtype=np.int8)
    problem_values = []
    correct = 0.0
    total = 0
    for rows in _groups(metadata).values():
        pos = [i for i in rows if labels[i] == 1]
        neg = [i for i in rows if labels[i] == 0]
        if not pos or not neg:
            continue
        vals = []
        for p in pos:
            for n in neg:
                v = 1.0 if scores[p] > scores[n] else 0.0 if scores[p] < scores[n] else 0.5
                vals.append(v)
                correct += v
                total += 1
        problem_values.append(float(np.mean(vals)))
    if not problem_values:
        raise ValueError("pair accuracy requires mixed-label problems")
    return float(np.mean(problem_values)), float(correct / total)


def candidate_auroc(metadata: Sequence[dict], scores: np.ndarray) -> float:
    labels = np.asarray([int(x["label"]) for x in metadata], dtype=np.int8)
    return float(roc_auc_score(labels, scores))


def load_subset_manifest(path: str | Path) -> dict[str, list[list[int]]]:
    raw = json.loads(Path(path).read_text())
    return {str(k): [[int(v) for v in subset] for subset in subsets] for k, subsets in raw.items()}


def best_at_k(
    metadata: Sequence[dict],
    scores: np.ndarray,
    subsets: Mapping[str, Sequence[Sequence[int]]],
) -> float:
    """Evaluate expected top-1 correctness over fixed candidate-index subsets.

    Subsets contain *candidate_index* values, not row offsets.
    """
    labels = np.asarray([int(x["label"]) for x in metadata], dtype=np.int8)
    groups = _groups(metadata)
    total = 0.0
    count = 0
    for problem_id, rows in groups.items():
        by_candidate = {int(metadata[i]["candidate_index"]): i for i in rows}
        for subset in subsets.get(problem_id, []):
            selected = [by_candidate[int(c)] for c in subset]
            total += tie_expected(labels, scores, selected)
            count += 1
    if count == 0:
        raise ValueError("subset manifest contains no evaluable subsets")
    return float(total / count)


def make_random_subsets(
    metadata: Sequence[dict],
    k: int = 8,
    repeats: int = 64,
    seed: int = 2024,
) -> dict[str, list[list[int]]]:
    """Convenience generator for exploratory evaluation.

    For exact paper reproduction use the fixed subset manifest distributed with the
    candidate artifacts rather than regenerating subsets.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[list[int]]] = {}
    for problem_id, rows in _groups(metadata).items():
        candidate_ids = np.asarray([int(metadata[i]["candidate_index"]) for i in rows], dtype=np.int64)
        if len(candidate_ids) < k:
            continue
        out[problem_id] = [
            sorted(rng.choice(candidate_ids, size=k, replace=False).tolist())
            for _ in range(repeats)
        ]
    return out
