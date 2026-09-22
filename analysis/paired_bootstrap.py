#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def paired_bootstrap(delta: np.ndarray, reps: int = 10_000, seed: int = 2025) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 250):
        n = min(250, reps - start)
        idx = rng.integers(0, len(delta), size=(n, len(delta)))
        draws[start : start + n] = delta[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Two-sided sign-of-bootstrap p-value with a small finite-sample correction.
    left = (np.sum(draws <= 0) + 1) / (reps + 1)
    right = (np.sum(draws >= 0) + 1) / (reps + 1)
    p = min(1.0, 2.0 * min(left, right))
    return {"mean_delta": float(delta.mean()), "ci95": [float(lo), float(hi)], "p_raw": float(p)}


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(1.0, running)
    return out.tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="Paired problem bootstrap from two .npy vectors")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()
    a, b = np.load(args.a), np.load(args.b)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have identical shapes")
    print(json.dumps(paired_bootstrap(a - b, args.reps, args.seed), indent=2))


if __name__ == "__main__":
    main()
