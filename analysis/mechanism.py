#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from clrm.data import HiddenStateDataset
from clrm.models import ResidualOnly


def geometry(h0: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0 = np.asarray(h0, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    d1 = h1 - h0
    d2 = h2 - h1
    n0 = np.linalg.norm(h0, axis=1)
    n1 = np.linalg.norm(h1, axis=1)
    n2 = np.linalg.norm(h2, axis=1)
    nd1 = np.linalg.norm(d1, axis=1)
    nd2 = np.linalg.norm(d2, axis=1)
    m1 = nd1 / (0.5 * (n0 + n1) + 1e-12)
    m2 = nd2 / (0.5 * (n1 + n2) + 1e-12)
    cos = (d1 * d2).sum(axis=1) / (nd1 * nd2 + 1e-12)
    return m1, m2, np.clip(cos, -1.0, 1.0)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.clip(
        (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12),
        -1.0,
        1.0,
    )


def bootstrap_mean(values: Sequence[float], reps: int = 10_000, seed: int = 2025) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 250):
        n = min(250, reps - start)
        idx = rng.integers(0, len(values), size=(n, len(values)))
        draws[start : start + n] = values[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def problem_contrasts(rows: Sequence[dict], field: str) -> np.ndarray:
    groups: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
    for row in rows:
        groups[str(row["problem_id"])][int(row["label"])].append(float(row[field]))
    out = []
    for neg, pos in groups.values():
        if neg and pos:
            out.append(float(np.mean(pos) - np.mean(neg)))
    return np.asarray(out, dtype=np.float64)


def _snr_by_problem(vectors: dict[str, list[list[np.ndarray]]]) -> np.ndarray:
    snrs = []
    for neg, pos in vectors.values():
        if not neg or not pos:
            continue
        neg = np.stack(neg)
        pos = np.stack(pos)
        dim = pos.shape[1]
        mn, mp = neg.mean(axis=0), pos.mean(axis=0)
        separation = np.linalg.norm(mp - mn) / np.sqrt(dim)
        dn = np.mean(np.sum((neg - mn) ** 2, axis=1)) / dim
        dp = np.mean(np.sum((pos - mp) ** 2, axis=1)) / dim
        dispersion = np.sqrt(0.5 * (dn + dp))
        snrs.append(separation / (dispersion + 1e-12))
    return np.asarray(snrs, dtype=np.float64)


def projected_alignment_model(checkpoint: str | None, hidden_size: int, depths: Sequence[int]) -> ResidualOnly | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ResidualOnly(hidden_size, depths)
    state = payload.get("model_state_dict", payload.get("head_state_dict"))
    if state is None:
        raise KeyError("checkpoint has no model_state_dict/head_state_dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        remap = {}
        for k, v in state.items():
            nk = k.replace("proj_ln.", "projector.norm.").replace("proj.", "projector.proj.").replace("mlp.", "readout.")
            remap[nk] = v
        model.load_state_dict(remap, strict=True)
    model.eval()
    return model


def analyze(cache_dir: str, depths: Sequence[int], checkpoint: str | None = None) -> dict:
    data = HiddenStateDataset(cache_dir)
    e, m, f = [int(x) for x in depths]
    if len(depths) != 3:
        raise ValueError("mechanism analysis expects exactly three depths")
    proj_model = projected_alignment_model(checkpoint, data.hidden_size, depths)

    candidate_rows = []
    state_vectors = {d: defaultdict(lambda: [[], []]) for d in depths}
    residual_vectors = {"r1": defaultdict(lambda: [[], []]), "r2": defaultdict(lambda: [[], []])}

    for i, meta in enumerate(data.metadata):
        hs = data.load_states(i)
        h0 = hs[e].numpy()
        h1 = hs[m].numpy()
        h2 = hs[f].numpy()
        m1, m2, coh = geometry(h0, h1, h2)
        row = {
            "problem_id": str(meta["problem_id"]),
            "candidate_index": int(meta["candidate_index"]),
            "label": int(meta["label"]),
            "response_length": len(h0),
            "mean_m1": float(m1.mean()),
            "mean_m2": float(m2.mean()),
            "mean_coherence": float(coh.mean()),
            "raw_cos12": float(cosine_rows(h0, h1).mean()),
            "raw_cos23": float(cosine_rows(h1, h2).mean()),
            "raw_cos13": float(cosine_rows(h0, h2).mean()),
        }
        if proj_model is not None:
            with torch.inference_mode():
                z0 = proj_model.projected(hs[e]).numpy()
                z1 = proj_model.projected(hs[m]).numpy()
                z2 = proj_model.projected(hs[f]).numpy()
            row.update(
                {
                    "projected_cos12": float(cosine_rows(z0, z1).mean()),
                    "projected_cos23": float(cosine_rows(z1, z2).mean()),
                    "projected_cos13": float(cosine_rows(z0, z2).mean()),
                }
            )
        candidate_rows.append(row)

        pid = str(meta["problem_id"])
        label = int(meta["label"])
        for d in depths:
            state_vectors[d][pid][label].append(hs[d].numpy().mean(axis=0).astype(np.float64))
        residual_vectors["r1"][pid][label].append((h1 - h0).mean(axis=0).astype(np.float64))
        residual_vectors["r2"][pid][label].append((h2 - h1).mean(axis=0).astype(np.float64))

    summaries = {}
    fields = ["mean_m1", "mean_m2", "mean_coherence", "raw_cos12", "raw_cos23", "raw_cos13"]
    if checkpoint:
        fields += ["projected_cos12", "projected_cos23", "projected_cos13"]
    for j, field in enumerate(fields):
        effects = problem_contrasts(candidate_rows, field)
        mean, lo, hi = bootstrap_mean(effects, seed=2025 + j)
        summaries[field] = {"correct_minus_incorrect": mean, "ci95": [lo, hi], "n_problems": len(effects)}

    state_snr = {}
    for j, d in enumerate(depths):
        values = _snr_by_problem(state_vectors[d])
        mean, lo, hi = bootstrap_mean(values, seed=2200 + j)
        state_snr[str(d)] = {"mean": mean, "ci95": [lo, hi], "n_problems": len(values)}

    residual_snr = {}
    for j, name in enumerate(("r1", "r2")):
        values = _snr_by_problem(residual_vectors[name])
        mean, lo, hi = bootstrap_mean(values, seed=2300 + j)
        residual_snr[name] = {"mean": mean, "ci95": [lo, hi], "n_problems": len(values)}

    return {
        "depths": list(depths),
        "candidate_rows": candidate_rows,
        "contrasts": summaries,
        "state_snr": state_snr,
        "residual_snr": residual_snr,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--depths", nargs=3, type=int, required=True)
    p.add_argument("--residual-checkpoint", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = analyze(args.cache, args.depths, args.residual_checkpoint)
    candidate_rows = result.pop("candidate_rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = output.with_suffix(".candidates.csv")
    if candidate_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(candidate_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
