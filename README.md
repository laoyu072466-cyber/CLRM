# CLRM

Code for our study of **where verification signal lives in LLM reasoning representations**.

The repository centers on **MD-State (Multi-Depth State Set Verifier)**, a deliberately simple verifier that summarizes response-token hidden-state distributions at several Transformer depths. It also includes the controlled representation variants and mechanism analyses used to compare:

- endpoint representations;
- trajectory-level absolute state;
- token-axis adjacent changes;
- matched-token transformations across depth.

The main scientific observation is that correctness is readable from both trajectory state and depth transformation, while their relative utility can differ across backbone families.

> Paper manuscript: in preparation.

## Repository layout

```text
clrm/
  models.py       # MD-State and controlled representation variants
  data.py         # lazy hidden-state cache and pair construction
  train.py        # pairwise ranking training
  eval.py         # Best@k, Pair-Macro, AUROC
  extract.py      # hidden-state extraction from tokenized candidates
scripts/
  extract_hidden_states.py
  train_verifier.py
  evaluate_verifier.py
analysis/
  mechanism.py            # transformation coherence, alignment, State/Residual SNR
  paired_bootstrap.py     # paired problem bootstrap + Holm helper
configs/
  qwen2.5-math-1.5b.json
  qwen2.5-math-7b.json
  llama-3.2-3b.json
  llama-3.1-8b.json
tests/
```

## Implemented verifier representations

All multi-depth models use a shared projection across sampled depths.

| CLI name | Representation |
|---|---|
| `md_state` | response-token State Set at three depths (paper method) |
| `single_state` | State Set at one depth |
| `multi_endpoint` | final response-token state from three depths |
| `multi_diff` | adjacent-token differences at three depths |
| `residual_only` | matched-token early→middle and middle→final depth transformations |
| `concat` | State + Residual feature concatenation |
| `weighted` | globally weighted State/Residual fusion |

### MD-State

For response-token hidden state \(h_t^{(\ell)}\), MD-State computes

\[
z_t^{(\ell)}=\mathrm{LN}(Wh_t^{(\ell)}),
\qquad
u_t^{(\ell)}=\phi(z_t^{(\ell)}),
\]

then pools the mean and standard deviation over valid response tokens at each depth:

\[
g_\ell=[\mu_t(u_t^{(\ell)});\sigma_t(u_t^{(\ell)})].
\]

The three depth summaries are concatenated and scored by a small MLP.

Default architecture used in the experiments:

- shared projection: `hidden_size -> 128`, no bias;
- LayerNorm;
- shared token map: `128 -> 32 -> ReLU`;
- per-depth mean/std pooling;
- final readout: `192 -> 96 -> 1`.

## Backbones and sampled depths

The paper uses the following sampled Transformer blocks:

| Backbone | Hidden size | Depths |
|---|---:|---|
| Qwen2.5-Math-1.5B | 1536 | 14, 21, 28 |
| Qwen2.5-Math-7B | 3584 | 14, 21, 28 |
| Llama-3.2-3B | 3072 | 14, 21, 28 |
| Llama-3.1-8B | 4096 | 16, 24, 32 |

The repository intentionally keeps model download paths separate from these paper-facing configurations so that local or Hugging Face checkpoints can be used.

## Installation

Python 3.10+ is recommended.

```bash
pip install -e .
```

## Input format

The extractor consumes **pre-tokenized** candidate JSONL. Tokenization and chat-template rendering are intentionally external because exact prompt templates are experiment-specific and changing them can change response-token boundaries.

Each row must contain:

```json
{
  "problem_id": "example-001",
  "candidate_index": 0,
  "label": 1,
  "input_ids": [1, 2, 3, 4],
  "response_start": 2
}
```

`response_start` is the index of the first response token in `input_ids`.

Extract hidden states:

```bash
python scripts/extract_hidden_states.py \
  --model /path/to/model \
  --input tokenized_train.jsonl \
  --output caches/train \
  --depths 14 21 28
```

The resulting cache is:

```text
caches/train/
  metadata.jsonl
  states/
    00000000.pt
    00000001.pt
    ...
```

Each state file contains a dictionary `{depth: [response_tokens, hidden_size] tensor}`.

## Train MD-State

```bash
python scripts/train_verifier.py \
  --model md_state \
  --train-cache caches/train \
  --val-cache caches/val \
  --depths 14 21 28 \
  --seed 42 \
  --output checkpoints/md_state_seed42.pt
```

Default training recipe:

- pairwise logistic ranking loss `softplus(-(s_pos - s_neg))`;
- AdamW;
- learning rate `1e-3`;
- weight decay `1e-4`;
- pair batch size `32`;
- max 5 epochs;
- patience 2;
- gradient clipping `1.0`;
- checkpoint selection by validation Pair-Macro.

The paper uses seeds `42`, `123`, and `456`.

For the reported setup, GSM8K and MATH are trained independently. SVAMP is evaluated zero-shot using the same-backbone GSM8K-trained verifier.

## Evaluation

```bash
python scripts/evaluate_verifier.py \
  --checkpoint checkpoints/md_state_seed42.pt \
  --cache caches/test \
  --subset-manifest fixed_best8_subsets.json
```

Reported metrics:

- Best@8 (primary);
- Pair-Macro;
- candidate-level AUROC (secondary).

### Exact Best@8 reproduction

The paper evaluates Best@8 on **fixed candidate subsets**. Exact reproduction therefore requires the same candidate artifacts and fixed subset manifest used in the paper. The evaluator accepts that manifest directly.

If no manifest is supplied, the script can generate deterministic random subsets for exploratory use, but those numbers must **not** be treated as exact paper reproduction.

Subset-manifest format:

```json
{
  "problem-id": [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 2, 4, 6, 8, 10, 12, 14]
  ]
}
```

Entries are candidate indices within each problem.

## Controlled representation study

The same training entry point can reproduce the controlled representation variants:

```bash
# Single-depth trajectory state
python scripts/train_verifier.py --model single_state ...

# Multiple final-token endpoints
python scripts/train_verifier.py --model multi_endpoint ...

# Multi-depth token-axis differences
python scripts/train_verifier.py --model multi_diff ...

# Cross-depth matched-token transformations
python scripts/train_verifier.py --model residual_only ...

# State + Residual controls
python scripts/train_verifier.py --model concat ...
python scripts/train_verifier.py --model weighted ...
```

## Mechanism analysis

`analysis/mechanism.py` implements the paper's main representation diagnostics from extracted hidden states:

- relative early→middle and middle→final transformation magnitude;
- directional coherence between successive depth transformations;
- raw cross-layer cosine alignment;
- State discriminability/SNR at each sampled depth;
- Residual discriminability/SNR;
- optional projected cross-layer alignment using a trained `residual_only` checkpoint.

Example:

```bash
python analysis/mechanism.py \
  --cache caches/test \
  --depths 14 21 28 \
  --residual-checkpoint checkpoints/residual_seed42.pt \
  --output results/mechanism.json
```

These analyses are descriptive/correlational; the repository does not treat them as causal interventions.

## Statistical testing

The paper's formal method comparisons use paired problem bootstrap with 10,000 replicates. Three seed-level per-problem Best@8 vectors are averaged before resampling. Holm correction is applied within each pre-specified method-pair family.

`analysis/paired_bootstrap.py` provides the paired bootstrap utility. Exact multi-cell paper statistics additionally require the released per-problem evaluation vectors.

## Data and checkpoints

This repository contains code only. Large hidden-state caches, candidate-generation artifacts, fixed Best@8 subset manifests, and trained checkpoints are not committed to Git.

When the paper artifact package is released, those files should be placed outside version control (or in a dedicated model/data host) and referenced from this README.

## Reproducibility notes

- Keep train/validation/test candidates fixed across representation variants.
- Select checkpoints using validation Pair-Macro only.
- Do not use test performance for architecture or epoch selection.
- Keep sampled depths fixed across matched comparisons.
- For cross-family comparisons, use each backbone's pre-specified relative-depth triplet.
- Treat mechanism analyses as post-performance, correlational analyses.

## Citation

Citation information will be added when the manuscript is public.
