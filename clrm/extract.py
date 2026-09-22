from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM


def extract_tokenized_candidates(
    model_name_or_path: str,
    input_jsonl: str | Path,
    output_dir: str | Path,
    depths: Sequence[int],
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
) -> None:
    """Extract response-token hidden states from pre-tokenized candidates.

    Input JSONL rows must contain:
      - problem_id
      - candidate_index
      - label (0/1)
      - input_ids: full prompt+response token ids
      - response_start: index of the first response token in input_ids

    Tokenization is intentionally kept outside this function because exact prompt/chat
    templates are model- and experiment-specific. This avoids silently changing token
    boundaries during reproduction.
    """
    output_dir = Path(output_dir)
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    rows = [json.loads(line) for line in Path(input_jsonl).read_text().splitlines() if line.strip()]
    metadata = []
    with torch.inference_mode():
        for row_idx, row in enumerate(rows):
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=model.device)
            response_start = int(row["response_start"])
            if not 0 <= response_start < input_ids.shape[1]:
                raise ValueError(f"invalid response_start at row {row_idx}")
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = outputs.hidden_states
            state_map = {}
            for depth in depths:
                depth = int(depth)
                if depth >= len(hidden):
                    raise ValueError(f"depth {depth} unavailable; model returned {len(hidden)-1} blocks")
                state_map[depth] = hidden[depth][0, response_start:].detach().cpu().to(torch.float32)
            torch.save(state_map, states_dir / f"{row_idx:08d}.pt")
            metadata.append(
                {
                    "problem_id": str(row["problem_id"]),
                    "candidate_index": int(row["candidate_index"]),
                    "label": int(row["label"]),
                    "response_tokens": int(input_ids.shape[1] - response_start),
                }
            )

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
