#!/usr/bin/env python
from __future__ import annotations

import argparse

from clrm.extract import extract_tokenized_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="Tokenized candidate JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--depths", required=True, nargs="+", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()
    extract_tokenized_candidates(args.model, args.input, args.output, args.depths, args.device, args.dtype)


if __name__ == "__main__":
    main()
