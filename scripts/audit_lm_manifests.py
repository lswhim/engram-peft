#!/usr/bin/env python
"""Verify 200-row access and 2,000-row evaluation manifests share one train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def main() -> None:
    args = parse_args()
    seeds: dict[str, Any] = {}
    all_equal = True
    for seed in (42, 43, 44):
        access = np.load(
            args.slice_root / f"manifest_seed{seed}.npz", allow_pickle=False
        )
        evaluation = np.load(
            args.slice_root / f"manifest_eval2000_seed{seed}.npz",
            allow_pickle=False,
        )
        orders: dict[str, Any] = {}
        for order in (2, 3):
            key = f"train_access_count_{order}"
            same = bool(np.array_equal(access[key], evaluation[key]))
            all_equal &= same
            orders[str(order)] = {
                "identical": same,
                "sha256_access": digest(access[key]),
                "sha256_evaluation": digest(evaluation[key]),
                "total_train_accesses": int(access[key].sum()),
            }
        seeds[str(seed)] = orders
    payload = {
        "status": "complete" if all_equal else "failed",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": (
            "The 200-row frequency-control manifest and 2,000-row evaluation "
            "manifest must have bit-identical train_access_count arrays."
        ),
        "all_train_access_counts_identical": all_equal,
        "seeds": seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
