#!/usr/bin/env python
"""Create a frequency-identical RQ-Shuffled control table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engram_peft.rq_table_tools import shuffle_rq_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--access-counts",
        type=Path,
        help="LM slice manifest containing train_access_count_<order> arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = shuffle_rq_table(
        args.source_dir, args.output_dir, args.seed, args.access_counts
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
