#!/usr/bin/env python3
"""Merge independent local RQ SQLite caches into one read-only cache."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA busy_timeout = 300000")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS codes "
        "(n INTEGER NOT NULL, key INTEGER NOT NULL, code BLOB NOT NULL, "
        "PRIMARY KEY (n, key))"
    )
    attached = []
    for index, source in enumerate(args.inputs):
        connection.execute(f"ATTACH DATABASE ? AS shard_{index}", (str(source),))
        attached.append(index)
    for index, source in enumerate(args.inputs):
        connection.execute(
            f"INSERT OR IGNORE INTO main.codes SELECT n, key, code FROM shard_{index}.codes"
        )
        connection.commit()
        print(f"[rq merge] {index + 1}/{len(args.inputs)} {source}", flush=True)
    connection.close()


if __name__ == "__main__":
    main()
