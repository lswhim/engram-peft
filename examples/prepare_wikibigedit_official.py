#!/usr/bin/env python
"""Convert ordered official WikiBigEdit timestep files into one auditable stream.

The output keeps every official evaluation axis.  Only ``prompt``/``target`` are
used for training; rephrase, persona, multi-hop, and locality prompts remain
evaluation-only in ``queries``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def nonempty(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def query(prompt: str, answer: str, role: str, axis: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "answers": [answer],
        "role": role,
        "axis": axis,
        "condition_prompts": [],
        "condition_answers": [],
        "condition": "OR",
        "lexical_similarity": None,
        "geometry_text": None,
    }


def convert_row(row: dict[str, Any], timestep: str, index: int) -> dict[str, Any] | None:
    prompt = nonempty(row.get("update"))
    target = nonempty(row.get("ans", row.get("alt")))
    if prompt is None or target is None:
        return None

    queries: list[dict[str, Any]] = [
        query(prompt, target, "should_propagate", "efficacy")
    ]
    rephrase = nonempty(row.get("rephrase"))
    if rephrase is not None:
        queries.append(query(rephrase, target, "should_propagate", "generalization"))

    for key, value in row.items():
        if not key.startswith("personas"):
            continue
        persona = nonempty(value)
        if persona is not None:
            queries.append(query(persona, target, "should_propagate", "personas"))

    multi_hop = nonempty(row.get("mhop"))
    multi_hop_answer = nonempty(row.get("mhop_ans"))
    if multi_hop is not None and multi_hop_answer is not None:
        queries.append(
            query(multi_hop, multi_hop_answer, "should_propagate", "multi_hop")
        )

    locality = nonempty(row.get("loc"))
    locality_answer = nonempty(row.get("loc_ans"))
    if locality is not None and locality_answer is not None:
        queries.append(
            query(locality, locality_answer, "should_not_propagate", "locality")
        )

    return {
        "case_id": f"{timestep}:{index}",
        "prompt": prompt,
        "target": target,
        "queries": queries,
        "metadata": {
            "timestep": timestep,
            "cohort_origin": timestep,
            "source_index": index,
            "tag": row.get("tag"),
            "subject": row.get("subject"),
            "subject_id": row.get("subject_id"),
            "relation": row.get("relation"),
            "relation_id": row.get("relation_id"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timestep-dir",
        type=Path,
        help="Optionally also write one JSONL manifest per official timestep.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: Counter[str] = Counter()
    timesteps: list[dict[str, Any]] = []
    total = 0
    skipped = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for path in args.files:
            rows = json.loads(path.read_text(encoding="utf-8"))
            timestep_total = 0
            timestep_handle = None
            timestep_temporary = None
            timestep_output = None
            if args.timestep_dir is not None:
                args.timestep_dir.mkdir(parents=True, exist_ok=True)
                timestep_output = args.timestep_dir / f"{path.stem}.jsonl"
                timestep_temporary = timestep_output.with_suffix(".jsonl.tmp")
                timestep_handle = timestep_temporary.open("w", encoding="utf-8")
            for index, row in enumerate(rows):
                case = convert_row(row, path.stem, index)
                if case is None:
                    skipped += 1
                    continue
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                if timestep_handle is not None:
                    timestep_handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                timestep_total += 1
                total += 1
                counts.update(item["axis"] for item in case["queries"])
            if timestep_handle is not None:
                timestep_handle.close()
                assert timestep_temporary is not None and timestep_output is not None
                timestep_temporary.replace(timestep_output)
            timesteps.append({"name": path.stem, "cases": timestep_total})
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "protocol": "official_wikibigedit_timestep_order",
                "cases": total,
                "skipped_missing_update_or_answer": skipped,
                "axes": dict(sorted(counts.items())),
                "timesteps": timesteps,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
