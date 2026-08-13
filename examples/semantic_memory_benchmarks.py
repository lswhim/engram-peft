#!/usr/bin/env python
"""Canonical data contracts for semantic-address memory benchmarks.

This module intentionally contains no model code.  It converts official
ParaRel, WikiBigEdit, and RippleEdits releases into one transparent write/query
schema consumed by the training and evaluation runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Query:
    prompt: str
    answers: tuple[str, ...]
    role: str
    axis: str
    condition_prompts: tuple[str, ...] = ()
    condition_answers: tuple[tuple[str, ...], ...] = ()
    condition: str = "OR"
    lexical_similarity: float | None = None


@dataclass(frozen=True)
class EditCase:
    case_id: str
    prompt: str
    target: str
    queries: tuple[Query, ...]
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fill_pattern(pattern: str, subject: str, target: str | None = None) -> str:
    text = pattern.replace("[X]", subject)
    if target is None:
        text = text.replace("[Y]", "").strip()
        text = re.sub(r"\s+([.,?!])", r"\1", text)
        return text.rstrip(" .")
    return re.sub(r"\s+([.,?!])", r"\1", text.replace("[Y]", target)).strip()


def _char_trigrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return {normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))}


def lexical_similarity(left: str, right: str) -> float:
    a, b = _char_trigrams(left), _char_trigrams(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def load_pararel(pattern_dir: Path, facts_dir: Path, seed: int = 42) -> list[EditCase]:
    """Build counterfactual canonical-write/unseen-template cases.

    Targets are deterministically deranged within each relation.  This avoids
    crediting a frozen base model for facts it already knew.
    """
    cases: list[EditCase] = []
    for fact_path in sorted(facts_dir.glob("*.jsonl")):
        relation = fact_path.stem
        pattern_path = pattern_dir / fact_path.name
        if not pattern_path.is_file():
            continue
        patterns = [row["pattern"] for row in _read_jsonl(pattern_path)]
        facts = _read_jsonl(fact_path)
        if len(patterns) < 2 or len(facts) < 2:
            continue
        # A per-relation rotation is a bijection and guarantees target != truth.
        shift = 1 + int(hashlib.sha256(f"{seed}:{relation}".encode()).hexdigest(), 16) % (len(facts) - 1)
        targets = [str(facts[(i + shift) % len(facts)]["obj_label"]) for i in range(len(facts))]
        for index, (fact, target) in enumerate(zip(facts, targets, strict=True)):
            subject = str(fact["sub_label"])
            canonical = _fill_pattern(patterns[0], subject)
            queries = tuple(
                Query(
                    prompt=_fill_pattern(pattern, subject),
                    answers=(target,),
                    role="should_propagate",
                    axis="unseen_template",
                    lexical_similarity=lexical_similarity(canonical, _fill_pattern(pattern, subject)),
                )
                for pattern in patterns[1:]
            )
            cases.append(EditCase(
                case_id=f"{relation}:{fact.get('uuid', index)}",
                prompt=canonical,
                target=target,
                queries=queries,
                metadata={"relation": relation, "original_target": str(fact["obj_label"]), "protocol": "within_relation_derangement"},
            ))
    return cases


def _answers(query: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for answer in query.get("answers", []):
        values.append(str(answer["value"]))
        values.extend(str(alias) for alias in answer.get("aliases", []))
    return tuple(dict.fromkeys(value for value in values if value))


RIPPLE_SHOULD = {"Logical_Generalization", "Compositionality_I", "Compositionality_II", "Subject_Aliasing"}
RIPPLE_SHOULD_NOT = {"Relation_Specificity", "Forgetfulness"}


def load_ripple(paths: Iterable[Path]) -> list[EditCase]:
    cases: list[EditCase] = []
    for path in paths:
        for index, example in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            edit = example["edit"]
            statement = str(edit["prompt"]).strip()
            if " is " not in statement:
                raise ValueError(f"Ripple edit lacks answer delimiter: {statement}")
            edit_prompt, target = statement.rsplit(" is ", 1)
            edit_prompt = f"{edit_prompt} is"
            target = target.rstrip(".").strip()
            queries: list[Query] = []
            for axis in sorted(RIPPLE_SHOULD | RIPPLE_SHOULD_NOT):
                role = "should_propagate" if axis in RIPPLE_SHOULD else "should_not_propagate"
                for testcase in example.get(axis, []):
                    conditions = testcase.get("condition_queries", [])
                    condition_prompts = tuple(str(item["prompt"]) for item in conditions)
                    condition_answers = tuple(_answers(item) for item in conditions)
                    for query in testcase.get("test_queries", []):
                        queries.append(Query(
                            prompt=str(query["prompt"]), answers=_answers(query), role=role,
                            axis=axis, condition_prompts=condition_prompts,
                            condition_answers=condition_answers,
                            condition=str(testcase.get("test_condition", "OR")),
                        ))
            cases.append(EditCase(
                case_id=f"{path.stem}:{index}", prompt=edit_prompt, target=target,
                queries=tuple(queries), metadata={"split": path.stem, "relation": edit.get("relation")},
            ))
    return cases


def load_wikibigedit(paths: Iterable[Path]) -> list[EditCase]:
    """Load official time increments in caller-provided chronological order."""
    cases: list[EditCase] = []
    for path in paths:
        for index, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            prompt = str(row.get("update", row.get("src", "")))
            target = str(row.get("ans", row.get("alt", "")))
            queries = [Query(str(row["rephrase"]), (target,), "should_propagate", "generalization")]
            if row.get("loc") and row.get("loc_ans"):
                queries.append(Query(str(row["loc"]), (str(row["loc_ans"]),), "should_not_propagate", "locality"))
            if row.get("mhop") and row.get("mhop_ans"):
                queries.append(Query(str(row["mhop"]), (str(row["mhop_ans"]),), "should_propagate", "multi_hop"))
            cases.append(EditCase(
                case_id=f"{path.parent.name}:{index}", prompt=prompt, target=target,
                queries=tuple(queries), metadata={"increment": path.parent.name, "tag": row.get("tag"), "subject": row.get("subject")},
            ))
    return cases


def stream_checkpoints(total: int, requested: Iterable[int]) -> tuple[int, ...]:
    points = sorted({point for point in requested if 0 < point <= total})
    return tuple(points + ([total] if total and (not points or points[-1] != total) else []))


def manifest_summary(cases: Iterable[EditCase]) -> dict[str, Any]:
    case_list = list(cases)
    axes: dict[str, int] = {}
    roles: dict[str, int] = {}
    conditioned = 0
    for case in case_list:
        for query in case.queries:
            axes[query.axis] = axes.get(query.axis, 0) + 1
            roles[query.role] = roles.get(query.role, 0) + 1
            conditioned += int(bool(query.condition_prompts))
    return {
        "cases": len(case_list),
        "queries": sum(axes.values()),
        "axes": dict(sorted(axes.items())),
        "roles": dict(sorted(roles.items())),
        "conditioned_queries": conditioned,
    }


def write_manifest(cases: Iterable[EditCase], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="benchmark", required=True)
    para = sub.add_parser("pararel")
    para.add_argument("--patterns", type=Path, required=True); para.add_argument("--facts", type=Path, required=True)
    ripple = sub.add_parser("ripple")
    ripple.add_argument("--files", type=Path, nargs="+", required=True)
    wiki = sub.add_parser("wikibigedit")
    wiki.add_argument("--files", type=Path, nargs="+", required=True)
    for command in (para, ripple, wiki): command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark == "pararel": cases = load_pararel(args.patterns, args.facts)
    elif args.benchmark == "ripple": cases = load_ripple(args.files)
    else: cases = load_wikibigedit(args.files)
    write_manifest(cases, args.output)
    print(json.dumps({"benchmark": args.benchmark, "output": str(args.output), **manifest_summary(cases)}, indent=2))


if __name__ == "__main__":
    main()
