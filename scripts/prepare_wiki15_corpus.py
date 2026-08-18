#!/usr/bin/env python
"""Build an unlabeled 15-language Wikipedia corpus for multilingual RQ training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


LANGUAGES = (
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs-per-language", type=int, default=1500)
    parser.add_argument("--snapshot", default="20231101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: dict[str, int] = {}
    with temporary.open("w", encoding="utf-8") as handle:
        for language in LANGUAGES:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                f"{args.snapshot}.{language}",
                split="train",
                streaming=True,
            )
            kept = 0
            for row in dataset:
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "language": language,
                            "title": str(row.get("title", "")),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
                if kept >= args.docs_per_language:
                    break
            if kept != args.docs_per_language:
                raise RuntimeError(
                    f"{language}: expected {args.docs_per_language} documents, got {kept}"
                )
            counts[language] = kept
            handle.flush()
            print(f"[wiki15] {language}: {kept}", flush=True)
    temporary.replace(args.output)
    manifest = args.output.with_suffix(args.output.suffix + ".meta.json")
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "source": "wikimedia/wikipedia",
                "snapshot": args.snapshot,
                "docs_per_language": args.docs_per_language,
                "languages": counts,
                "total_documents": sum(counts.values()),
                "contains_benchmark_labels": False,
                "uses_benchmark_evaluation_text": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {args.output} ({sum(counts.values())} documents)", flush=True)


if __name__ == "__main__":
    main()
