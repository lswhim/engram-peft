# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
import copy
from itertools import islice
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def _load_tinystories(subset_size: int, eval_size: int) -> tuple[Any, Any]:
    print(f"Loading TinyStories dataset (subset={subset_size})...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)
    train_ds = train_ds.select(range(subset_size))
    val_ds = val_ds.select(range(min(len(val_ds), eval_size)))
    return train_ds, val_ds


def _load_fineweb(subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    """Load disjoint FineWeb-Edu rows through the streaming interface.

    The stream is shuffled once and then partitioned before tokenization.  This avoids
    reading benchmark test text into either the LM training set or the address table.
    """
    print(
        f"Loading FineWeb-Edu sample-10BT (streaming, train={subset_size}, "
        f"eval={eval_size}, seed={seed})..."
    )
    address_reserve_rows = 6_000
    raw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    ).skip(address_reserve_rows).shuffle(
        seed=seed, buffer_size=max(subset_size + eval_size, 20_000)
    )
    print(
        f"Reserved the first {address_reserve_rows} stream rows exclusively for "
        "offline address construction; LM train/eval starts after that boundary."
    )
    rows = [
        {"text": str(example["text"])}
        for example in islice(raw, subset_size + eval_size)
        if example.get("text")
    ]
    if len(rows) < subset_size + eval_size:
        raise RuntimeError(
            f"FineWeb stream returned {len(rows)} usable rows; "
            f"expected {subset_size + eval_size}"
        )
    train_rows = rows[:subset_size]
    eval_rows = rows[subset_size : subset_size + eval_size]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _load_biomed(subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    """Biomed-Enriched (TinyEngram protocol): en + biomedical; train edu>4.0, eval edu<4.0."""
    print(f"Loading Biomed-Enriched (streaming, train subset={subset_size}, seed={seed})...")
    raw = load_dataset("almanach/Biomed-Enriched", split="commercial", streaming=True)
    # Shuffle the stream so different seeds draw different train subsets.
    # Without this we always took the fixed first-N rows -> seed was a no-op
    # (every seed trained on identical data, see SKILL.md §11.1).
    raw = raw.shuffle(seed=seed, buffer_size=max(subset_size + eval_size, 20000))
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for ex in raw:
        if ex.get("language") != "en" or ex.get("domain") != "biomedical":
            continue
        text = ex.get("text")
        if not text:
            continue
        edu = ex.get("educational_score") or 0
        if edu > 4.0 and len(train_rows) < subset_size:
            train_rows.append({"text": text})
        elif edu < 4.0 and len(eval_rows) < eval_size:
            eval_rows.append({"text": text})
        if len(train_rows) >= subset_size and len(eval_rows) >= eval_size:
            break
    print(f"Biomed: {len(train_rows)} train / {len(eval_rows)} eval examples.")
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _load_jsonl_corpus(path: str, subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    """Load a pre-generated JSONL corpus (one {'text': ...} per line) + carve eval split.
    Used by zsRE / MQuAKE which we preprocess offline."""
    import json
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = list(range(len(rows)))
    import random
    random.Random(seed).shuffle(rng)
    # Carve eval FIRST so it's never empty; train takes the rest (capped at subset_size).
    eval_n = min(eval_size, len(rows))
    eval_rows = [rows[i] for i in rng[:eval_n]]
    train_n = min(subset_size, len(rows) - eval_n)
    train_rows = [rows[i] for i in rng[eval_n : eval_n + train_n]]
    print(f"[jsonl] {path}: total {len(rows)} -> {len(train_rows)} train / {len(eval_rows)} eval.")
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _load_zsre(subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    return _load_jsonl_corpus("data/zsre/corpus_train.jsonl", subset_size, eval_size, seed)


def _load_mquake(subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    return _load_jsonl_corpus("data/mquake/corpus_train.jsonl", subset_size, eval_size, seed)


def _load_counterfact(subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    """COUNTERFACT (Meng+22): each train case expanded into ~6 fact sentences
    (prompt-template fill + paraphrases + generation prompts), all paired with
    target_new so engram memorizes the counterfactual answer. subset_size = #cases."""
    print(f"Loading COUNTERFACT (cases subset={subset_size}, seed={seed})...")
    train = load_dataset("azhx/counterfact", split="train").shuffle(seed=seed)
    test = load_dataset("azhx/counterfact", split="test")

    def expand(ex: dict[str, Any]) -> list[str]:
        rr = ex["requested_rewrite"]
        prompt = rr["prompt"].format(rr["subject"])
        new = rr["target_new"]["str"]
        sents = [f"{prompt} {new}."]
        sents += [f"{p} {new}." for p in ex["paraphrase_prompts"]]
        sents += [f"{g} {new}." for g in ex["generation_prompts"][:3]]
        return sents

    train_rows: list[dict[str, str]] = []
    for ex in train.select(range(min(subset_size, len(train)))):
        for s in expand(ex):
            train_rows.append({"text": s})
    eval_rows: list[dict[str, str]] = []
    for ex in test.select(range(min(eval_size, len(test)))):
        rr = ex["requested_rewrite"]
        eval_rows.append({"text": f"{rr['prompt'].format(rr['subject'])} {rr['target_new']['str']}."})
    print(f"COUNTERFACT: {len(train_rows)} train sentences / {len(eval_rows)} eval sentences.")
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _load_counterfact_canonical(
    subset_size: int, eval_size: int, seed: int = 42
) -> tuple[Any, Any]:
    """Full batch-edit protocol: train only canonical rewrites from CounterFact test.

    Paraphrase, neighborhood, and generation prompts are deliberately excluded and
    remain evaluation-only. ``subset_size`` caps cases, not expanded sentences.
    """
    del seed
    cases = load_dataset("azhx/counterfact", split="test")
    if subset_size > 0:
        cases = cases.select(range(min(subset_size, len(cases))))
    rows = []
    for example in cases:
        rewrite = example["requested_rewrite"]
        prompt = rewrite["prompt"].format(rewrite["subject"])
        rows.append(
            {
                "text": f"{prompt} {rewrite['target_new']['str']}",
                "prompt": prompt,
                "target": str(rewrite["target_new"]["str"]),
            }
        )
    train = Dataset.from_list(rows)
    validation = Dataset.from_list(rows[: min(eval_size, len(rows))])
    print(
        f"CounterFact canonical-only batch edit: {len(train)} edits; "
        "paraphrases/neighborhoods excluded from training."
    )
    return train, validation


def _load_zsre_canonical(
    subset_size: int, eval_size: int, seed: int = 42
) -> tuple[Any, Any]:
    """KnowEdit ZsRE test edits, canonical prompt only; rephrases/locality held out."""
    del seed
    import json

    path = "data/zsre/benchmark/ZsRE/ZsRE-test-all.json"
    cases = json.load(open(path, encoding="utf-8"))
    if subset_size > 0:
        cases = cases[: min(subset_size, len(cases))]
    rows = [
        {
            "text": f"{case['prompt']} {case['target_new']}",
            "prompt": str(case["prompt"]),
            "target": str(case["target_new"]),
        }
        for case in cases
    ]
    train = Dataset.from_list(rows)
    validation = Dataset.from_list(rows[: min(eval_size, len(rows))])
    print(
        f"ZsRE canonical-only batch edit: {len(train)} edits; "
        "rephrases/locality excluded from training."
    )
    return train, validation


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    subset_size: int,
    eval_size: int,
    max_length: int,
    num_proc: int = 4,
    dataset: str = "tinystories",
    seed: int = 42,
) -> tuple[Any, Any]:
    """Standardized dataset preparation. dataset in {tinystories, biomed, counterfact, zsre, mquake}."""
    if dataset == "fineweb":
        train_ds, val_ds = _load_fineweb(subset_size, eval_size, seed=seed)
    elif dataset == "biomed":
        train_ds, val_ds = _load_biomed(subset_size, eval_size, seed=seed)
    elif dataset == "counterfact":
        # Use the pre-built JSONL (data/counterfact/corpus_train.jsonl: 118k sentences from 19728 cases).
        # subset_size here is in SENTENCES (not cases); set big to use full corpus.
        train_ds, val_ds = _load_jsonl_corpus("data/counterfact/corpus_train.jsonl", subset_size, eval_size, seed=seed)
    elif dataset == "counterfact_canonical":
        train_ds, val_ds = _load_counterfact_canonical(
            subset_size, eval_size, seed=seed
        )
    elif dataset == "wiki_recent":
        train_ds, val_ds = _load_jsonl_corpus("data/wiki_recent/corpus_train.jsonl", subset_size, eval_size, seed=seed)
    elif dataset == "wiki_cf":
        train_ds, val_ds = _load_jsonl_corpus("data/wiki_cf/corpus_train.jsonl", subset_size, eval_size, seed=seed)
    elif dataset == "zsre":
        train_ds, val_ds = _load_zsre(subset_size, eval_size, seed=seed)
    elif dataset == "zsre_canonical":
        train_ds, val_ds = _load_zsre_canonical(subset_size, eval_size, seed=seed)
    elif dataset == "mquake":
        train_ds, val_ds = _load_mquake(subset_size, eval_size, seed=seed)
    else:
        train_ds, val_ds = _load_tinystories(subset_size, eval_size)

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        if "prompt" in examples and "target" in examples:
            input_rows: list[list[int]] = []
            attention_rows: list[list[int]] = []
            label_rows: list[list[int]] = []
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError("tokenizer needs a pad or EOS token")
            for prompt, target in zip(
                examples["prompt"], examples["target"], strict=True
            ):
                prompt_ids = tokenizer(
                    str(prompt), add_special_tokens=True
                )["input_ids"]
                target_ids = tokenizer(
                    " " + str(target).strip(), add_special_tokens=False
                )["input_ids"]
                prompt_ids = prompt_ids[: max(0, max_length - len(target_ids))]
                target_ids = target_ids[: max_length - len(prompt_ids)]
                ids = list(prompt_ids) + list(target_ids)
                labels = [-100] * len(prompt_ids) + list(target_ids)
                padding = max_length - len(ids)
                input_rows.append(ids + [int(pad_id)] * padding)
                attention_rows.append([1] * len(ids) + [0] * padding)
                label_rows.append(labels + [-100] * padding)
            return {
                "input_ids": input_rows,
                "attention_mask": attention_rows,
                "labels": label_rows,
            }
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized_dict = dict(tokenized)
        tokenized_dict["labels"] = copy.deepcopy(tokenized_dict["input_ids"])
        return tokenized_dict

    print(f"Tokenizing with {num_proc} processes...")
    train_dataset = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=list(train_ds.column_names),
        num_proc=num_proc,
    )
    eval_dataset = val_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=list(val_ds.column_names),
        num_proc=num_proc,
    )
    return train_dataset, eval_dataset
