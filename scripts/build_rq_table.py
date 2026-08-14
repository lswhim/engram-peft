#!/usr/bin/env python
"""
Offline builder for the semantic-hash (RQ) Engram address table.

All steps are OFFLINE and the result is frozen, so runtime stays O(1) / no neural compute.

  1. Build CompressedTokenizer for the base model tokenizer (same normalization Engram
     uses at runtime).
  2. Stream a corpus, extract suffix n-grams over *compressed* token ids, count freq,
     keep top-N per size.
  3. Embed each kept n-gram's surface text with an external embedder (Qwen3-Embedding).
     This is the only semantic step; it happens here once.
  4. Train a faiss ResidualQuantizer (M levels x K codes) per n-gram size; encode each
     n-gram into M integer codes.
  5. Save {sorted int64 key -> M codes} arrays, metadata, and the trained FAISS RQ index.
     The persisted quantizer is required to encode previously unseen n-gram vectors with
     the same frozen address geometry. Runtime refuses missing keys instead of falling
     back to a different hash family.

Key encoding (per n-gram size n): k = sum_j c_j * base^(n-1-j), base = V'+1, int64.
With V'~1e5 and n<=3 this stays < int64 max.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RQ semantic-hash table for Engram.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", default="text")
    p.add_argument(
        "--text_columns",
        nargs="+",
        default=None,
        help=(
            "Optional ordered JSON fields to concatenate into each corpus document. "
            "For a semantic-memory manifest use: --text_columns prompt target."
        ),
    )
    p.add_argument("--num_docs", type=int, default=20000)
    p.add_argument("--max_doc_tokens", type=int, default=512)
    # optional corpus filters (e.g. Biomed-Enriched: en + biomedical + edu>4.0)
    p.add_argument("--filter_language", default=None)
    p.add_argument("--filter_domain", default=None)
    p.add_argument("--min_edu_score", type=float, default=None)
    p.add_argument("--base_tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--ngram_sizes", type=int, nargs="+", default=[2, 3])
    p.add_argument("--num_levels", type=int, default=8,
                   help="RQ levels M per n-gram size (= per-ngram head count).")
    p.add_argument("--codebook_size", type=int, default=256, help="codes per level (K)")
    p.add_argument("--max_ngrams_per_size", type=int, default=500000)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--embed_batch_size", type=int, default=256)
    p.add_argument("--rq_train_threads", type=int, default=32,
                   help="faiss OMP threads for RQ training (CPU-bound step)")
    p.add_argument(
        "--projection_dim",
        type=int,
        default=0,
        help="If >0, train a corpus-specific autoencoder and run RQ in this latent space.",
    )
    p.add_argument("--autoencoder_epochs", type=int, default=30)
    p.add_argument("--autoencoder_batch_size", type=int, default=512)
    p.add_argument("--autoencoder_lr", type=float, default=1e-3)
    p.add_argument("--data_files", type=str, default=None,
                   help="If set, load corpus from local JSONL via load_dataset('json', data_files=...). Overrides --dataset/--dataset_config.")
    p.add_argument("--output_dir", default="rq_tables/fineweb_qwen3")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_compressor(base_tokenizer: str):
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer

    tok = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    comp = CompressedTokenizer(tokenizer=tok)
    print(f"[compressor] V={comp.vocab_size} -> V'={comp.compressed_vocab_size}")
    return comp, tok


def poly_key(comp_ids, base: int) -> int:
    k = 0
    for c in comp_ids:
        k = k * base + int(c)
    return k


def scan_ngrams(args, comp, tok):
    from datasets import load_dataset

    cfg = args.dataset_config if args.dataset_config not in (None, "", "none") else None
    if args.data_files:
        ds = load_dataset("json", data_files=args.data_files, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, cfg, split=args.split, streaming=True)
    base = comp.compressed_vocab_size + 1
    counters = {n: Counter() for n in args.ngram_sizes}
    repr_tokens = {n: {} for n in args.ngram_sizes}
    seen = 0
    for ex in ds:
        if args.filter_language and ex.get("language") != args.filter_language:
            continue
        if args.filter_domain and ex.get("domain") != args.filter_domain:
            continue
        if args.min_edu_score is not None and (ex.get("educational_score") or 0) < args.min_edu_score:
            continue
        if args.text_columns:
            parts = [str(ex.get(column, "")).strip() for column in args.text_columns]
            text = " ".join(part for part in parts if part)
        else:
            text = ex.get(args.text_column)
        if not text:
            continue
        ids = tok(text, truncation=True, max_length=args.max_doc_tokens)["input_ids"]
        if len(ids) < max(args.ngram_sizes):
            continue
        cids = comp.map_ids(np.asarray(ids)).tolist()
        for n in args.ngram_sizes:
            for i in range(n - 1, len(cids)):
                key = poly_key(cids[i - n + 1 : i + 1], base)
                counters[n][key] += 1
                if key not in repr_tokens[n]:
                    repr_tokens[n][key] = ids[i - n + 1 : i + 1]
        seen += 1
        if seen >= args.num_docs:
            break
    print(f"[scan] {seen} docs; "
          + ", ".join(f"{n}-gram uniq={len(counters[n])}" for n in args.ngram_sizes))
    return base, counters, repr_tokens


def select_ngrams(args, counters, repr_tokens, tok):
    selected = {}
    for n in args.ngram_sizes:
        items = [(k, c) for k, c in counters[n].items() if c >= args.min_count]
        items.sort(key=lambda kc: kc[1], reverse=True)
        items = items[: args.max_ngrams_per_size]
        keys = np.array([k for k, _ in items], dtype=np.int64)
        keys.sort()
        texts = [tok.decode(repr_tokens[n][int(k)], skip_special_tokens=False)
                 for k in keys]
        selected[n] = {"keys": keys, "texts": texts}
        print(f"[select] {n}-gram kept {len(keys)}")
    return selected


def embed_texts(args, texts):
    import torch
    from transformers import AutoModel, AutoTokenizer

    etok = AutoTokenizer.from_pretrained(args.embedder, trust_remote_code=True)
    emodel = AutoModel.from_pretrained(
        args.embedder, trust_remote_code=True, dtype=torch.float16
    ).cuda().eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), args.embed_batch_size):
            enc = etok(texts[s : s + args.embed_batch_size], padding=True,
                       truncation=True, max_length=32, return_tensors="pt").to("cuda")
            hs = emodel(**enc).last_hidden_state
            lengths = enc["attention_mask"].sum(dim=1) - 1
            vec = hs[torch.arange(hs.size(0)), lengths]
            vec = torch.nn.functional.normalize(vec, dim=-1)
            out.append(vec.float().cpu().numpy())
            if (s // args.embed_batch_size) % 20 == 0:
                print(f"[embed] {s}/{len(texts)}")
    return np.concatenate(out, axis=0)


def unpack_codes(packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    import faiss

    return np.asarray(faiss.unpack_bitstrings(packed, M, nbits), dtype=np.int64)


def train_rq(args, emb):
    import faiss

    faiss.omp_set_num_threads(args.rq_train_threads)
    print(f"[rq] faiss omp threads = {faiss.omp_get_max_threads()}")
    d = emb.shape[1]
    if len(emb) < d:
        raise ValueError(
            f"RQ training needs at least embedding_dim samples for the FAISS training "
            f"path (got {len(emb)} samples, dim={d}); increase max_ngrams_per_size"
        )
    nbits = int(np.log2(args.codebook_size))
    assert (1 << nbits) == args.codebook_size, "codebook_size must be power of 2"
    rq = faiss.ResidualQuantizer(d, args.num_levels, nbits)
    x = np.ascontiguousarray(emb.astype(np.float32))
    rq.train(x)
    packed = rq.compute_codes(x)
    codes = unpack_codes(packed, args.num_levels, nbits).astype(np.uint16)
    # FAISS serializes Index objects, not bare ResidualQuantizer instances. Attach the
    # already-trained exact RQ object to a codec-only index; do not retrain through the
    # IndexResidualQuantizer wrapper (FAISS 1.13 may enable an incompatible PCA path).
    index = faiss.IndexResidualQuantizer(d, args.num_levels, nbits)
    index.rq = rq
    index.is_trained = True
    return codes, index


def train_autoencoder(args, emb: np.ndarray, n: int) -> np.ndarray:
    """Learn a task-corpus bottleneck without using evaluation queries or labels."""
    import torch

    if args.projection_dim <= 0:
        return emb
    torch.manual_seed(args.seed + n)
    input_dim = int(emb.shape[1])
    hidden_dim = max(args.projection_dim * 2, min(512, input_dim))

    class Autoencoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, args.projection_dim),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(args.projection_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = torch.nn.functional.normalize(self.encoder(x), dim=-1)
            return self.decoder(z), z

    device = torch.device("cuda")
    model = Autoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.autoencoder_lr)
    tensor = torch.from_numpy(emb.astype(np.float32))
    generator = torch.Generator().manual_seed(args.seed + n)
    loader = torch.utils.data.DataLoader(
        tensor,
        batch_size=args.autoencoder_batch_size,
        shuffle=True,
        generator=generator,
    )
    model.train()
    for epoch in range(args.autoencoder_epochs):
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            cosine = 1.0 - torch.nn.functional.cosine_similarity(
                reconstruction, batch, dim=-1
            ).mean()
            mse = torch.nn.functional.mse_loss(reconstruction, batch)
            loss = cosine + mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.autoencoder_epochs:
            print(f"[ae n={n}] epoch={epoch + 1}/{args.autoencoder_epochs} loss={total / len(tensor):.6f}")
    model.eval()
    latents = []
    with torch.inference_mode():
        for start in range(0, len(tensor), args.autoencoder_batch_size):
            _, z = model(tensor[start : start + args.autoencoder_batch_size].to(device))
            latents.append(z.cpu().numpy())
    torch.save(
        {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "projection_dim": args.projection_dim,
            "state_dict": model.encoder.state_dict(),
        },
        os.path.join(args.output_dir, f"projector_{n}.pt"),
    )
    return np.ascontiguousarray(np.concatenate(latents).astype(np.float32))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    comp, tok = build_compressor(args.base_tokenizer)
    base, counters, repr_tokens = scan_ngrams(args, comp, tok)
    selected = select_ngrams(args, counters, repr_tokens, tok)

    meta = {
        "base": int(base),
        "compressed_vocab_size": int(comp.compressed_vocab_size),
        "ngram_sizes": args.ngram_sizes,
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "embedder": args.embedder,
        "base_tokenizer": args.base_tokenizer,
        "pad_token_id": int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        "strict_semantic_rq": True,
        "projection": "autoencoder" if args.projection_dim > 0 else "none",
        "projection_dim": int(args.projection_dim),
        "address_corpus_fields": args.text_columns or [args.text_column],
    }
    for n in args.ngram_sizes:
        keys, texts = selected[n]["keys"], selected[n]["texts"]
        if len(keys) == 0:
            print(f"[warn] no {n}-grams kept; skipping")
            continue
        emb = embed_texts(args, texts)
        rq_input = train_autoencoder(args, emb, n)
        codes, rq_index = train_rq(args, rq_input)
        import faiss

        np.save(os.path.join(args.output_dir, f"keys_{n}.npy"), keys)
        np.save(os.path.join(args.output_dir, f"codes_{n}.npy"), codes)
        faiss.write_index(rq_index, os.path.join(args.output_dir, f"rq_{n}.faiss"))
        print(f"[save] {n}-gram keys{keys.shape} codes{codes.shape}")

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] -> {args.output_dir}")


if __name__ == "__main__":
    main()
