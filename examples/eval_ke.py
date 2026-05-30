#!/usr/bin/env python
"""Unified knowledge-editing eval (ROME first-token logprob protocol).

Supports 3 datasets:
  - counterfact (azhx/counterfact test split)
  - zsre       (data/zsre/benchmark/ZsRE/ZsRE-test-all.json)
  - mquake     (data/mquake.json - single-hop variant via new_single_hops)

Supports loading:
  - --engram_weights <dir>   (EngramModel.from_pretrained)
  - --lora_weights <dir>     (PeftModel.from_pretrained)
  - neither (= base model)

Reports:
  - efficacy   : P(target_new[0]) > P(target_true[0]) on main prompt
  - paraphrase : same averaged over paraphrase prompts (may be empty)
  - specificity: P(neighbor_gt[0]) > P(target_new[0]) on neighbor prompts
                 (neighbor_gt = neighbor's own ground truth, not edited)
"""
import argparse, json, os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def first_token_id(tok, target):
    if target is None or target == "":
        return None
    ids = tok(" " + str(target), add_special_tokens=False).input_ids
    return ids[0] if ids else None


@torch.no_grad()
def logprob_first(tok, model, prompt, target):
    tid = first_token_id(tok, target)
    if tid is None:
        return float("-inf")
    enc = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1, :].float()
    return torch.log_softmax(logits, dim=-1)[tid].item()


# ---- dataset normalizers ----
# All return list of dicts: {prompt, target_new, target_true, paraphrases:[str], neighbors:[(prompt, gt)]}

def load_counterfact(limit):
    ds = load_dataset("azhx/counterfact", split="test")
    if limit: ds = ds.select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        rr = ex["requested_rewrite"]
        out.append({
            "prompt": rr["prompt"].format(rr["subject"]),
            "target_new": rr["target_new"]["str"],
            "target_true": rr["target_true"]["str"],
            "paraphrases": list(ex["paraphrase_prompts"]),
            "neighbors": [(p, rr["target_true"]["str"]) for p in ex["neighborhood_prompts"]],
        })
    return out


def load_zsre(limit):
    data = json.load(open("data/zsre/benchmark/ZsRE/ZsRE-test-all.json"))
    if limit: data = data[: min(limit, len(data))]
    out = []
    for ex in data:
        gt = ex["ground_truth"]
        target_true = gt[0] if isinstance(gt, list) and gt else (gt if isinstance(gt, str) else "")
        neighbors = []
        for nb in ex.get("locality", {}).get("Relation_Specificity", []):
            nb_gt = nb["ground_truth"]
            # nested list weirdness: sometimes [[s,s2]], sometimes [s]
            while isinstance(nb_gt, list) and nb_gt:
                nb_gt = nb_gt[0]
            if not isinstance(nb_gt, str): continue
            neighbors.append((nb["prompt"], nb_gt))
        out.append({
            "prompt": ex["prompt"],
            "target_new": ex["target_new"],
            "target_true": target_true,
            "paraphrases": [ex["rephrase_prompt"]] if ex.get("rephrase_prompt") else [],
            "neighbors": neighbors,
        })
    return out


def _load_wiki_json(path: str, limit):
    """KnowEdit wiki_recent / wiki_counterfact loader.
    Both have {subject, prompt, target_new, rephrase (str), locality, [ground_truth for cf]}.
    wiki_recent has no ground_truth -> set target_true="" (efficacy reduces to just predicting new).
    """
    data = json.load(open(path))
    if limit: data = data[: min(limit, len(data))]
    out = []
    for ex in data:
        target_true = ""
        gt = ex.get("ground_truth")
        if gt:
            if isinstance(gt, list) and gt:
                gt = gt[0]
            if isinstance(gt, str): target_true = gt
        rp = ex.get("rephrase")
        paraphrases = [rp] if isinstance(rp, str) and rp else (list(rp) if isinstance(rp, list) else [])
        # locality (neighborhood): Relation_Specificity prompts with their own ground_truth
        neighbors = []
        for nb in ex.get("locality", {}).get("Relation_Specificity", []):
            nb_gt = nb.get("ground_truth")
            while isinstance(nb_gt, list) and nb_gt:
                nb_gt = nb_gt[0]
            if isinstance(nb_gt, str) and nb["prompt"]:
                neighbors.append((nb["prompt"], nb_gt))
        out.append({
            "prompt": ex["prompt"],
            "target_new": ex["target_new"],
            "target_true": target_true,
            "paraphrases": paraphrases,
            "neighbors": neighbors,
        })
    return out


def load_wiki_recent(limit):
    return _load_wiki_json("data/wiki_recent/benchmark/wiki_recent/recent_test.json", limit)


def load_wiki_cf(limit):
    return _load_wiki_json("data/wiki_cf/benchmark/wiki_counterfact/test_cf.json", limit)


def load_mquake(limit):
    data = json.load(open("data/mquake.json"))
    if limit: data = data[: min(limit, len(data))]
    out = []
    for ex in data:
        # Use new_single_hops (post-edit) vs single_hops (pre-edit) as new/true pairs
        new_hops = ex.get("new_single_hops", [])
        old_hops = ex.get("single_hops", [])
        n = min(len(new_hops), len(old_hops))
        for i in range(n):
            q = new_hops[i]["question"]
            new_a = new_hops[i]["answer"]
            old_a = old_hops[i]["answer"]
            # neighbors: pull other entries' hops (random external facts)
            out.append({
                "prompt": q,
                "target_new": new_a,
                "target_true": old_a,
                "paraphrases": [],
                "neighbors": [],  # MQuAKE single-hop doesn't ship a neighborhood; specificity stays 1.0
            })
        if limit and len(out) >= limit: break
    return out[:limit] if limit else out


def eval_one(tok, model, ex):
    p_new = logprob_first(tok, model, ex["prompt"], ex["target_new"])
    p_true = logprob_first(tok, model, ex["prompt"], ex["target_true"])
    eff = int(p_new > p_true)

    ps = []
    for pp in ex["paraphrases"]:
        ps.append(int(logprob_first(tok, model, pp, ex["target_new"]) > logprob_first(tok, model, pp, ex["target_true"])))
    para = sum(ps) / max(len(ps), 1)

    spec = []
    for (np_, nb_gt) in ex["neighbors"]:
        spec.append(int(logprob_first(tok, model, np_, nb_gt) > logprob_first(tok, model, np_, ex["target_new"])))
    sp = sum(spec) / max(len(spec), 1) if spec else 1.0
    return eff, para, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["counterfact", "zsre", "mquake", "wiki_recent", "wiki_cf"])
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--engram_weights", default=None)
    ap.add_argument("--lora_weights", default=None)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.engram_weights:
        from engram_peft import EngramModel
        print(f"[eval_ke] loading engram ckpt from {args.engram_weights}")
        model = EngramModel.from_pretrained(base, args.engram_weights, tokenizer=tok)
    elif args.lora_weights:
        from peft import PeftModel
        print(f"[eval_ke] loading LoRA ckpt from {args.lora_weights}")
        model = PeftModel.from_pretrained(base, args.lora_weights)
    else:
        model = base
    model.eval()

    loader = {"counterfact": load_counterfact, "zsre": load_zsre, "mquake": load_mquake,
              "wiki_recent": load_wiki_recent, "wiki_cf": load_wiki_cf}[args.dataset]
    cases = loader(args.limit)
    n = len(cases)
    print(f"[eval_ke] dataset={args.dataset} n={n}")

    eff_sum = ps_sum = sp_sum = 0.0
    for i, ex in enumerate(cases):
        e, p, s = eval_one(tok, model, ex)
        eff_sum += e; ps_sum += p; sp_sum += s
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{n}] eff={eff_sum/(i+1):.3f} ps={ps_sum/(i+1):.3f} ns={sp_sum/(i+1):.3f}")

    print(f"RESULT  ke_{args.dataset}_efficacy    acc={eff_sum/n:.4f}")
    print(f"RESULT  ke_{args.dataset}_paraphrase  acc={ps_sum/n:.4f}")
    print(f"RESULT  ke_{args.dataset}_specificity acc={sp_sum/n:.4f}")
    print("EVAL_KE_DONE")


if __name__ == "__main__":
    main()
