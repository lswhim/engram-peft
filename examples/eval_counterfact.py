#!/usr/bin/env python
"""COUNTERFACT eval for engram/base, ROME-style first-token logprob protocol.

For each test case (requested_rewrite + paraphrase_prompts + neighborhood_prompts):
  - Efficacy   : P_first(target_new) > P_first(target_true) on the main prompt.
  - Paraphrase : same comparison averaged over paraphrase_prompts.
  - Specificity: P_first(target_true) > P_first(target_new) on neighborhood_prompts
                 (these share the same target_true; an edit must NOT leak to them).

Engram ckpts load via EngramModel.from_pretrained(base, ckpt_dir, tokenizer=tok).
"""
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from engram_peft import EngramModel


def first_token_id(tok, target: str):
    # Autoregressive continuation: target appears after the prompt's last token,
    # so it conventionally begins with a leading space (Qwen2/3 BBPE merges " X").
    ids = tok(f" {target}", add_special_tokens=False).input_ids
    return ids[0] if ids else None


@torch.no_grad()
def logprob_first(tok, model, prompt: str, target: str) -> float:
    tid = first_token_id(tok, target)
    if tid is None:
        return float("-inf")
    enc = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1, :].float()
    return torch.log_softmax(logits, dim=-1)[tid].item()


def eval_case(tok, model, ex):
    rr = ex["requested_rewrite"]
    p_main = rr["prompt"].format(rr["subject"])
    new = rr["target_new"]["str"]
    true = rr["target_true"]["str"]

    # efficacy
    eff = int(logprob_first(tok, model, p_main, new) > logprob_first(tok, model, p_main, true))

    # paraphrase
    para = [int(logprob_first(tok, model, pp, new) > logprob_first(tok, model, pp, true))
            for pp in ex["paraphrase_prompts"]]
    ps = sum(para) / max(len(para), 1)

    # specificity (neighbors share target_true; edit must NOT flip them)
    spec = [int(logprob_first(tok, model, np_, true) > logprob_first(tok, model, np_, new))
            for np_ in ex["neighborhood_prompts"]]
    ns = sum(spec) / max(len(spec), 1)
    return eff, ps, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--engram_weights", default=None)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.engram_weights:
        print(f"[eval_counterfact] loading engram ckpt from {args.engram_weights}")
        model = EngramModel.from_pretrained(base, args.engram_weights, tokenizer=tok)
    else:
        model = base
    model.eval()

    test = load_dataset("azhx/counterfact", split="test")
    n = min(args.limit, len(test))
    test = test.select(range(n))

    eff_sum = ps_sum = ns_sum = 0.0
    for i, ex in enumerate(test):
        e, p, s = eval_case(tok, model, ex)
        eff_sum += e; ps_sum += p; ns_sum += s
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}] eff={eff_sum/(i+1):.3f} ps={ps_sum/(i+1):.3f} ns={ns_sum/(i+1):.3f}")

    print(f"RESULT  counterfact_efficacy    acc={eff_sum/n:.4f}")
    print(f"RESULT  counterfact_paraphrase  acc={ps_sum/n:.4f}")
    print(f"RESULT  counterfact_specificity acc={ns_sum/n:.4f}")
    print("EVAL_COUNTERFACT_DONE")


if __name__ == "__main__":
    main()
