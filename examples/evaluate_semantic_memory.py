#!/usr/bin/env python
"""Evaluate a semantic-memory manifest with auditable query-level outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engram-weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--geometry-cache", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--evaluation-cohort",
        help="Name of the trained-through WikiBigEdit timestep for retention audits.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "qa"),
        default="plain",
        help="Use the WikiBigEdit Q:/A: evaluation template when set to qa.",
    )
    parser.add_argument(
        "--locality-mode",
        choices=("answer_accuracy", "pre_post_preservation"),
        default="answer_accuracy",
        help="Official WikiBigEdit locality compares edited predictions with the base model.",
    )
    return parser.parse_args()


def read_manifest(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    cases=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): cases.append(json.loads(line))
            if limit and len(cases) >= limit: break
    return cases


def normalized_aliases(values: list[str]) -> set[str]:
    return {" ".join(value.lower().strip().split()) for value in values if value.strip()}


def first_nonempty(values: list[str]) -> str | None:
    return next((str(value).strip() for value in values if str(value).strip()),None)


def formatted_pair(tokenizer: Any, prompt: str, answer: str, prompt_format: str) -> tuple[list[int], list[int]]:
    if prompt_format == "qa":
        prefix = f"Q: {prompt} A:"
        prompt_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    else:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    return prompt_ids, answer_ids


@torch.inference_mode()
def target_token_predictions(
    tokenizer: Any,
    model: Any,
    pairs: list[tuple[str, str]],
    batch_size: int,
    prompt_format: str = "plain",
) -> tuple[list[float], list[list[int]]]:
    encoded=[]
    for prompt, answer in pairs:
        prompt_ids, answer_ids = formatted_pair(tokenizer, prompt, answer, prompt_format)
        encoded.append((prompt_ids+answer_ids,len(prompt_ids)))
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    scores=[]; token_predictions=[]
    for start in range(0,len(encoded),batch_size):
        batch=encoded[start:start+batch_size]; width=max(len(ids) for ids,_ in batch)
        ids=torch.full((len(batch),width),pad,dtype=torch.long,device=model.device); mask=torch.zeros_like(ids)
        for row,(tokens,_) in enumerate(batch): ids[row,:len(tokens)]=torch.tensor(tokens,device=model.device); mask[row,:len(tokens)]=1
        predictions=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1].argmax(-1)
        for row,(tokens,prompt_len) in enumerate(batch):
            gold=ids[row,prompt_len:len(tokens)]; pred=predictions[row,prompt_len-1:len(tokens)-1]
            scores.append(float((pred==gold).float().mean()) if gold.numel() else 0.0)
            token_predictions.append(pred.detach().cpu().tolist())
        complete=min(start+len(batch),len(encoded))
        if complete==len(encoded) or complete//1000 != start//1000:
            print(f"[target-score {complete}/{len(encoded)}]",flush=True)
    return scores, token_predictions


def target_token_accuracy(
    tokenizer: Any,
    model: Any,
    pairs: list[tuple[str, str]],
    batch_size: int,
    prompt_format: str = "plain",
) -> list[float]:
    scores, _ = target_token_predictions(
        tokenizer, model, pairs, batch_size, prompt_format
    )
    return scores


def prediction_preservation(before: list[int], after: list[int]) -> float:
    if len(before) != len(after):
        raise ValueError("pre/post predictions must have identical lengths")
    return float(np.mean(np.equal(before, after))) if before else 0.0


@torch.inference_mode()
def semantic_cosines(text_pairs: list[tuple[str,str]], model_name: str, batch_size: int) -> list[float]:
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    encoder=AutoModel.from_pretrained(model_name,dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
    texts=[text for pair in text_pairs for text in pair]; vectors=[]
    for start in range(0,len(texts),batch_size):
        batch=tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=128,return_tensors="pt").to("cuda")
        hidden=encoder(**batch).last_hidden_state; lengths=batch["attention_mask"].sum(1)-1
        vector=hidden[torch.arange(hidden.size(0),device=hidden.device),lengths]
        vectors.append(torch.nn.functional.normalize(vector.float(),dim=-1).cpu())
        complete=min(start+len(batch["input_ids"]),len(texts))
        if complete==len(texts) or complete//2000 != start//2000:
            print(f"[geometry-embed {complete}/{len(texts)}]",flush=True)
    matrix=torch.cat(vectors).reshape(-1,2,vectors[0].shape[-1])
    del encoder
    torch.cuda.empty_cache()
    return (matrix[:,0]*matrix[:,1]).sum(-1).tolist()


def quartile_bins(rows: list[dict[str, Any]]) -> None:
    if not rows or not all(isinstance(row.get("semantic_similarity"),(int,float)) for row in rows): return
    semantic=np.array([row["semantic_similarity"] for row in rows]); lexical=np.array([row.get("lexical_similarity",0.0) for row in rows])
    sq=np.quantile(semantic,[.25,.5,.75]); lq=np.quantile(lexical,[.25,.5,.75])
    for row,s,l in zip(rows,semantic,lexical,strict=True):
        row["semantic_bin"] = "low" if s <= sq[0] else "mid_low" if s <= sq[1] else "mid_high" if s <= sq[2] else "high"
        row["lexical_bin"] = "low" if l <= lq[0] else "mid_low" if l <= lq[1] else "mid_high" if l <= lq[2] else "high"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str,list[float]]=defaultdict(list)
    for row in rows:
        if not row["eligible"]: continue
        groups[f"axis/{row['axis']}"] .append(row["accuracy"])
        groups[f"role/{row['role']}"] .append(row["accuracy"])
        if row.get("cohort_origin") is not None:
            groups[f"cohort/{row['cohort_origin']}/{row['axis']}"] .append(row["accuracy"])
        if row.get("semantic_bin"): groups[f"geometry/{row['semantic_bin']}_{row['lexical_bin']}"] .append(row["accuracy"])
    return {key:{"mean":sum(values)/len(values),"n":len(values)} for key,values in sorted(groups.items())}


def main() -> None:
    args=parse_args(); cases=read_manifest(args.manifest,args.limit)
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    base=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,low_cpu_mem_usage=True).to("cuda")
    base.eval(); query_specs=[]; condition_specs=[]; skipped_unscorable=0; skipped_conditions=0
    for case in cases:
        for query in case["queries"]:
            answer=first_nonempty(query.get("answers",[]))
            if answer is None:
                skipped_unscorable+=1
                continue
            query_specs.append((case,query,answer))
            for prompt,answers in zip(query.get("condition_prompts",[]),query.get("condition_answers",[]),strict=True):
                condition_answer=first_nonempty(answers)
                if condition_answer is None: skipped_conditions+=1
                else: condition_specs.append((len(query_specs)-1,prompt,condition_answer))
    pairs=[(q["prompt"],answer) for _,q,answer in query_specs]
    locality_indices=[index for index,(_,query,_) in enumerate(query_specs) if query["axis"]=="locality"]
    base_locality_predictions: dict[int,list[int]]={}
    if args.locality_mode == "pre_post_preservation" and locality_indices:
        _, predictions=target_token_predictions(
            tokenizer,
            base,
            [pairs[index] for index in locality_indices],
            args.batch_size,
            args.prompt_format,
        )
        base_locality_predictions=dict(zip(locality_indices,predictions,strict=True))

    model: Any=base
    if args.engram_weights:
        from engram_peft import EngramModel
        model=EngramModel.from_pretrained(base,args.engram_weights,tokenizer=tokenizer).to("cuda")
    model.eval()
    query_scores,query_predictions=target_token_predictions(
        tokenizer,model,pairs,args.batch_size,args.prompt_format
    )
    if args.locality_mode == "pre_post_preservation":
        for index,pre in base_locality_predictions.items():
            post=query_predictions[index]
            try:
                query_scores[index]=prediction_preservation(pre,post)
            except ValueError as error:
                raise RuntimeError(
                    f"locality token length mismatch at query {index}"
                ) from error
    condition_scores=target_token_accuracy(tokenizer,model,[(prompt,answer) for _,prompt,answer in condition_specs],args.batch_size,args.prompt_format) if condition_specs else []
    condition_by_query: dict[int,list[bool]]=defaultdict(list)
    for (index,_,_),score in zip(condition_specs,condition_scores,strict=True): condition_by_query[index].append(score==1.0)
    rows=[]
    for index,((case,query,_),score) in enumerate(zip(query_specs,query_scores,strict=True)):
        checks=condition_by_query[index]
        eligible=(all(checks) if query.get("condition","OR")=="AND" else any(checks)) if checks else True
        metadata = case.get("metadata", {})
        rows.append({"case_id":case["case_id"],"prompt":query["prompt"],"axis":query["axis"],"role":query["role"],"accuracy":score,"eligible":eligible,"lexical_similarity":query.get("lexical_similarity"),"cohort_origin":metadata.get("cohort_origin") or metadata.get("timestep"),"evaluated_at":args.evaluation_cohort or metadata.get("evaluated_at")})
    geometry_indices=[i for i,row in enumerate(rows) if row["axis"]=="unseen_template"]
    if geometry_indices:
        cached: dict[str, float] = {}
        if args.geometry_cache and args.geometry_cache.is_file():
            cached = json.loads(args.geometry_cache.read_text())
        missing_indices=[]; pairs=[]
        for i in geometry_indices:
            key=f"{rows[i]['case_id']}\t{rows[i]['prompt']}"
            if key in cached: rows[i]["semantic_similarity"]=cached[key]
            else:
                missing_indices.append(i)
                pairs.append((query_specs[i][0].get("metadata",{}).get("canonical_geometry_text",query_specs[i][0]["prompt"]),query_specs[i][1].get("geometry_text") or query_specs[i][1]["prompt"]))
        if pairs:
            for i,cosine in zip(missing_indices,semantic_cosines(pairs,args.embedder,args.embed_batch_size),strict=True):
                rows[i]["semantic_similarity"]=cosine
                cached[f"{rows[i]['case_id']}\t{rows[i]['prompt']}"]=cosine
            if args.geometry_cache:
                args.geometry_cache.parent.mkdir(parents=True,exist_ok=True)
                temporary=args.geometry_cache.with_suffix(".tmp")
                temporary.write_text(json.dumps(cached),encoding="utf-8"); os.replace(temporary,args.geometry_cache)
        quartile_bins([rows[i] for i in geometry_indices])
    payload={"status":"complete","cases":len(cases),"queries":len(rows),"eligible_queries":sum(row["eligible"] for row in rows),"skipped_unscorable_queries":skipped_unscorable,"skipped_unscorable_conditions":skipped_conditions,"metrics":aggregate(rows),"protocol":{"condition_gated":True,"score":"teacher_forced_complete_target_token_accuracy","prompt_format":args.prompt_format,"locality_mode":args.locality_mode,"evaluation_cohort":args.evaluation_cohort,"geometry_bins":"within-run quartiles"}}
    args.output.parent.mkdir(parents=True,exist_ok=True); samples=args.output.with_suffix(".jsonl")
    with samples.open("w",encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    payload["samples"]=str(samples); temporary=args.output.with_suffix(".tmp"); temporary.write_text(json.dumps(payload,indent=2),encoding="utf-8"); os.replace(temporary,args.output)
    print(json.dumps(payload,indent=2))


if __name__ == "__main__": main()
