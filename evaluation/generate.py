#!/usr/bin/env python3
"""Greedy generation of an AURA checkpoint on an evaluation manifest.

Writes one JSONL record per item:
    {"subset": "held-out", "key", "qt", "gt", "pred" (final answer), "raw" (full generation)}
which evaluation/evaluate.py scores with the official BAT/SpatialSoundQA metrics.

    python evaluation/generate.py --ckpt checkpoints/stage3_c3 \
        --manifest data/manifests/eval_stage3.jsonl --codes_dir data/codes \
        --max_new_tokens 384 --batch_size 128 --num_gpus 4 --out pred.jsonl

Items are processed in a fixed seed-42 order.  With --num_gpus N the order is split into N
shards, one process per GPU, and the shards are merged into --out when all are complete.
Generation is resumable: rerunning skips every item already written.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "training"))
from aura_model import AuraQwen  # noqa: E402
from metrics import extract_final_answer  # noqa: E402
from train import load_codes, prompt_text  # noqa: E402

SUBSET = "held-out"


def load_model(ckpt, qwen, codebooks):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(qwen)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AuraQwen.from_pretrained(ckpt, qwen, codebooks)
    return model.prepare_for_generation().cuda().eval(), tok


@torch.no_grad()
def generate_batch(model, tok, questions, inputs, max_new_tokens):
    """inputs: list of (codes_l, codes_r, frame_mask).  Prompts are left-padded after the
    fixed-length audio tokens so every row ends with its real assistant prompt."""
    device = next(model.proj.parameters()).device
    codes_l = torch.stack([x[0] for x in inputs]).to(device)
    codes_r = torch.stack([x[1] for x in inputs]).to(device)
    frame_mask = torch.stack([x[2] for x in inputs]).to(device)
    audio_embeds, pooled_mask = model.encode_audio(codes_l, codes_r, frame_mask)
    old_side, tok.padding_side = tok.padding_side, "left"
    try:
        enc = tok([prompt_text(q) for q in questions], add_special_tokens=False,
                  padding=True, return_tensors="pt")
    finally:
        tok.padding_side = old_side
    p_ids, p_attn = enc.input_ids.to(device), enc.attention_mask.to(device)
    embeds = torch.cat([audio_embeds, model.llm.get_input_embeddings()(p_ids)], dim=1)
    attn = torch.cat([model.audio_attention(pooled_mask), p_attn], dim=1)
    out = model.llm.generate(inputs_embeds=embeds, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             use_cache=True, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    # With inputs_embeds, generate() returns only the new tokens.
    return [tok.decode(row, skip_special_tokens=True).strip() for row in out]


def load_resumable_records(path):
    """Completed records by key.  Only an interrupted (non-newline-terminated) last line is
    repaired; any other malformed line is treated as corruption."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if raw.endswith(b"\n"):
                    raise SystemExit(f"resume JSONL corruption at byte {offset}: {error}") from error
                with open(path, "r+b") as repair:
                    repair.truncate(offset)
                print(f"[resume] dropped interrupted trailing line at byte {offset}")
                break
            if rec.get("key") in done:
                raise SystemExit(f"duplicate record in {path}: {rec.get('key')!r}")
            done[rec["key"]] = rec
    return done


def ordered_items(manifest, limit=0):
    items = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
    random.Random(42).shuffle(items)
    return items[:limit] if limit else items


def generate_shard(args, items, out):
    done = load_resumable_records(out)
    todo = [it for it in items if it["key"] not in done]
    print(f"[generate] {out}: {len(items)} items, {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    model, tok = load_model(args.ckpt, args.qwen, args.codebooks)
    codes_dir = Path(args.codes_dir)
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    def read(batch):
        return [load_codes(codes_dir / it["codes_file"]) for it in batch]

    # One background reader prefetches the next batch of codes while the GPU generates.
    with open(out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read, batches[0])
        finished = 0
        for i, batch in enumerate(batches):
            inputs = future.result()
            if i + 1 < len(batches):
                future = pool.submit(read, batches[i + 1])
            texts = generate_batch(model, tok, [it["question"] for it in batch], inputs, args.max_new_tokens)
            for it, text in zip(batch, texts):
                fout.write(json.dumps({"subset": SUBSET, "key": it["key"], "qt": it["question_type"],
                                       "gt": it["answer"], "pred": extract_final_answer(text),
                                       "raw": text}, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())   # a finished batch is the durability boundary for resuming
            finished += len(batch)
            print(f"[generate] {finished}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="AURA checkpoint directory")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--codes_dir", default="data/codes")
    ap.add_argument("--qwen", default="models/Qwen3-8B")
    ap.add_argument("--codebooks", default="models/xcodec/codebooks.pt")
    ap.add_argument("--max_new_tokens", type=int, required=True,
                    help="decode budget of the protocol: 192 for stages 1/2, 384 for stage 3")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_gpus", type=int, default=1, help="one generation process per GPU")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke tests)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)   # internal: "index/count"
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    items = ordered_items(args.manifest, args.limit)
    if not args.shard and len(load_resumable_records(args.out)) == len(items):
        print(f"[generate] {args.out} is already complete")
        return

    if args.shard:                                   # worker process for one GPU
        index, count = map(int, args.shard.split("/"))
        generate_shard(args, items[index::count], f"{args.out}.part{index}")
        return
    if args.num_gpus <= 1:
        generate_shard(args, items, args.out)
    else:
        argv = [a for a in sys.argv[1:]]
        procs = [subprocess.Popen([sys.executable, "-u", __file__, *argv, "--shard", f"{i}/{args.num_gpus}"],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)})
                 for i in range(args.num_gpus)]
        if any(proc.wait() for proc in procs):
            raise SystemExit("a generation shard failed; rerun the same command to resume")
        records = {}
        for i in range(args.num_gpus):
            records.update(load_resumable_records(f"{args.out}.part{i}"))
        with open(args.out, "w", encoding="utf-8") as fout:
            for it in items:
                if it["key"] in records:
                    fout.write(json.dumps(records[it["key"]], ensure_ascii=False) + "\n")
    written = len(load_resumable_records(args.out))
    if written != len(items):
        raise SystemExit(f"{args.out}: {written} of {len(items)} predictions")
    print(f"[generate] {written} predictions -> {args.out}")


if __name__ == "__main__":
    main()
