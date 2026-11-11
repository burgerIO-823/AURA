#!/usr/bin/env python3
"""Train AURA (projector + LoRA) on a manifest of pre-encoded RVQ codes.

Each manifest row provides ``question``, ``narrative`` (the target: scene / geometric
supervision ending in ``Final answer: ...``) and ``codes_file`` (relative to --codes_dir).
The loss is the language-model loss on the target only; audio and prompt positions are -100.

    torchrun --standalone --nproc_per_node 4 training/train.py \
        --manifest data/manifests/train_c1.jsonl \
        --codes_dir data/codes --init runs/c1_align/final \
        --bs 4 --accum 1 --lr 1e-4 --warmup 100 --save_steps 1000 --out_dir runs/c1_joint

train.sh wraps this with the recipes in training/configs/*.env.
Checkpoints contain the four model files (see aura_model.py) + Trainer optimizer/scheduler state;
the final weights are written to <out_dir>/final.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from aura_model import CHECKPOINT_FILES, AuraQwen  # noqa: E402

try:  # keep log lines visible when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

T_AUDIO = 500     # fixed number of codec frames (10 s at 50 Hz)
MAX_TEXT = 512    # prompt + target token limit


def prompt_text(question):
    """Qwen chat format written by hand (no thinking template)."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def pad_codes(codes):
    """(8, T) -> (8, T_AUDIO) right-padded/truncated, plus the (T_AUDIO,) frame mask."""
    codes = codes.long()
    frames = codes.shape[1]
    mask = torch.zeros(T_AUDIO, dtype=torch.long)
    if frames >= T_AUDIO:
        codes = codes[:, :T_AUDIO]
        mask[:] = 1
    else:
        codes = torch.cat([codes, torch.zeros(8, T_AUDIO - frames, dtype=torch.long)], dim=1)
        mask[:frames] = 1
    return codes, mask


def load_codes(path):
    data = torch.load(path, map_location="cpu")
    codes_l, frame_mask = pad_codes(data["codes_left"])
    codes_r, _ = pad_codes(data["codes_right"])
    return codes_l, codes_r, frame_mask


def complete_checkpoint_step(path):
    """Global step of a fully written Trainer checkpoint, else None (e.g. after preemption)."""
    required = CHECKPOINT_FILES + ("optimizer.pt", "scheduler.pt",
                                   "trainer_state.json", "training_args.bin")
    if not os.path.isdir(path):
        return None
    if any(not os.path.isfile(os.path.join(path, n)) or not os.path.getsize(os.path.join(path, n))
           for n in required):
        return None
    if not glob.glob(os.path.join(path, "rng_state*.pth")):
        return None
    try:
        with open(os.path.join(path, "trainer_state.json"), encoding="utf-8") as stream:
            step = int(json.load(stream)["global_step"])
        dir_step = int(os.path.basename(path).rsplit("-", 1)[1])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return step if step == dir_step else None


def resolve_resume(value, out_dir):
    """--resume PATH must be complete; bare --resume picks the newest complete checkpoint."""
    if value is None:
        return None
    if value != "auto":
        if complete_checkpoint_step(value) is None:
            raise SystemExit(f"incomplete or invalid resume checkpoint: {value}")
        return value
    found = [(complete_checkpoint_step(p), p) for p in glob.glob(os.path.join(out_dir, "checkpoint-*"))]
    found = [(step, p) for step, p in found if step is not None]
    if not found:
        print("[resume] no complete checkpoint found, starting from initialization")
        return None
    path = max(found)[1]
    print(f"[resume] {path}")
    return path


class AuraDataset(Dataset):
    def __init__(self, items, codes_dir, tokenizer):
        self.items = items
        self.codes_dir = Path(codes_dir)
        self.tok = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        codes_l, codes_r, frame_mask = load_codes(self.codes_dir / item["codes_file"])
        p_ids = self.tok(prompt_text(item["question"]), add_special_tokens=False).input_ids
        t_ids = self.tok(f"{item['narrative']}<|im_end|>", add_special_tokens=False).input_ids
        return dict(input_ids=(p_ids + t_ids)[:MAX_TEXT],
                    labels=([-100] * len(p_ids) + t_ids)[:MAX_TEXT],
                    codes_l=codes_l, codes_r=codes_r, frame_mask=frame_mask)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        length = max(len(b["input_ids"]) for b in batch)
        ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * (length - n))
            labels.append(b["labels"] + [-100] * (length - n))
            attn.append([1] * n + [0] * (length - n))
        return dict(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn),
                    labels=torch.tensor(labels),
                    codes_l=torch.stack([b["codes_l"] for b in batch]),
                    codes_r=torch.stack([b["codes_r"] for b in batch]),
                    frame_mask=torch.stack([b["frame_mask"] for b in batch]))


class AuraTrainer(Trainer):
    """Checkpoints hold only the trained parts, never the frozen 8B backbone."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # optimizer / scheduler / RNG / global step are restored by Trainer from the same dir.
        (model or self.model).load_weights(resume_from_checkpoint)


class FlushLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs and state.is_world_process_zero:
            row = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in logs.items()}
            print(f"[log step {state.global_step}] {row}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--codes_dir", default="data/codes")
    ap.add_argument("--qwen", default="models/Qwen3-8B")
    ap.add_argument("--codebooks", default="models/xcodec/codebooks.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init", default=None,
                    help="warm start from a checkpoint directory (weights only; fresh optimizer and schedule)")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume Trainer state from a checkpoint; bare --resume picks the newest complete one")
    ap.add_argument("--freeze_llm", action="store_true",
                    help="projector alignment: freeze the LLM including LoRA, train only the projector")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="aura")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="one forward + backward pass with gradient checkpointing, then exit")
    args = ap.parse_args()
    if args.init and args.resume:
        raise SystemExit("--init (new warm start) and --resume are mutually exclusive")

    set_seed(args.seed)
    args.resume = resolve_resume(args.resume, args.out_dir)
    tok = AutoTokenizer.from_pretrained(args.qwen)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # The dataset order is a seeded shuffle of the manifest; Trainer then samples with data_seed.
    items = [json.loads(line) for line in open(args.manifest, encoding="utf-8") if line.strip()]
    random.Random(args.seed).shuffle(items)

    model = AuraQwen(args.qwen, args.codebooks)
    if args.init:
        model.load_weights(args.init)
        print(f"[init] weights from {args.init}")
    if args.freeze_llm:
        for p in model.llm.parameters():
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in model.proj.parameters())
    print(f"trainable parameters: {n_train / 1e6:.2f}M (projector {n_proj / 1e6:.2f}M"
          f"{', LLM frozen' if args.freeze_llm else ' + LoRA'}) | training rows: {len(items)}")

    collate = Collator(tok.pad_token_id)
    train_ds = AuraDataset(items, args.codes_dir, tok)

    if args.smoke:
        model.cuda().train()   # gradient checkpointing only applies in training mode
        model.enable_grad_checkpointing()
        batch = collate([train_ds[i % len(train_ds)] for i in range(args.bs)])
        batch = {k: v.cuda() for k, v in batch.items()}
        torch.cuda.reset_peak_memory_stats()
        out = model(**batch)
        out.loss.backward()
        n_audio = model.roles_per_frame * ((T_AUDIO + model.time_pool - 1) // model.time_pool)
        print(f"[smoke] loss={out.loss.item():.4f} logits={tuple(out.logits.shape)} "
              f"(audio {n_audio} + text {batch['input_ids'].shape[1]}) bs={args.bs} "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return

    use_wandb = args.wandb or bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    model.enable_grad_checkpointing()
    targs = TrainingArguments(
        output_dir=args.out_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, warmup_steps=args.warmup, lr_scheduler_type="cosine",
        bf16=True, logging_steps=10, save_steps=args.save_steps, save_total_limit=None,
        eval_strategy="no", prediction_loss_only=True,
        report_to=("wandb" if use_wandb else "none"), run_name=args.wandb_run,
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=args.num_workers, dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, seed=args.seed, data_seed=args.seed)
    trainer = AuraTrainer(model=model, args=targs, train_dataset=train_ds,
                          data_collator=collate, callbacks=[FlushLogCallback()])
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        model.save_pretrained(os.path.join(args.out_dir, "final"))
        print(f"done -> {args.out_dir}/final")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
