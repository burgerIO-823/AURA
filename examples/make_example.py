#!/usr/bin/env python3
"""Render the README inference example from a local copy of SpatialSoundQA.

    python examples/make_example.py --ssqa data/SpatialSoundQA

writes examples/example.wav: the binaural scene of the Stage-3 evaluation question
MIXUP_NONBINARY_DIRECTION-1156 ("Does the sound from the hammer emanate from the front or
behind side of the music's sound?", answer: front), rendered exactly as for training.
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preparation"))
from build_manifests import load_ssqa_rows  # noqa: E402
from encode_audio import HOP, SAMPLE_RATE, render_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssqa", type=Path, default=Path("data/SpatialSoundQA"))
    ap.add_argument("--question_type", default="MIXUP_NONBINARY_DIRECTION")
    ap.add_argument("--question_id", type=int, default=1156)
    ap.add_argument("--json", default="closed-end/stage3-mixup/eval-stage3-nonbinary.json")
    ap.add_argument("--out", type=Path, default=ROOT / "examples" / "example.wav")
    args = ap.parse_args()

    rows = [r for r in load_ssqa_rows(args.ssqa / args.json)
            if r["question_type"] == args.question_type and int(r["question_id"]) == args.question_id]
    if len(rows) != 1:
        raise SystemExit(f"found {len(rows)} rows for {args.question_type}-{args.question_id}")
    row = rows[0]
    mix, n_valid = render_scene(args.ssqa, row, {})
    sf.write(str(args.out), mix[:, : n_valid * HOP].T, SAMPLE_RATE, subtype="PCM_16")
    print(f"{args.out}\nQ: {row['question']}\nA: {row['answer']}")


if __name__ == "__main__":
    main()
