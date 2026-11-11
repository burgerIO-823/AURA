#!/usr/bin/env python3
"""Ask an AURA checkpoint a question about a binaural recording.

    python inference.py --ckpt checkpoints/stage3_c3 \
        --audio example_binaural.wav \
        --question "Is the sound of the dog barking to the left of the sound of the piano?"

``--audio`` must be a 2-channel (left, right) binaural waveform; it is resampled to 16 kHz,
truncated to 10 s and encoded per ear with the frozen X-Codec RVQ (requires the X-Codec
repository at $XCODEC_ROOT and the checkpoint in models/xcodec, see setup_models.sh).
A pre-encoded ``--codes`` file produced by data_preparation/encode_audio.py can be given instead.

The model first describes every source (identity, direction, distance), then gives a
``Final answer:`` line.  Use --max_new_tokens 384 for Stage-3 relational questions.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
for sub in ("model", "training", "evaluation", "data_preparation"):
    sys.path.insert(0, str(ROOT / sub))

from generate import generate_batch, load_model  # noqa: E402
from train import pad_codes  # noqa: E402

SAMPLE_RATE = 16000
MAX_SAMPLES = 160000   # 10 s
HOP = 320              # X-Codec frame hop at 16 kHz (50 Hz)


def encode_wav(path, xcodec_dir, bw=4.0):
    """Stereo wav -> (codes_left, codes_right), each (8, T), with the training-time pipeline."""
    import soundfile as sf
    from encode_audio import encode_batch, load_xcodec, resample_to_16k

    wav, sr = sf.read(str(path), always_2d=True)          # (T, C)
    if wav.shape[1] != 2:
        raise SystemExit(f"{path}: expected 2 channels (binaural L/R), found {wav.shape[1]}")
    wav = resample_to_16k(wav.T, sr).astype(np.float32)   # (2, T)
    peak = float(np.abs(wav).max())
    if peak > 1.0:
        wav *= 0.95 / peak
    n = min(MAX_SAMPLES, wav.shape[1])
    buf = np.zeros((2, MAX_SAMPLES), dtype=np.float32)
    buf[:, :n] = wav[:, :n]
    n_valid = min(MAX_SAMPLES // HOP, int(math.ceil(n / HOP)))
    codec = load_xcodec(xcodec_dir)
    codes = encode_batch(codec, torch.from_numpy(buf).unsqueeze(1).cuda(), bw).cpu()  # (8, 2, T)
    del codec
    torch.cuda.empty_cache()
    return codes[:, 0, :n_valid], codes[:, 1, :n_valid]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="AURA checkpoint directory, e.g. checkpoints/stage3_c3")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="binaural 2-channel wav")
    src.add_argument("--codes", help="pre-encoded .pt with codes_left / codes_right")
    ap.add_argument("--question", required=True)
    ap.add_argument("--qwen", default="models/Qwen3-8B")
    ap.add_argument("--codebooks", default="models/xcodec/codebooks.pt")
    ap.add_argument("--xcodec_dir", default="models/xcodec",
                    help="directory with config_hubert_general.yaml + xcodec_hubert_general_audio_v2.pth")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    args = ap.parse_args()

    if args.audio:
        os.environ.setdefault("XCODEC_ROOT", str(ROOT / "third_party" / "xcodec"))
        codes_l, codes_r = encode_wav(args.audio, args.xcodec_dir)
    else:
        data = torch.load(args.codes, map_location="cpu")
        codes_l, codes_r = data["codes_left"], data["codes_right"]

    cl, fm = pad_codes(codes_l)
    cr, _ = pad_codes(codes_r)
    model, tok = load_model(args.ckpt, args.qwen, args.codebooks)
    print(generate_batch(model, tok, [args.question], [(cl, cr, fm)], args.max_new_tokens)[0])


if __name__ == "__main__":
    main()
