#!/usr/bin/env python3
"""Extract the 8 RVQ codebooks from an X-Codec checkpoint.

AURA only needs the frozen codebooks (8 x 1024 x 1024) for an embedding lookup of the
pre-encoded codes, so training and evaluation never load the X-Codec model itself.

    python data_preparation/extract_codebooks.py \
        --ckpt models/xcodec/xcodec_hubert_general_audio_v2.pth --out models/xcodec/codebooks.pt
"""
import argparse

import torch

N_Q = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="models/xcodec/xcodec_hubert_general_audio_v2.pth")
    ap.add_argument("--out", default="models/xcodec/codebooks.pt")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    tables = []
    for i in range(N_Q):
        key = f"quantizer.vq.layers.{i}._codebook.embed"
        if key not in state:
            raise KeyError(f"missing codebook key {key}; unexpected checkpoint layout")
        tables.append(state[key].float())
    codebooks = torch.stack(tables, 0).contiguous()
    assert codebooks.shape == (N_Q, 1024, 1024), codebooks.shape
    torch.save(codebooks, args.out)
    print(f"codebooks {tuple(codebooks.shape)} ({codebooks.numel() * 4 / 1e6:.0f} MB fp32) -> {args.out}")


if __name__ == "__main__":
    main()
