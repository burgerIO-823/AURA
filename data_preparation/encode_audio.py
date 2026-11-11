#!/usr/bin/env python3
"""Render the SpatialSoundQA binaural scenes and encode them with the frozen X-Codec RVQ.

    single source   wav (*) BRIR
    two sources     0.5 * (wav1 (*) BRIR1 + wav2 (*) BRIR2)
Dry AudioSet clips are resampled to 16 kHz and keep their raw amplitude; the mixture is
peak-limited only if it clips, then truncated to 10 s.  Each ear is encoded separately at
4 kbps (8 RVQ layers, 50 Hz) and stored as {codes_left, codes_right} int16 (8, T), one file per
unique scene (955,422 across all SpatialSoundQA question files).  Finished files are skipped, so
an interrupted run can simply be restarted.

    python data_preparation/encode_audio.py --ssqa data/SpatialSoundQA --codes_dir data/codes

Before encoding, every referenced wav / BRIR is checked to exist; after encoding, every scene is
checked to have its codes file (--verify also loads each one).
Requires the X-Codec repository at $XCODEC_ROOT (default third_party/xcodec).
"""
import argparse
import fractions
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifests import codes_file_name, load_ssqa_rows  # noqa: E402

SAMPLE_RATE = 16000
MAX_SAMPLES = 160000   # 10 s
HOP = 320              # one codec frame (50 Hz)


# ------------------------------------------------------------------------- X-Codec model
def load_xcodec(checkpoint_dir):
    """X-Codec general-audio model (config_hubert_general.yaml + xcodec_hubert_general_audio_v2.pth)."""
    root = os.environ.get("XCODEC_ROOT", "third_party/xcodec")
    for path in (root, os.path.join(root, "descriptaudiocodec")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from models.soundstream_semantic import SoundStream
    from omegaconf import OmegaConf

    config = OmegaConf.load(os.path.join(checkpoint_dir, "config_hubert_general.yaml"))
    model = SoundStream(**config.generator.config)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "xcodec_hubert_general_audio_v2.pth"),
                                     map_location="cpu"))
    return model.cuda().eval()


@torch.no_grad()
def encode_batch(model, x, bw=4.0):
    """x (B, 1, L) waveforms -> codes (n_q, B, T); a batch-safe version of SoundStream.encode."""
    e_sem = model.encoder_semantic(model.get_regress_target(x).detach().transpose(1, 2))
    e_aco = model.encoder(x)
    if e_aco.shape[2] != e_sem.shape[2]:
        e_aco = model.encoder(F.pad(x[:, 0, :], (160, 160)).unsqueeze(1))
    e = model.fc_prior(torch.cat([e_aco, e_sem], dim=1).transpose(1, 2)).transpose(1, 2)
    _, codes, _, _ = model.quantizer(e.float(), model.frame_rate, bw)
    return codes


# ------------------------------------------------------------------- binaural rendering
def resample_to_16k(wav, sr):
    if sr == SAMPLE_RATE:
        return wav
    frac = fractions.Fraction(SAMPLE_RATE, sr)
    return resample_poly(wav.astype(np.float64), frac.numerator, frac.denominator, axis=-1)


def wav_path(ssqa, audio_id):
    return ssqa / "AudioSet" / f"{audio_id}.wav"


def brir_path(ssqa, reverb_id):
    return ssqa / "mp3d_reverb" / "binaural" / reverb_id


def render_source(ssqa, audio_id, reverb_id, brir_cache):
    """One dry AudioSet clip convolved with its binaural RIR -> (2, n) at 16 kHz."""
    if not audio_id or not reverb_id:
        return None
    if reverb_id not in brir_cache:
        brir_cache[reverb_id] = np.load(str(brir_path(ssqa, reverb_id)))
    wav, sr = sf.read(str(wav_path(ssqa, audio_id)))
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    dry = resample_to_16k(wav, sr).astype(np.float32)
    brir = brir_cache[reverb_id]
    out = np.zeros((2, dry.shape[-1] + brir.shape[-1] - 1))
    for ch in range(2):
        out[ch] = fftconvolve(dry, brir[ch])
    return out.astype(np.float32)


def render_scene(ssqa, row, brir_cache):
    """Binaural scene of a manifest row -> ((2, MAX_SAMPLES) waveform, number of valid frames)."""
    first = render_source(ssqa, row["audio_id"], row["reverb_id"], brir_cache)
    second = render_source(ssqa, row.get("audio_id2"), row.get("reverb_id2"), brir_cache)
    if second is not None:
        mix = np.zeros((2, max(first.shape[1], second.shape[1])), dtype=np.float32)
        mix[:, :first.shape[1]] += first
        mix[:, :second.shape[1]] += second
        mix *= 0.5
    else:
        mix = first
    peak = float(np.abs(mix).max())
    if peak > 1.0:
        mix *= 0.95 / peak
    n = min(MAX_SAMPLES, mix.shape[1])
    out = np.zeros((2, MAX_SAMPLES), dtype=np.float32)
    out[:, :n] = mix[:, :n]
    return out, min(MAX_SAMPLES // HOP, int(math.ceil(n / HOP)))


# ------------------------------------------------------------------------------ pipeline
def unique_scenes(question_files):
    """codes file name -> SpatialSoundQA row, one entry per distinct (audio, BRIR) scene."""
    scenes = {}
    for path in question_files:
        for row in load_ssqa_rows(path):
            name = codes_file_name(row["audio_id"], row["reverb_id"], row.get("audio_id2"), row.get("reverb_id2"))
            scenes.setdefault(name, row)
    return scenes


def check_assets(ssqa, rows):
    missing = set()
    for row in rows:
        for audio_key, reverb_key in (("audio_id", "reverb_id"), ("audio_id2", "reverb_id2")):
            if row.get(audio_key):
                for path in (wav_path(ssqa, row[audio_key]), brir_path(ssqa, row[reverb_key])):
                    if not path.is_file():
                        missing.add(str(path))
    return sorted(missing)


def check_codes(codes_dir, names, deep):
    bad = []
    for name in names:
        path = codes_dir / name
        if not path.is_file() or not path.stat().st_size:
            bad.append((name, "missing"))
        elif deep:
            try:
                data = torch.load(path, map_location="cpu")
                left, right = data["codes_left"], data["codes_right"]
                if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 8:
                    bad.append((name, f"bad shape {tuple(left.shape)} / {tuple(right.shape)}"))
            except Exception as error:   # corrupt file
                bad.append((name, repr(error)))
    return bad


def encode(ssqa, pending, codes_dir, xcodec_dir, bw, batch_size, workers, backlog):
    model = load_xcodec(xcodec_dir)
    brir_cache = {}
    start, done = time.time(), 0

    def render(item):
        name, row = item
        mix, n_valid = render_scene(ssqa, row, brir_cache)
        return name, row, mix, n_valid

    def flush(batch):
        x = torch.from_numpy(np.stack([ch for _, _, mix, _ in batch for ch in mix])).unsqueeze(1).cuda()
        codes = encode_batch(model, x, bw).cpu()
        for j, (name, row, _, n_valid) in enumerate(batch):
            payload = {
                "codes_left": codes[:, 2 * j, :n_valid].to(torch.int16),
                "codes_right": codes[:, 2 * j + 1, :n_valid].to(torch.int16),
                "audio_id": row["audio_id"], "reverb_id": row["reverb_id"],
                "audio_id2": row.get("audio_id2"), "reverb_id2": row.get("reverb_id2"),
                "mix_scale": 0.5 if row.get("audio_id2") else 1.0,
                "preprocessing": {"sample_rate_hz": SAMPLE_RATE, "xcodec_bandwidth": bw,
                                  "conditional_peak_limit": True},
            }
            tmp = codes_dir / f"{name}.tmp"   # atomic write: a crash never leaves a partial .pt
            torch.save(payload, str(tmp))
            os.replace(tmp, codes_dir / name)
        return len(batch)

    # CPU threads render ahead (scipy releases the GIL) while the GPU encodes, in input order.
    batch = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator, queued = iter(pending), deque()
        for item in iterator:
            queued.append(pool.submit(render, item))
            if len(queued) >= backlog:
                break
        while queued:
            batch.append(queued.popleft().result())
            nxt = next(iterator, None)
            if nxt is not None:
                queued.append(pool.submit(render, nxt))
            if len(batch) == batch_size:
                done += flush(batch)
                batch = []
                rate = done / (time.time() - start)
                print(f"[encode] {done}/{len(pending)} {rate:.1f}/s "
                      f"eta {(len(pending) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
        if batch:
            done += flush(batch)
    print(f"[encode] {done} scenes in {(time.time() - start) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssqa", type=Path, default=Path("data/SpatialSoundQA"))
    ap.add_argument("--jsons", type=Path, nargs="+", default=None,
                    help="question files to take scenes from (default: every closed-end/*/*.json)")
    ap.add_argument("--codes_dir", type=Path, default=Path("data/codes"))
    ap.add_argument("--xcodec_dir", default="models/xcodec")
    ap.add_argument("--bw", type=float, default=4.0, help="X-Codec bandwidth (kbps); 4.0 = 8 RVQ layers")
    ap.add_argument("--batch", type=int, default=16, help="scenes per GPU batch (2 channels each)")
    ap.add_argument("--render_workers", type=int, default=4)
    ap.add_argument("--render_backlog", type=int, default=16)
    ap.add_argument("--verify", action="store_true", help="also load every codes file after encoding")
    args = ap.parse_args()

    question_files = args.jsons or sorted((args.ssqa / "closed-end").glob("*/*.json"))
    if not question_files:
        raise SystemExit(f"no SpatialSoundQA question files under {args.ssqa / 'closed-end'}")
    scenes = unique_scenes(question_files)
    args.codes_dir.mkdir(parents=True, exist_ok=True)
    pending = [(name, row) for name, row in scenes.items()
               if not (args.codes_dir / name).is_file() or not (args.codes_dir / name).stat().st_size]
    print(f"[encode] {len(scenes)} unique scenes, {len(pending)} to encode")

    missing = check_assets(args.ssqa, [row for _, row in pending])
    if missing:
        raise SystemExit(f"{len(missing)} SpatialSoundQA files are missing, e.g. {missing[:3]}")
    if pending:
        encode(args.ssqa, pending, args.codes_dir, args.xcodec_dir, args.bw, args.batch,
               args.render_workers, max(args.render_backlog, args.render_workers))

    bad = check_codes(args.codes_dir, sorted(scenes), args.verify)
    if bad:
        raise SystemExit(f"[check] {len(bad)} of {len(scenes)} codes files missing/invalid, e.g. {bad[:3]}")
    print(f"[check] all {len(scenes)} codes files present{' and loadable' if args.verify else ''}")


if __name__ == "__main__":
    main()
