#!/usr/bin/env python3
"""Model-tagged embedding caches for the classification mAP (OpenAI text embeddings).

    python evaluation/embeddings.py --predictions pred.jsonl --subset held-out \
        --model text-embedding-3-small --output pred_embeddings.npz
Class-name embeddings are generated with --labels-csv (see evaluation/README.md).
The cache is resumable: only texts that are not yet cached are sent to the API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from metrics import EMBEDDING_MAP_TYPES, load_labels, prediction, question_type


def save_embedding_cache(path: str | Path, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("vectors must have shape (len(texts), embedding_dim)")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray("ssqa_eval_embedding_cache_v1"),
        model=np.asarray(model),
        texts=np.asarray(list(texts), dtype=np.str_),
        vectors=vectors,
    )


def load_embedding_cache(path: str | Path, expected_model: str | None = None) -> tuple[str, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        version = str(data["format_version"].item())
        model = str(data["model"].item())
        texts = data["texts"].astype(str).tolist()
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if version != "ssqa_eval_embedding_cache_v1":
        raise ValueError(f"unsupported embedding cache format: {version}")
    if expected_model is not None and model != expected_model:
        raise ValueError(f"embedding model mismatch: cache={model!r}, expected={expected_model!r}")
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("corrupt embedding cache dimensions")
    return model, texts, vectors


def prediction_texts(path: Path, subset: str | None) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if subset is not None and record.get("subset") != subset:
                continue
            if question_type(record) in EMBEDDING_MAP_TYPES:
                texts.append(prediction(record))
    return list(dict.fromkeys(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels-csv", type=Path)
    source.add_argument("--predictions", type=Path)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    texts = load_labels(args.labels_csv) if args.labels_csv else prediction_texts(args.predictions, args.subset)
    if not texts:
        raise SystemExit("no texts selected for embedding")

    cached = {}
    if args.output.exists():
        _, old_texts, old_vectors = load_embedding_cache(args.output, args.model)
        cached = dict(zip(old_texts, old_vectors))
    missing = [text for text in texts if text not in cached]
    print(f"texts={len(texts)} cached={len(texts)-len(missing)} missing={len(missing)}")

    if missing:
        from openai import OpenAI

        client = OpenAI(
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start : start + args.batch_size]
            response = client.embeddings.create(model=args.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            for text, item in zip(batch, ordered):
                cached[text] = np.asarray(item.embedding, dtype=np.float32)
            current_texts = [text for text in texts if text in cached]
            save_embedding_cache(
                args.output,
                args.model,
                current_texts,
                np.stack([cached[text] for text in current_texts]),
            )
            print(f"embedded {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)

    vectors = np.stack([cached[text] for text in texts])
    save_embedding_cache(args.output, args.model, texts, vectors)
    print(f"cache -> {args.output} ({vectors.shape[0]} x {vectors.shape[1]}, model={args.model})")


if __name__ == "__main__":
    main()
