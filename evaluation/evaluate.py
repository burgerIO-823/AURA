#!/usr/bin/env python3
"""Score an AURA prediction JSONL with the BAT/SpatialSoundQA metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embeddings import load_embedding_cache
from metrics import (
    CLASSIFICATION_TYPES,
    EMBEDDING_MAP_TYPES,
    NONBINARY_SOURCE_TYPES,
    evaluate_classification_map,
    evaluate_records,
    ground_truth,
    load_labels,
    prediction,
    question_type,
)


def read_records(path: Path, subset: str | None) -> list[dict]:
    records = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
            if subset is not None and record.get("subset") != subset:
                continue
            identity = (record.get("subset"), record.get("key"))
            if identity in seen:
                raise SystemExit(f"duplicate prediction record: subset={identity[0]!r} key={identity[1]!r}")
            seen.add(identity)
            records.append(record)
    return records


def attach_map(summary: dict, records: list[dict], args: argparse.Namespace) -> None:
    """Attach embedding mAP, keeping the BAT metric and the extension apart.

    BAT's mAP is the type A|C sound-event-detection metric.  MIXUP_NONBINARY_SOURCE
    also has a label-set answer, so it uses the same embedding cache, but its mAP
    is reported under additional_ssqa_tasks — never in the official slot, where it
    would read as a detection number comparable to stage1/stage2.
    """
    cls_records = [record for record in records if question_type(record) in EMBEDDING_MAP_TYPES]
    counts_by_type = {}
    for record in cls_records:
        kind = question_type(record)
        counts_by_type[kind] = counts_by_type.get(kind, 0) + 1
    official_n = sum(count for kind, count in counts_by_type.items()
                     if kind in CLASSIFICATION_TYPES)
    if not cls_records:
        summary["ssqa_official"]["classification"] = {"n": 0, "status": "not_present"}
        return
    if not (args.labels_csv and args.class_embeddings and args.prediction_embeddings):
        summary["ssqa_official"]["classification"] = (
            {
                "n": official_n,
                "counts_by_question_type": {
                    kind: count for kind, count in sorted(counts_by_type.items())
                    if kind in CLASSIFICATION_TYPES
                },
                "status": "embedding_caches_required",
            }
            if official_n
            else {"n": 0, "status": "not_present"}
        )
        return

    labels = load_labels(args.labels_csv)
    class_model, class_texts, class_vectors = load_embedding_cache(
        args.class_embeddings, args.embedding_model
    )
    pred_model, pred_texts, pred_vectors = load_embedding_cache(
        args.prediction_embeddings, args.embedding_model
    )
    if class_model != pred_model:
        raise SystemExit(f"class/prediction embedding model mismatch: {class_model} != {pred_model}")
    if [text.strip() for text in class_texts] != [text.strip() for text in labels]:
        raise SystemExit("class embedding cache texts/order do not match labels CSV")

    by_text = {}
    for text, vector in zip(pred_texts, pred_vectors):
        if text in by_text:
            raise SystemExit(f"duplicate text in prediction embedding cache: {text!r}")
        by_text[text] = vector
    missing = sorted({prediction(record) for record in cls_records if prediction(record) not in by_text})
    if missing:
        raise SystemExit(
            f"prediction embedding cache is missing {len(missing)} unique texts; first={missing[:3]!r}"
        )
    by_type = {}
    for kind in sorted(EMBEDDING_MAP_TYPES):
        task_records = [record for record in cls_records if question_type(record) == kind]
        if not task_records:
            continue
        vectors = [by_text[prediction(record)] for record in task_records]
        by_type[kind] = evaluate_classification_map(
            task_records, labels, vectors, class_vectors
        )

    official = {kind: block for kind, block in by_type.items() if kind in CLASSIFICATION_TYPES}
    summary["ssqa_official"]["classification"] = (
        {
            "n": official_n,
            "status": "ok",
            "embedding_model": class_model,
            "by_question_type": official,
        }
        if official
        else {"n": 0, "status": "not_present"}
    )
    for kind in sorted(NONBINARY_SOURCE_TYPES):
        block = by_type.get(kind)
        if block is None:
            continue
        target = summary.setdefault("additional_ssqa_tasks", {}).setdefault(
            "nonbinary_source", {}
        )
        target["label_set_map"] = {
            **block,
            "embedding_model": class_model,
            "note": "same embedding pipeline as detection mAP, but a different "
                    "question over a different set — not comparable to the "
                    "type A|C detection mAP and not a BAT-reported metric",
        }


def parse_expected_counts(values: list[str]) -> dict[str, int]:
    expected = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--expect-count must be QUESTION_TYPE=N, got {value!r}")
        kind, count = value.rsplit("=", 1)
        try:
            expected[kind] = int(count)
        except ValueError as error:
            raise SystemExit(f"invalid expected count: {value!r}") from error
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSONL produced by generate.py or another generator")
    parser.add_argument("--subset", default=None,
                        help="only score this subset value, e.g. held-out")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--class-embeddings", type=Path,
                        help="model-tagged NPZ created by embeddings.py")
    parser.add_argument("--prediction-embeddings", type=Path,
                        help="model-tagged NPZ created by embeddings.py")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--expect-records", type=int, default=0,
                        help="fail unless exactly this many records are selected")
    parser.add_argument("--expect-count", action="append", default=[],
                        help="repeatable QUESTION_TYPE=N completeness guard")
    args = parser.parse_args()

    records = read_records(args.predictions, args.subset)
    if not records:
        raise SystemExit("no prediction records selected")
    summary = evaluate_records(records)
    if args.expect_records and len(records) != args.expect_records:
        raise SystemExit(f"record count guard failed: expected {args.expect_records}, found {len(records)}")
    expected_counts = parse_expected_counts(args.expect_count)
    for kind, expected in expected_counts.items():
        actual = summary["question_type_counts"].get(kind, 0)
        if actual != expected:
            raise SystemExit(f"question type count guard failed for {kind}: expected {expected}, found {actual}")
    summary["source"] = str(args.predictions)
    summary["subset_filter"] = args.subset
    attach_map(summary, records, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsummary -> {args.output}")


if __name__ == "__main__":
    main()
