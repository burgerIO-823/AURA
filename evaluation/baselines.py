#!/usr/bin/env python3
"""Same-set audio-free statistical baselines for any Aura eval manifest.

Every task in these sets admits a trivial strategy that never listens to the
audio — the Stage-3 binary sets are near 50/50 by construction, and a constant
2.0 m answer already scores DER .4459 on inter-source distance — so a
generative number is not interpretable on its own.  This computes, from the
ground truth alone, what a constant predictor scores under the SAME parsers the
formal scorer uses (evaluation/metrics.py).  No model, no GPU, no
predictions.

Reported per task type:
  DOA / MIXUP_SINGLE_DOA     per-axis majority accuracy, joint all-3 accuracy of
                             the majority direction, and distance DER/MAE for the
                             best constant and the ground-truth mean
  CLASSIFICATION types       sample P/R/F1 of always answering the single most
                             frequent label set, plus its exact-match rate (mAP
                             has no closed-form constant baseline — it needs the
                             embedding path, so only the prior is reported)
  binary sets                majority-class accuracy and the 0.5 chance line
  nonbinary direction        majority-class accuracy over the six axis words
  nonbinary distance         DER of the best constant answer, and of the mean and
                             median of the ground truth
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


from metrics import (  # noqa: E402
    AXES,
    CLASSIFICATION_TYPES,
    DOA_TYPES,
    NONBINARY_SOURCE_TYPES,
    normalize_text,
    parse_binary,
    parse_distance,
    parse_doa,
    parse_relative_direction,
)

BINARY_TYPES = ("MIXUP_DIRECTION", "MIXUP_DISTANCE_BOTH")
DER_TOLERANCE_M = 0.5


def binary_baseline(answers: list[str]) -> dict:
    parsed = [parse_binary(answer) for answer in answers]
    if any(value is None for value in parsed):
        raise SystemExit("unparseable binary ground truth in manifest")
    counts = Counter(parsed)
    majority_value, majority_count = counts.most_common(1)[0]
    return {
        "n": len(parsed),
        "class_counts": {"yes": counts.get(True, 0), "no": counts.get(False, 0)},
        "majority_class": "yes" if majority_value else "no",
        "majority_class_accuracy": majority_count / len(parsed),
        "chance_accuracy": 0.5,
    }


def relative_direction_baseline(answers: list[str]) -> dict:
    parsed = [parse_relative_direction(answer) for answer in answers]
    if any(value is None for value in parsed):
        raise SystemExit("unparseable relative-direction ground truth in manifest")
    counts = Counter(parsed)
    majority_value, majority_count = counts.most_common(1)[0]
    return {
        "n": len(parsed),
        "class_counts": dict(sorted(counts.items())),
        "majority_class": majority_value,
        "majority_class_accuracy": majority_count / len(parsed),
        "uniform_chance_accuracy": 1.0 / len(counts),
    }


def der(values: list[float], constant: float) -> float:
    return sum(abs(value - constant) > DER_TOLERANCE_M for value in values) / len(values)


def distance_baseline(answers: list[str]) -> dict:
    parsed = [parse_distance(answer) for answer in answers]
    if any(value is None for value in parsed):
        raise SystemExit("unparseable distance ground truth in manifest")
    ordered = sorted(parsed)
    mean = sum(parsed) / len(parsed)
    median = ordered[len(ordered) // 2]
    # The DER-optimal constant is always one of the observed values: DER only
    # changes when the +-0.5 m window crosses a ground-truth point.
    best = min(set(parsed), key=lambda candidate: (der(parsed, candidate), candidate))
    return {
        "n": len(parsed),
        "gt_mean_m": mean,
        "gt_median_m": median,
        "best_constant_m": best,
        "best_constant_der": der(parsed, best),
        "mean_constant_der": der(parsed, mean),
        "median_constant_der": der(parsed, median),
        "mean_constant_mae_m": sum(abs(value - mean) for value in parsed) / len(parsed),
        "der_tolerance_m": DER_TOLERANCE_M,
    }


def doa_baseline(answers: list[str]) -> dict:
    """Constant-direction and constant-distance baselines, metrics.py caliber."""
    parsed = [parse_doa(answer) for answer in answers]
    if any(value is None for value in parsed):
        raise SystemExit("unparseable DOA ground truth in manifest")
    result: dict[str, object] = {"n": len(parsed)}

    majority = {}
    for axis in AXES:
        values = [row[axis] for row in parsed if row[axis] is not None]
        if not values:
            continue
        counts = Counter(values)
        majority[axis], count = counts.most_common(1)[0]
        result[f"axis_{axis}_majority"] = majority[axis]
        result[f"axis_{axis}_majority_accuracy"] = count / len(values)
    complete = [row for row in parsed if all(row[axis] is not None for axis in AXES)]
    if complete and len(majority) == len(AXES):
        hits = sum(all(row[axis] == majority[axis] for axis in AXES) for row in complete)
        result["n_direction"] = len(complete)
        result["majority_direction"] = dict(majority)
        result["doa_accuracy_majority_constant"] = hits / len(complete)

    distances = [row["dist"] for row in parsed if row["dist"] is not None]
    if distances:
        mean = sum(distances) / len(distances)
        best = min(set(distances), key=lambda c: (der(distances, c), c))
        result.update({
            "n_distance": len(distances),
            "gt_mean_m": mean,
            "best_constant_m": best,
            "best_constant_der": der(distances, best),
            "mean_constant_der": der(distances, mean),
            "mean_constant_mae_m": sum(abs(v - mean) for v in distances) / len(distances),
            "der_tolerance_m": DER_TOLERANCE_M,
        })
    return result


def classification_baseline(answers: list[str]) -> dict:
    """Sample P/R/F1 of always answering the single most frequent label set."""
    label_sets = [
        {part.strip().lower() for part in answer.split(";") if part.strip()}
        for answer in answers
    ]
    counts = Counter(normalize_text(answer) for answer in answers)
    constant_text, constant_count = counts.most_common(1)[0]
    constant = {part.strip() for part in constant_text.split(";") if part.strip()}
    precision = recall = f1 = 0.0
    for truth in label_sets:
        overlap = len(truth & constant)
        p = overlap / len(constant) if constant else 0.0
        r = overlap / len(truth) if truth else 0.0
        precision += p
        recall += r
        f1 += 2 * p * r / (p + r) if (p + r) else 0.0
    n = len(label_sets)
    return {
        "n": n,
        "distinct_answers": len(counts),
        "most_frequent_answer": constant_text,
        "most_frequent_exact_match_rate": constant_count / n,
        "constant_answer_sample_precision": precision / n,
        "constant_answer_sample_recall": recall / n,
        "constant_answer_sample_f1": f1 / n,
        "map_baseline": "not_applicable_without_embeddings",
    }


def manifest_question_type(row: dict) -> str:
    """Group rows under the same name the formal scorer will report.

    metrics.py resolves a record's type from raw_question_type when it
    knows it, and otherwise falls back to the qt field.  Mirroring that here
    keeps baseline keys joinable with question_type_counts in the summary:
    stage3 -> MIXUP_*, stage2 -> MIXUP_SINGLE_*, stage1 -> DOA/CLASSIFICATION.
    """
    raw = str(row.get("raw_question_type") or "").strip()
    known = (
        set(BINARY_TYPES)
        | {"MIXUP_NONBINARY_DIRECTION", "MIXUP_NONBINARY_DISTANCE"}
        | set(DOA_TYPES)
        | set(CLASSIFICATION_TYPES)
        | set(NONBINARY_SOURCE_TYPES)
    )
    if raw in known:
        return raw
    return str(row.get("question_type") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_type: dict[str, list[str]] = {}
    with open(args.manifest, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            by_type.setdefault(manifest_question_type(row), []).append(str(row["answer"]))

    baselines: dict[str, object] = {}
    for kind, answers in sorted(by_type.items()):
        if kind in BINARY_TYPES:
            baselines[kind] = binary_baseline(answers)
        elif kind == "MIXUP_NONBINARY_DIRECTION":
            baselines[kind] = relative_direction_baseline(answers)
        elif kind == "MIXUP_NONBINARY_DISTANCE":
            baselines[kind] = distance_baseline(answers)
        elif kind in DOA_TYPES:
            baselines[kind] = doa_baseline(answers)
        elif kind in CLASSIFICATION_TYPES or kind in NONBINARY_SOURCE_TYPES:
            baselines[kind] = classification_baseline(answers)
        else:
            baselines[kind] = {"n": len(answers), "status": "no_baseline_defined"}

    summary = {
        "protocol": "same-set audio-free statistical baselines v1",
        "manifest": str(args.manifest),
        "record_count": sum(len(values) for values in by_type.values()),
        "question_type_counts": {kind: len(values) for kind, values in sorted(by_type.items())},
        "baselines": baselines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nbaselines -> {args.output}")


if __name__ == "__main__":
    main()
