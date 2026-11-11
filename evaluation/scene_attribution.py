#!/usr/bin/env python3
"""Evaluate source-wise semantic/spatial attribution in dual_scene_v1 outputs.

The metric is deliberately independent from the final-answer SSQA metrics.
It parses the two generated scene sources, aligns them to the two manifest
sources by label overlap (order invariant), and measures whether identity and
spatial attributes remain bound to the same source.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from metrics import extract_final_answer, normalize_text, parse_doa, question_type


AXES = ("lr", "fb", "ud")
SOURCE_RE = re.compile(
    r"^\s*-\s*Source\s+at\s+"
    r"(?P<lr>left|right)\s*,\s*"
    r"(?P<fb>front|behind|back)\s*,\s*"
    r"(?:and\s+)?(?P<ud>above|below)\s*,\s*"
    r"approximately\s+(?P<distance>\d+(?:\.\d+)?)\s*"
    r"(?:meters?|m)\s+away\s*:\s*(?P<labels>.+?)\s*\.?\s*$",
    re.IGNORECASE,
)


def label_set(value: object) -> set[str]:
    return {
        normalize_text(part).strip(" .")
        for part in str(value or "").split(";")
        if normalize_text(part).strip(" .")
    }


def label_scores(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def parse_scene(text: object) -> list[dict]:
    sources = []
    for line in str(text or "").splitlines():
        match = SOURCE_RE.match(line)
        if not match:
            continue
        fb = match.group("fb").lower()
        sources.append({
            "labels": label_set(match.group("labels")),
            "location": {
                "lr": match.group("lr").lower(),
                "fb": "behind" if fb == "back" else fb,
                "ud": match.group("ud").lower(),
                "distance_m": float(match.group("distance")),
            },
        })
    return sources


def normalize_gt_source(source: Mapping[str, object]) -> dict:
    location = source["location"]
    return {
        "labels": {normalize_text(label) for label in source["labels"]},
        "location": {
            "lr": str(location["lr"]),
            "fb": str(location["fb"]),
            "ud": str(location["ud"]),
            "distance_m": float(location["distance_m"]),
        },
        "is_target": bool(source.get("is_target")),
    }


def source_components(predicted: Mapping[str, object], expected: Mapping[str, object]) -> dict:
    precision, recall, f1 = label_scores(predicted["labels"], expected["labels"])
    pred_loc = predicted["location"]
    gt_loc = expected["location"]
    axis = {name: pred_loc[name] == gt_loc[name] for name in AXES}
    distance_error = abs(float(pred_loc["distance_m"]) - float(gt_loc["distance_m"]))
    return {
        "label_precision": precision,
        "label_recall": recall,
        "label_f1": f1,
        "label_exact": predicted["labels"] == expected["labels"],
        "axis": axis,
        "direction_exact": all(axis.values()),
        "distance_error": distance_error,
        "distance_within_0_5m": distance_error <= 0.5,
        "tuple_exact": (
            predicted["labels"] == expected["labels"]
            and all(axis.values())
            and distance_error <= 0.5
        ),
    }


def semantic_alignment(predicted: Sequence[dict], expected: Sequence[dict]) -> tuple[int, ...]:
    """Map each GT source to a prediction, prioritizing semantic identity."""
    candidates = []
    for permutation in itertools.permutations(range(len(predicted)), len(expected)):
        parts = [source_components(predicted[p], expected[g]) for g, p in enumerate(permutation)]
        score = (
            sum(item["label_f1"] for item in parts),
            sum(item["label_exact"] for item in parts),
            sum(item["direction_exact"] for item in parts),
            sum(item["distance_within_0_5m"] for item in parts),
            -sum(min(item["distance_error"], 30.0) for item in parts),
        )
        candidates.append((score, permutation))
    return max(candidates)[1]


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def source_summary(items: Sequence[dict]) -> dict:
    if not items:
        return {"n": 0}
    return {
        "n": len(items),
        "label_precision": mean_or_none([item["label_precision"] for item in items]),
        "label_recall": mean_or_none([item["label_recall"] for item in items]),
        "label_f1": mean_or_none([item["label_f1"] for item in items]),
        "label_exact_accuracy": mean_or_none([float(item["label_exact"]) for item in items]),
        "axis_lr_accuracy": mean_or_none([float(item["axis"]["lr"]) for item in items]),
        "axis_fb_accuracy": mean_or_none([float(item["axis"]["fb"]) for item in items]),
        "axis_ud_accuracy": mean_or_none([float(item["axis"]["ud"]) for item in items]),
        "direction_exact_accuracy": mean_or_none(
            [float(item["direction_exact"]) for item in items]
        ),
        "distance_der": mean_or_none(
            [float(not item["distance_within_0_5m"]) for item in items]
        ),
        "distance_mae": mean_or_none([item["distance_error"] for item in items]),
        "tuple_exact_accuracy": mean_or_none([float(item["tuple_exact"]) for item in items]),
    }


def final_answer_doa_correct(record: Mapping[str, object], target: Mapping[str, object]) -> bool:
    answer = parse_doa(extract_final_answer(record.get("raw", "")))
    if answer is None:
        return False
    location = target["location"]
    return (
        all(answer[axis] == location[axis] for axis in AXES)
        and answer["dist"] is not None
        and abs(float(answer["dist"]) - float(location["distance_m"])) <= 0.5
    )


def final_answer_classification_scores(
    record: Mapping[str, object], target: Mapping[str, object]
) -> tuple[float, float, float]:
    return label_scores(label_set(extract_final_answer(record.get("raw", ""))), target["labels"])


def evaluate(manifest_rows: Sequence[dict], prediction_rows: Sequence[dict]) -> dict:
    manifest = {row["key"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate keys in manifest")
    predictions = {row["key"]: row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("duplicate keys in predictions")
    if set(predictions) != set(manifest):
        missing = sorted(set(manifest) - set(predictions))
        extra = sorted(set(predictions) - set(manifest))
        raise ValueError(f"manifest/prediction key mismatch: missing={missing[:3]} extra={extra[:3]}")

    all_sources = []
    target_sources = []
    distractor_sources = []
    union_precision = []
    union_recall = []
    union_f1 = []
    scene_direction_exact = 0
    scene_distance_within = 0
    scene_tuple_exact = 0
    parsed_two = 0
    counts_by_type = defaultdict(int)
    doa_final_correct = 0
    doa_scene_target_tuple_present = 0
    doa_scene_correct_final_wrong = 0
    doa_n = 0
    cls_final_p = []
    cls_final_r = []
    cls_final_f1 = []
    cls_scene_target_tuple_present = 0
    cls_scene_correct_final_wrong = 0
    cls_n = 0
    by_type_sources = defaultdict(lambda: {"all": [], "target": [], "distractor": []})
    by_type_scene = defaultdict(lambda: {
        "n": 0, "parsed": 0, "direction": 0, "distance": 0, "tuple": 0,
        "union_precision": [], "union_recall": [], "union_f1": [],
    })

    for key in sorted(manifest):
        gt_row = manifest[key]
        pred_row = predictions[key]
        expected = [normalize_gt_source(source) for source in gt_row.get("scene_sources", [])]
        if len(expected) != 2:
            raise ValueError(f"{key}: expected exactly two GT scene sources")
        target_indices = [index for index, source in enumerate(expected) if source["is_target"]]
        if len(target_indices) != 1:
            raise ValueError(f"{key}: expected exactly one target source")
        target_index = target_indices[0]
        predicted = parse_scene(pred_row.get("raw", ""))
        kind = question_type(pred_row)
        counts_by_type[kind] += 1
        by_type_scene[kind]["n"] += 1

        if len(predicted) != 2:
            zero = {
                "label_precision": 0.0, "label_recall": 0.0, "label_f1": 0.0,
                "label_exact": False, "axis": {axis: False for axis in AXES},
                "direction_exact": False, "distance_error": 30.0,
                "distance_within_0_5m": False, "tuple_exact": False,
            }
            parts = [dict(zero), dict(zero)]
            pred_union = set()
        else:
            parsed_two += 1
            by_type_scene[kind]["parsed"] += 1
            assignment = semantic_alignment(predicted, expected)
            parts = [
                source_components(predicted[pred_index], expected[gt_index])
                for gt_index, pred_index in enumerate(assignment)
            ]
            pred_union = set().union(*(source["labels"] for source in predicted))

        gt_union = set().union(*(source["labels"] for source in expected))
        p, r, f1 = label_scores(pred_union, gt_union)
        union_precision.append(p)
        union_recall.append(r)
        union_f1.append(f1)
        by_type_scene[kind]["union_precision"].append(p)
        by_type_scene[kind]["union_recall"].append(r)
        by_type_scene[kind]["union_f1"].append(f1)
        all_sources.extend(parts)
        target_sources.append(parts[target_index])
        distractor_sources.append(parts[1 - target_index])
        by_type_sources[kind]["all"].extend(parts)
        by_type_sources[kind]["target"].append(parts[target_index])
        by_type_sources[kind]["distractor"].append(parts[1 - target_index])
        scene_direction_exact += int(all(part["direction_exact"] for part in parts))
        scene_distance_within += int(all(part["distance_within_0_5m"] for part in parts))
        scene_tuple_exact += int(all(part["tuple_exact"] for part in parts))
        by_type_scene[kind]["direction"] += int(all(part["direction_exact"] for part in parts))
        by_type_scene[kind]["distance"] += int(
            all(part["distance_within_0_5m"] for part in parts)
        )
        by_type_scene[kind]["tuple"] += int(all(part["tuple_exact"] for part in parts))

        target_tuple_present = parts[target_index]["tuple_exact"]
        if kind == "MIXUP_SINGLE_DOA":
            doa_n += 1
            final_correct = final_answer_doa_correct(pred_row, expected[target_index])
            doa_final_correct += int(final_correct)
            doa_scene_target_tuple_present += int(target_tuple_present)
            doa_scene_correct_final_wrong += int(target_tuple_present and not final_correct)
        elif kind == "MIXUP_SINGLE_CLASSIFICATION":
            cls_n += 1
            fp, fr, ff1 = final_answer_classification_scores(pred_row, expected[target_index])
            cls_final_p.append(fp)
            cls_final_r.append(fr)
            cls_final_f1.append(ff1)
            final_exact = label_set(extract_final_answer(pred_row.get("raw", ""))) == expected[target_index]["labels"]
            cls_scene_target_tuple_present += int(target_tuple_present)
            cls_scene_correct_final_wrong += int(target_tuple_present and not final_exact)

    n = len(manifest_rows)
    by_question_type = {}
    for kind in sorted(by_type_scene):
        scene = by_type_scene[kind]
        count = scene["n"]
        sources = by_type_sources[kind]
        by_question_type[kind] = {
            "n": count,
            "two_source_parse_rate": scene["parsed"] / count if count else None,
            "label_union_precision": mean_or_none(scene["union_precision"]),
            "label_union_recall": mean_or_none(scene["union_recall"]),
            "label_union_f1": mean_or_none(scene["union_f1"]),
            "both_sources_direction_exact_rate": scene["direction"] / count if count else None,
            "both_sources_distance_within_0_5m_rate": scene["distance"] / count if count else None,
            "full_scene_tuple_exact_rate": scene["tuple"] / count if count else None,
            "identity_aligned_sources": source_summary(sources["all"]),
            "target_source": source_summary(sources["target"]),
            "distractor_source": source_summary(sources["distractor"]),
        }

    return {
        "protocol": "dual_scene_v1 source-wise semantic-spatial attribution v1",
        "record_count": n,
        "question_type_counts": dict(sorted(counts_by_type.items())),
        "format": {
            "two_source_parse_rate": parsed_two / n if n else None,
            "two_source_parsed": parsed_two,
        },
        "scene": {
            "label_union_precision": mean_or_none(union_precision),
            "label_union_recall": mean_or_none(union_recall),
            "label_union_f1": mean_or_none(union_f1),
            "both_sources_direction_exact_rate": scene_direction_exact / n if n else None,
            "both_sources_distance_within_0_5m_rate": scene_distance_within / n if n else None,
            "full_scene_tuple_exact_rate": scene_tuple_exact / n if n else None,
        },
        "identity_aligned_sources": source_summary(all_sources),
        "target_source": source_summary(target_sources),
        "distractor_source": source_summary(distractor_sources),
        "by_question_type": by_question_type,
        "query": {
            "MIXUP_SINGLE_DOA": {
                "n": doa_n,
                "final_answer_full_accuracy": doa_final_correct / doa_n if doa_n else None,
                "target_tuple_present_rate": (
                    doa_scene_target_tuple_present / doa_n if doa_n else None
                ),
                "target_tuple_present_but_final_wrong_rate": (
                    doa_scene_correct_final_wrong / doa_n if doa_n else None
                ),
            },
            "MIXUP_SINGLE_CLASSIFICATION": {
                "n": cls_n,
                "final_answer_label_precision": mean_or_none(cls_final_p),
                "final_answer_label_recall": mean_or_none(cls_final_r),
                "final_answer_label_f1": mean_or_none(cls_final_f1),
                "target_tuple_present_rate": (
                    cls_scene_target_tuple_present / cls_n if cls_n else None
                ),
                "target_tuple_present_but_final_not_exact_rate": (
                    cls_scene_correct_final_wrong / cls_n if cls_n else None
                ),
            },
        },
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--subset", default="held-out")
    parser.add_argument("--expect-records", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    prediction_rows = read_jsonl(args.predictions)
    if args.subset is not None:
        prediction_rows = [row for row in prediction_rows if row.get("subset") == args.subset]
    if args.expect_records:
        if len(manifest_rows) != args.expect_records or len(prediction_rows) != args.expect_records:
            raise SystemExit(
                f"record guard failed: expected={args.expect_records} "
                f"manifest={len(manifest_rows)} predictions={len(prediction_rows)}"
            )
    summary = evaluate(manifest_rows, prediction_rows)
    summary["manifest"] = str(args.manifest)
    summary["predictions"] = str(args.predictions)
    summary["subset_filter"] = args.subset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsummary -> {args.output}")


if __name__ == "__main__":
    main()
