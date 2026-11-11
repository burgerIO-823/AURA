"""BAT/SpatialSoundQA metrics.

The official metrics deliberately use every ground-truth example as part of
the denominator.  A malformed or missing prediction is therefore an error,
not a sample that silently disappears from evaluation.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


# BAT/SpatialSoundQA question types and the metric each one is scored with in
# the paper (Zheng et al., ICML 2024, Table 4):
#
#   A  single-source detection      | mAP    -> CLASSIFICATION_TYPES
#   B  single-source DoA + distance | Acc/DER-> DOA_TYPES
#   C  two-source detection         | mAP    -> CLASSIFICATION_TYPES
#   D  two-source DoA + distance    | Acc/DER-> DOA_TYPES
#   E  two-source spatial reasoning | BA     -> DIRECTION_BINARY/DISTANCE_BINARY
#
# Type E is the whole of stage 3, and the paper scores it with Binary Accuracy
# ONLY: "To make evaluation straightforward, we have chosen a binary (Yes/No)
# response format for these questions."  Random is 50.00 on both categories.
#
# The public SpatialSoundQA release also ships eval-stage3-nonbinary.json
# (MIXUP_NONBINARY_{DIRECTION,DISTANCE,SOURCE}), which the paper never reports.
# Those three are scored here as clearly-labelled EXTENSIONS with no published
# reference number.  In particular MIXUP_NONBINARY_SOURCE is deliberately NOT a
# CLASSIFICATION type: BAT's mAP is the type A|C sound-event-detection metric
# over the whole clip, and reusing it for "what is heard behind X" would produce
# a number that looks comparable to the stage1/2 detection mAP but is not.
DOA_TYPES = {"DOA", "MIXUP_SINGLE_DOA"}
CLASSIFICATION_TYPES = {
    "CLASSIFICATION",
    "MIXUP_SINGLE_CLASSIFICATION",
}
DIRECTION_BINARY_TYPES = {"MIXUP_DIRECTION"}
DISTANCE_BINARY_TYPES = {"MIXUP_DISTANCE_BOTH"}
NONBINARY_DIRECTION_TYPES = {"MIXUP_NONBINARY_DIRECTION"}
NONBINARY_DISTANCE_TYPES = {"MIXUP_NONBINARY_DISTANCE"}
NONBINARY_SOURCE_TYPES = {"MIXUP_NONBINARY_SOURCE"}
# Types whose answers are label sets and therefore need a text-embedding cache.
# Scoring keeps them apart; only the cache path treats them alike.
EMBEDDING_MAP_TYPES = CLASSIFICATION_TYPES | NONBINARY_SOURCE_TYPES
KNOWN_TYPES = (
    DOA_TYPES
    | CLASSIFICATION_TYPES
    | DIRECTION_BINARY_TYPES
    | DISTANCE_BINARY_TYPES
    | NONBINARY_DIRECTION_TYPES
    | NONBINARY_DISTANCE_TYPES
    | NONBINARY_SOURCE_TYPES
)
# BAT Table 4, Bat (B+P) row and the Random row, for reporting alongside our
# numbers.  Different backbone, front-end and training data — an external
# reference point, never a controlled comparison.
BAT_TYPE_E_REFERENCE = {
    "source": "BAT (Zheng et al., ICML 2024), Table 4, Bat B+P row",
    "binary_accuracy": {"direction": 0.6977, "distance": 0.8401, "average": 0.7689},
    "random_baseline": {"direction": 0.5000, "distance": 0.5000, "average": 0.5000},
}

AXES = ("lr", "fb", "ud")


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").strip().lower().split())


def extract_final_answer(text: object) -> str:
    """Extract the last Final Answer/So segment when a raw generation is used."""
    value = str(text or "")
    matches = list(re.finditer(r"final\s+answer:\s*", value, re.IGNORECASE))
    if matches:
        return value[matches[-1].end() :].strip()
    matches = list(re.finditer(r"\bso:\s*", value, re.IGNORECASE))
    if matches:
        return value[matches[-1].end() :].strip()
    return value.strip()


def question_type(record: Mapping[str, object]) -> str:
    raw = str(record.get("raw_question_type") or "").strip()
    if raw in KNOWN_TYPES:
        return raw
    key_match = re.match(r"(.+)-(\d+)$", str(record.get("key") or ""))
    if key_match and key_match.group(1) in KNOWN_TYPES:
        return key_match.group(1)
    return str(record.get("qt") or record.get("question_type") or "").strip()


def ground_truth(record: Mapping[str, object]) -> str:
    return str(record.get("gt") if "gt" in record else record.get("answer", ""))


def prediction(record: Mapping[str, object]) -> str:
    if "pred" in record:
        return extract_final_answer(record.get("pred"))
    return extract_final_answer(record.get("raw", ""))


def parse_doa(text: object) -> dict[str, object] | None:
    value = normalize_text(text)
    lr = "left" if re.search(r"\bleft\b", value) else (
        "right" if re.search(r"\bright\b", value) else None
    )
    fb = "behind" if re.search(r"\b(?:behind|back)\b", value) else (
        "front" if re.search(r"\b(?:front|ahead)\b", value) else None
    )
    ud = "above" if re.search(r"\b(?:above|up)\b", value) else (
        "below" if re.search(r"\b(?:below|down)\b", value) else None
    )
    distances = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:m|meters?)\b", value)
    dist = float(distances[-1]) if distances else None
    if dist is not None and (not math.isfinite(dist) or dist > 30):
        dist = None
    if lr is None and fb is None and ud is None and dist is None:
        return None
    return {"lr": lr, "fb": fb, "ud": ud, "dist": dist}


def parse_binary(text: object) -> bool | None:
    """Parse a standalone Yes/No answer without accepting incidental prose."""
    value = normalize_text(extract_final_answer(text)).strip(" .,!?:;\"'")
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def evaluate_doa(records: Sequence[Mapping[str, object]]) -> tuple[dict, dict]:
    """Return official DoA/DER metrics and separate component diagnostics."""
    n_direction = n_distance = 0
    direction_correct = direction_valid = 0
    distance_errors = 0
    distance_valid = 0
    distance_abs_errors: list[float] = []
    axis_correct = Counter()
    axis_total = Counter()

    for record in records:
        gt = parse_doa(ground_truth(record))
        pred = parse_doa(prediction(record))
        if gt is None:
            raise ValueError(f"unparseable DOA ground truth for key={record.get('key')!r}")

        if all(gt[axis] is not None for axis in AXES):
            n_direction += 1
            pred_complete = pred is not None and all(pred[axis] is not None for axis in AXES)
            if pred_complete:
                direction_valid += 1
                direction_correct += int(all(pred[axis] == gt[axis] for axis in AXES))
            for axis in AXES:
                axis_total[axis] += 1
                axis_correct[axis] += int(
                    pred is not None and pred[axis] is not None and pred[axis] == gt[axis]
                )

        if gt["dist"] is not None:
            n_distance += 1
            if pred is None or pred["dist"] is None:
                distance_errors += 1
            else:
                distance_valid += 1
                error = abs(float(pred["dist"]) - float(gt["dist"]))
                distance_abs_errors.append(error)
                distance_errors += int(error > 0.5)

    official = {
        "n_direction": n_direction,
        "doa_accuracy": direction_correct / n_direction if n_direction else None,
        "direction_valid_rate": direction_valid / n_direction if n_direction else None,
        "n_distance": n_distance,
        "distance_der": distance_errors / n_distance if n_distance else None,
        "distance_valid_rate": distance_valid / n_distance if n_distance else None,
    }
    diagnostics = {
        **{
            f"axis_{axis}_accuracy": axis_correct[axis] / axis_total[axis]
            if axis_total[axis]
            else None
            for axis in AXES
        },
        "distance_mae_on_valid_predictions": float(np.mean(distance_abs_errors))
        if distance_abs_errors
        else None,
        "distance_mae_valid_n": len(distance_abs_errors),
    }
    return official, diagnostics


def evaluate_binary(records: Sequence[Mapping[str, object]]) -> dict:
    correct = valid = 0
    for record in records:
        gt = parse_binary(ground_truth(record))
        if gt is None:
            raise ValueError(f"unparseable binary ground truth for key={record.get('key')!r}")
        pred = parse_binary(prediction(record))
        valid += int(pred is not None)
        correct += int(pred is not None and pred == gt)
    n = len(records)
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "valid_rate": valid / n if n else None,
        "correct": correct,
    }


def parse_relative_direction(text: object) -> str | None:
    value = normalize_text(extract_final_answer(text))
    found = [
        word
        for word in ("left", "right", "front", "behind", "above", "below")
        if re.search(rf"\b{word}\b", value)
    ]
    return found[0] if len(found) == 1 else None


def evaluate_nonbinary_direction(records: Sequence[Mapping[str, object]]) -> dict:
    correct = valid = 0
    for record in records:
        gt = parse_relative_direction(ground_truth(record))
        if gt is None:
            raise ValueError(f"unparseable direction ground truth for key={record.get('key')!r}")
        pred = parse_relative_direction(prediction(record))
        valid += int(pred is not None)
        correct += int(pred == gt)
    n = len(records)
    return {"n": n, "accuracy": correct / n if n else None,
            "valid_rate": valid / n if n else None}


def parse_distance(text: object) -> float | None:
    values = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:m|meters?)?\b", normalize_text(text))
    if not values:
        return None
    value = float(values[-1])
    return value if math.isfinite(value) and value <= 30 else None


def evaluate_nonbinary_distance(records: Sequence[Mapping[str, object]]) -> dict:
    errors = valid = 0
    for record in records:
        gt = parse_distance(ground_truth(record))
        if gt is None:
            raise ValueError(f"unparseable distance ground truth for key={record.get('key')!r}")
        pred = parse_distance(prediction(record))
        valid += int(pred is not None)
        errors += int(pred is None or abs(pred - gt) > 0.5)
    n = len(records)
    return {"n": n, "der": errors / n if n else None,
            "valid_rate": valid / n if n else None}


def parse_label_set(text: object) -> set[str]:
    """'speech; walk, footsteps' -> {'speech', 'walk, footsteps'}."""
    return {
        part
        for part in (normalize_text(chunk) for chunk in str(text or "").split(";"))
        if part
    }


def evaluate_nonbinary_source(records: Sequence[Mapping[str, object]]) -> dict:
    """Sample-level P/R/F1 and exact set match for 'what is heard at X'.

    NOT a BAT metric: the paper does not evaluate this split.  Scored as a label
    set rather than with detection mAP so it can never be mistaken for the type
    A|C detection number.
    """
    precision = recall = f1 = exact = valid = 0.0
    for record in records:
        truth = parse_label_set(ground_truth(record))
        if not truth:
            raise ValueError(f"empty label-set ground truth for key={record.get('key')!r}")
        predicted = parse_label_set(prediction(record))
        valid += int(bool(predicted))
        overlap = len(truth & predicted)
        p = overlap / len(predicted) if predicted else 0.0
        r = overlap / len(truth)
        precision += p
        recall += r
        f1 += 2 * p * r / (p + r) if (p + r) else 0.0
        exact += int(predicted == truth)
    n = len(records)
    return {
        "n": n,
        "sample_precision": precision / n if n else None,
        "sample_recall": recall / n if n else None,
        "sample_f1": f1 / n if n else None,
        "exact_set_match": exact / n if n else None,
        "valid_rate": valid / n if n else None,
        "bat_reference": "none — not evaluated in the BAT paper",
    }


def load_labels(labels_csv: str | Path) -> list[str]:
    # The distributed 355-class subset file is not valid RFC CSV: display
    # names containing commas are not quoted (for example
    # ``73,/m/068hy,Domestic animals, pets``).  BAT's original evaluator uses
    # split(',', 2), which preserves the complete display name; csv.DictReader
    # would silently truncate it to ``Domestic animals``.
    with open(labels_csv, encoding="utf-8") as stream:
        lines = [line.rstrip("\r\n") for line in stream if line.strip()]
    if not lines or lines[0].split(",", 2)[-1].strip().lower() != "display_name":
        raise ValueError(f"unexpected AudioSet label header in {labels_csv}")
    labels = []
    for line_number, line in enumerate(lines[1:], 2):
        parts = line.split(",", 2)
        if len(parts) != 3 or not parts[2].strip():
            raise ValueError(f"malformed AudioSet label row {labels_csv}:{line_number}")
        labels.append(parts[2].strip().strip('"'))
    if len({normalize_text(label) for label in labels}) != len(labels):
        raise ValueError("AudioSet label names are not unique after normalization")
    return labels


def cosine_scores(vectors: np.ndarray, class_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    class_vectors = np.asarray(class_vectors, dtype=np.float64)
    if vectors.ndim != 2 or class_vectors.ndim != 2:
        raise ValueError("embedding arrays must both be rank 2")
    if vectors.shape[1] != class_vectors.shape[1]:
        raise ValueError("prediction and class embedding dimensions differ")
    v_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    c_norm = np.linalg.norm(class_vectors, axis=1, keepdims=True).T
    denom = v_norm * c_norm
    if np.any(denom == 0):
        raise ValueError("zero-norm embedding encountered")
    return vectors @ class_vectors.T / denom


def evaluate_classification_map(
    records: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    prediction_vectors: np.ndarray,
    class_vectors: np.ndarray,
) -> dict:
    """Compute BAT generated-text mAP using class cosine-similarity scores."""
    if len(records) != len(prediction_vectors):
        raise ValueError("one prediction embedding is required per classification record")
    if len(labels) != len(class_vectors):
        raise ValueError("one class embedding is required per AudioSet label")

    label_to_id = {normalize_text(label): index for index, label in enumerate(labels)}
    targets = np.zeros((len(records), len(labels)), dtype=np.int8)
    unknown = Counter()
    for row_index, record in enumerate(records):
        for label in str(ground_truth(record)).split(";"):
            normalized = normalize_text(label)
            if not normalized:
                continue
            class_index = label_to_id.get(normalized)
            if class_index is None:
                unknown[normalized] += 1
            else:
                targets[row_index, class_index] = 1

    scores = cosine_scores(prediction_vectors, class_vectors)
    valid_classes = np.flatnonzero(targets.sum(axis=0) > 0)
    per_class = {
        labels[index]: float(average_precision_score(targets[:, index], scores[:, index]))
        for index in valid_classes
    }
    return {
        "n": len(records),
        "map": float(np.mean(list(per_class.values()))) if per_class else None,
        "num_classes_evaluated": len(per_class),
        "num_labels": len(labels),
        "unknown_gt_label_occurrences": int(sum(unknown.values())),
        "unknown_gt_labels": dict(unknown.most_common()),
        "per_class_ap": per_class,
    }


def angular_error_degrees(
    true_azimuth_deg: float,
    true_elevation_deg: float,
    pred_azimuth_deg: float,
    pred_elevation_deg: float,
) -> float:
    def unit(azimuth: float, elevation: float) -> np.ndarray:
        azimuth = math.radians(azimuth)
        elevation = math.radians(elevation)
        return np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ])

    dot = float(np.dot(unit(true_azimuth_deg, true_elevation_deg),
                       unit(pred_azimuth_deg, pred_elevation_deg)))
    return math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))


def evaluate_continuous_doa(errors_deg: Iterable[float]) -> dict:
    errors = np.asarray(list(errors_deg), dtype=np.float64)
    if errors.size == 0:
        return {"n": 0, "er20": None, "angular_mae_deg": None}
    if np.any(~np.isfinite(errors)) or np.any((errors < 0) | (errors > 180)):
        raise ValueError("angular errors must be finite and in [0, 180]")
    return {
        "n": int(errors.size),
        "er20": float(np.mean(errors > 20.0)),
        "angular_mae_deg": float(np.mean(errors)),
    }


def evaluate_records(records: Sequence[Mapping[str, object]]) -> dict:
    """Evaluate all non-embedding SSQA metrics present in a record collection."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[question_type(record)].append(record)

    def collect(types: set[str]) -> list[Mapping[str, object]]:
        return [record for kind in types for record in grouped.get(kind, [])]

    direction_rows = collect(DIRECTION_BINARY_TYPES)
    distance_rows = collect(DISTANCE_BINARY_TYPES)
    official: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    doa_by_type = {}
    doa_diagnostics_by_type = {}
    for kind in sorted(DOA_TYPES):
        if grouped.get(kind):
            doa_by_type[kind], doa_diagnostics_by_type[kind] = evaluate_doa(grouped[kind])
    if doa_by_type:
        official["doa"] = doa_by_type
        diagnostics["doa"] = doa_diagnostics_by_type
    if direction_rows:
        official["direction_binary_accuracy"] = evaluate_binary(direction_rows)
    if distance_rows:
        official["distance_binary_accuracy"] = evaluate_binary(distance_rows)
    if direction_rows and distance_rows:
        official["average_binary_accuracy"] = (
            official["direction_binary_accuracy"]["accuracy"]
            + official["distance_binary_accuracy"]["accuracy"]
        ) / 2.0

    if direction_rows or distance_rows:
        official["type_e_reference"] = BAT_TYPE_E_REFERENCE

    # Everything below is OUTSIDE the BAT metric set: the eval-stage3-nonbinary
    # split exists in the public release but the paper reports no number for it.
    additional: dict[str, object] = {}
    relative_rows = collect(NONBINARY_DIRECTION_TYPES)
    numeric_rows = collect(NONBINARY_DISTANCE_TYPES)
    source_rows = collect(NONBINARY_SOURCE_TYPES)
    if relative_rows:
        additional["nonbinary_direction_accuracy"] = evaluate_nonbinary_direction(relative_rows)
    if numeric_rows:
        additional["nonbinary_distance"] = evaluate_nonbinary_distance(numeric_rows)
    if source_rows:
        additional["nonbinary_source"] = evaluate_nonbinary_source(source_rows)

    cls_rows = collect(CLASSIFICATION_TYPES)
    return {
        "protocol": "BAT/SpatialSoundQA generated-model metrics v2",
        "metric_provenance": {
            "ssqa_official": "metrics and question types reported in BAT Table 4 "
                             "(A|C mAP, B|D Acc/DER, E Binary Accuracy)",
            "additional_ssqa_tasks": "eval-stage3-nonbinary.json splits, present in the "
                                     "public release but never scored in the paper; "
                                     "extensions with no published reference number",
            "do_not_pool": "stage-3 Binary Accuracy must not be averaged with stage1/2 "
                           "mAP/Acc/DER, and the nonbinary extensions must not be "
                           "averaged into Binary Accuracy",
        },
        "record_count": len(records),
        "question_type_counts": dict(sorted((key, len(value)) for key, value in grouped.items())),
        "ssqa_official": official,
        "classification_records": len(cls_rows),
        "additional_ssqa_tasks": additional,
        "diagnostics": diagnostics,
    }
