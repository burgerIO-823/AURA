#!/usr/bin/env python3
"""Build every AURA training / evaluation manifest from the raw SpatialSoundQA JSONs.

Scene and geometric supervision is recovered from the AudioSet labels and BRIR geometry that
ship with SpatialSoundQA; no audio is read.  Each manifest row carries the question, the
supervised target text ("narrative") and the file name of its pre-encoded RVQ codes.

    python data_preparation/build_manifests.py --ssqa data/SpatialSoundQA --out data/manifests

Outputs (rows):
    train_c1_align.jsonl      55,008   single-source subset for projector alignment
    train_c1.jsonl           278,784   single-source scene supervision             (stage 1)
    train_c2.jsonl           514,784   + dual-source scene supervision             (stage 2)
    train_c3.jsonl           872,312   + Stage-3 geometric supervision             (stage 3)
    eval_stage1.jsonl         29,656 / eval_stage2.jsonl 21,280 / eval_stage3.jsonl 32,298
    report.json                        row counts, question types and SHA256 of every file

Target formats:
    single source   Scene description / Number of sources: 1 / - Source at <dir>, ~<d> m: <labels>.
    two sources     the same with both sources (sorted by location), then the original answer
    Stage 3         listener-centred coordinates of both sources + a geometric analysis that
                    reconstructs the original answer from the BRIR metadata
Every row ends with "Final answer: <original SpatialSoundQA answer>".
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE2_TRAIN_KEY_ORDER = HERE / "stage2_train_key_order.txt.gz"
SCENE_HEADER = "Scene description:"
COUNT_LINE = "Number of sources: {n}"
SCENE_FORMAT = "curriculum_scene_v1"
DUAL_EVAL_FORMAT = "dual_scene_v1"
GEOMETRY_FORMAT = "stage3_geometry_v1"
STAGE3_TYPES = {"MIXUP_DIRECTION", "MIXUP_DISTANCE_BOTH", "MIXUP_NONBINARY_DIRECTION",
                "MIXUP_NONBINARY_DISTANCE", "MIXUP_NONBINARY_SOURCE"}


# ----------------------------------------------------------------------------- SSQA inputs
def load_ssqa_rows(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["data"] if isinstance(data, dict) else data


def norm_label(text):
    return " ".join(text.strip().lower().split())


def load_audioset_labels(metadata_dir, metadata_files=("balanced.json", "eval.json")):
    """audio_id -> sorted AudioSet display names."""
    mid_to_name = {}
    with open(metadata_dir / "class_labels_indices.csv", newline="", encoding="utf-8") as fin:
        for row in csv.DictReader(fin):
            mid_to_name[row["mid"]] = norm_label(row["display_name"])
    by_audio = {}
    for name in metadata_files:
        for row in json.load(open(metadata_dir / name, encoding="utf-8")):
            labels = sorted({mid_to_name[mid] for mid in row["label"] if mid in mid_to_name})
            if labels:
                by_audio[f"{row['folder'].strip('/')}/{row['id']}"] = labels
    return by_audio


def codes_file_name(audio_id, reverb_id, audio_id2=None, reverb_id2=None):
    """Stable file name of one rendered (single- or two-source) scene."""
    def stem(audio):
        return audio.split("/")[-1] if audio else "none"

    def rid(reverb):
        return reverb.replace("/", "_").replace(".npy", "") if reverb else "none"

    raw = "|".join([audio_id or "", reverb_id or "", audio_id2 or "", reverb_id2 or ""])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"mix_{stem(audio_id)}_{rid(reverb_id)}__{stem(audio_id2)}_{rid(reverb_id2)}_{digest}.pt"


class Geometry:
    """Source positions relative to the listener, from the SpatialSoundQA BRIR metadata.

    Frame: x-right, y-above, z-behind (metres).  ``location`` applies the SpatialSoundQA
    quantisation (octant + distance rounded to 0.5 m); ``listener_position`` is unquantised.
    """

    def __init__(self, ssqa):
        self.root = ssqa / "mp3d_reverb" / "binaural"
        self.scenes = {}

    def scene(self, scene_id):
        if scene_id not in self.scenes:
            path = self.root / scene_id / f"{scene_id}.json"
            self.scenes[scene_id] = json.load(open(path, encoding="utf-8"))
        return self.scenes[scene_id]

    def listener_position(self, reverb_id):
        scene_id, filename = reverb_id.split("/", 1)
        item = self.scene(scene_id)[Path(filename).stem]
        source, sensor = item["source"], item["audio_sensor"]
        dx, dy, dz = (source[i] - sensor[i] for i in range(3))
        return {"x": dx, "y": dy, "z": dz, "distance_m": math.sqrt(dx * dx + dy * dy + dz * dz),
                "listener": list(sensor), "source": list(source)}

    def location(self, reverb_id):
        p = self.listener_position(reverb_id)
        return {"lr": "right" if p["x"] > 0 else "left",
                "fb": "behind" if p["z"] > 0 else "front",
                "ud": "above" if p["y"] > 0 else "below",
                "distance_m": round(p["distance_m"] * 2) / 2}


# --------------------------------------------------------------- single / dual scene text
def parse_location(text):
    value = text.lower()

    def pick(a, b):
        return a if re.search(rf"\b{a}\b", value) else (b if re.search(rf"\b{b}\b", value) else None)

    location = {"lr": pick("right", "left"), "fb": pick("behind", "front"), "ud": pick("above", "below")}
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:meters?|m)\b", value)
    location["distance_m"] = float(match.group(1)) if match else None
    if any(v is None for v in location.values()):
        raise ValueError(f"could not parse a complete location from {text!r}: {location}")
    return location


def location_tuple(location):
    return location["lr"], location["fb"], location["ud"], float(location["distance_m"])


def location_sort_key(location):
    return ({"left": 0, "right": 1}[location["lr"]], {"front": 0, "behind": 1}[location["fb"]],
            {"above": 0, "below": 1}[location["ud"]], float(location["distance_m"]))


def render_scene_source(source):
    loc = source["location"]
    d = float(loc["distance_m"])
    distance = f"{int(d)}" if d.is_integer() else f"{d:g}"
    unit = "meter" if d == 1.0 else "meters"
    return (f"- Source at {loc['lr']}, {loc['fb']}, and {loc['ud']}, "
            f"approximately {distance} {unit} away: {'; '.join(source['labels'])}.")


def single_source_row(item, key, labels, geometry, split):
    """Stage-1 question -> single-source scene supervision."""
    qt = item["question_type"]
    if qt not in ("CLASSIFICATION", "DOA"):
        raise ValueError(f"unexpected stage-1 question_type {qt!r}")
    if item.get("audio_id2") not in (None, "", "None") or item.get("reverb_id2") not in (None, "", "None"):
        raise ValueError(f"stage-1 row is not single-source: {item}")
    audio_id, reverb_id, answer = item["audio_id"], item["reverb_id"], item["answer"]
    if not labels.get(audio_id):
        raise ValueError(f"missing AudioSet labels for {audio_id!r}")
    location = geometry.location(reverb_id)
    if qt == "DOA":
        answer_loc = parse_location(answer)
        if location_tuple(answer_loc) != location_tuple(location):
            same_axes = all(answer_loc[a] == location[a] for a in ("lr", "fb", "ud"))
            if not (same_axes and abs(answer_loc["distance_m"] - location["distance_m"]) <= 0.5):
                raise ValueError(f"DOA answer/geometry mismatch: {answer!r} vs {location}")
            # A few distances lie on a 0.5 m rounding boundary; the answer is authoritative.
            location = dict(location, distance_m=answer_loc["distance_m"])
    source = {"audio_id": audio_id, "reverb_id": reverb_id, "location": location,
              "labels": labels[audio_id], "is_target": True}
    scene = "\n".join([SCENE_HEADER, COUNT_LINE.format(n=1), render_scene_source(source)])
    return {
        "key": key, "question_type": qt, "raw_question_type": f"SINGLE_{qt}",
        "question": item["question"], "narrative": f"{scene}\nFinal answer: {answer}",
        "answer": answer, "audio_id": audio_id, "reverb_id": reverb_id,
        "audio_id2": None, "reverb_id2": None,
        "codes_file": codes_file_name(audio_id, reverb_id, None, None),
        "split": split, "original_narrative": answer, "scene_inventory": scene,
        "scene_sources": [source], "final_answer": answer, "target_format": SCENE_FORMAT,
    }


def dual_source_row(item, labels, geometry, split, count_line):
    """Stage-2 question about one of two sources -> scene supervision for both sources.

    count_line=True gives the curriculum training format ("Number of sources: 2");
    the stage-2 evaluation manifest keeps the plain two-source format.
    """
    raw_qt = item["question_type"]
    qt = {"MIXUP_SINGLE_DOA": "DOA", "MIXUP_SINGLE_CLASSIFICATION": "CLASSIFICATION"}[raw_qt]
    answer = item["answer"]
    sources = []
    for audio_key, reverb_key in (("audio_id", "reverb_id"), ("audio_id2", "reverb_id2")):
        audio_id = item[audio_key]
        if not labels.get(audio_id):
            raise ValueError(f"missing AudioSet labels for {audio_id!r}")
        sources.append({"audio_id": audio_id, "reverb_id": item[reverb_key],
                        "location": geometry.location(item[reverb_key]), "labels": labels[audio_id]})
    if location_tuple(sources[0]["location"]) == location_tuple(sources[1]["location"]):
        raise ValueError(f"two sources share one observable location: question {item['question_id']}")
    # The queried source is identified by the location in the answer (DOA) or question (CLS).
    target = parse_location(answer if qt == "DOA" else item["question"])
    matches = [i for i, s in enumerate(sources) if location_tuple(s["location"]) == location_tuple(target)]
    if len(matches) != 1:
        raise ValueError(f"target location matches {len(matches)} sources: question {item['question_id']}")
    sources[matches[0]]["is_target"] = True
    sources[1 - matches[0]]["is_target"] = False
    sources.sort(key=lambda s: location_sort_key(s["location"]))

    header = [SCENE_HEADER] + ([COUNT_LINE.format(n=2)] if count_line else [])
    scene = "\n".join(header + [render_scene_source(s) for s in sources])
    return {
        "key": f"{raw_qt}-{item['question_id']}", "question_type": qt, "raw_question_type": raw_qt,
        "question": item["question"], "narrative": f"{scene}\nFinal answer: {answer}",
        "answer": answer, "audio_id": item["audio_id"], "reverb_id": item["reverb_id"],
        "audio_id2": item.get("audio_id2"), "reverb_id2": item.get("reverb_id2"),
        "codes_file": codes_file_name(item["audio_id"], item["reverb_id"],
                                      item.get("audio_id2"), item.get("reverb_id2")),
        "split": split + ("_curriculum" if count_line else ""),
        "original_narrative": answer, "scene_inventory": scene, "scene_sources": sources,
        "final_answer": answer, "target_format": SCENE_FORMAT if count_line else DUAL_EVAL_FORMAT,
    }


# ------------------------------------------------------------------- Stage-3 geometry text
DIRECTION_RE = re.compile(r"\b(left|right|front|behind|above|below)\b", re.I)
AXIS = {"left": ("x", -1), "right": ("x", 1), "below": ("y", -1), "above": ("y", 1),
        "front": ("z", -1), "behind": ("z", 1)}
RELATIVE = {"left": "to the left of", "right": "to the right of", "front": "in front of",
            "behind": "behind", "above": "above", "below": "below"}


def round_tenth(value):
    result = float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return 0.0 if result == 0 else result


def fmt_tenth(value, signed=False):
    rounded = round_tenth(value)
    if signed and rounded == 0:
        return "-0.0" if value < 0 else "+0.0"
    return f"{rounded:+.1f}" if signed else f"{rounded:.1f}"


def direction_for(vector, axis):
    positive, negative = {"x": ("right", "left"), "y": ("above", "below"), "z": ("behind", "front")}[axis]
    return positive if vector[axis] > 0 else negative


def rounded_vector(vector):
    return {axis: round_tenth(vector[axis]) for axis in ("x", "y", "z")}


def vector_text(v):
    return (f"(x-right={fmt_tenth(v['x'], True)}, y-above={fmt_tenth(v['y'], True)}, "
            f"z-behind={fmt_tenth(v['z'], True)}) m")


def check_answer(item, expected):
    if str(item["answer"]).strip().lower() != str(expected).strip().lower():
        raise ValueError(f"geometry does not reproduce the answer of {item['question_type']} "
                         f"question {item.get('question_id')}: expected {expected!r}, got {item['answer']!r}")


def yes_no(condition):
    return "Yes" if condition else "No"


def consistent(flag):
    return "consistent" if flag else "not consistent"


def direction_claim(item, a, b, delta):
    words = [w.lower() for w in DIRECTION_RE.findall(item["question"])]
    ia, ib = a["source_index"], b["source_index"]
    if "both" in item["question"].lower():
        if len(words) != 1:
            raise ValueError(f"ambiguous both-source question: {item['question']!r}")
        word = words[0]
        axis = AXIS[word][0]
        da, db = direction_for(a["listener_relative_m_raw"], axis), direction_for(b["listener_relative_m_raw"], axis)
        ok = da == word and db == word
        check_answer(item, yes_no(ok))
        return (f"Source {ia} is {da} and Source {ib} is {db} on the {axis}-axis. "
                f"The claim that both are {word} is {consistent(ok)}.")
    if len(words) == 2:
        first, second = words
        axis_a, axis_b = AXIS[first][0], AXIS[second][0]
        da, db = direction_for(a["listener_relative_m_raw"], axis_a), direction_for(b["listener_relative_m_raw"], axis_b)
        ok = da == first and db == second
        check_answer(item, yes_no(ok))
        return (f"Source {ia} is {da} on the {axis_a}-axis, while Source {ib} is {db} on the "
                f"{axis_b}-axis. The stated directions are {consistent(ok)}.")
    if len(words) == 1:
        word = words[0]
        axis = AXIS[word][0]
        actual = direction_for(delta, axis)
        check_answer(item, yes_no(actual == word))
        return (f"Subtracting Source {ib} from Source {ia} gives {vector_text(delta)}. Therefore "
                f"Source {ia} is {RELATIVE[actual]} Source {ib} on the {axis}-axis, so the stated "
                f"relation is {consistent(actual == word)}.")
    raise ValueError(f"unsupported direction question: {item['question']!r}")


def distance_comparison(item, a, b):
    q = item["question"].lower()
    closer = "closer" in q or "nearer" in q or "shorter distance" in q
    farther = "further" in q or "farther" in q or "more distant" in q or "longer distance" in q
    if closer == farther:
        raise ValueError(f"could not identify the distance comparison: {item['question']!r}")
    da, db = a["radial_distance_m_raw"], b["radial_distance_m_raw"]
    ok = da < db if closer else da > db
    check_answer(item, yes_no(ok))
    ia, ib = a["source_index"], b["source_index"]
    return (f"Source {ia} is {fmt_tenth(da)} m from the listener, while Source {ib} is "
            f"{fmt_tenth(db)} m away. Thus Source {ia} is {'closer to' if da < db else 'farther from'} "
            f"the listener than Source {ib}; the claim that it is "
            f"{'closer to' if closer else 'farther from'} the listener is {consistent(ok)}.")


def nonbinary_direction(item, a, b, delta):
    answer = str(item["answer"]).strip().lower()
    if answer not in AXIS:
        raise ValueError(f"invalid direction answer: {item['answer']!r}")
    axis = AXIS[answer][0]
    actual = direction_for(delta, axis)
    check_answer(item, actual)
    ia, ib = a["source_index"], b["source_index"]
    return (f"Subtracting Source {ib} from Source {ia} gives {vector_text(delta)}. On the queried "
            f"{axis}-axis, Source {ia} is {RELATIVE[actual]} Source {ib}.")


def nonbinary_distance(item, a, b, delta, separation):
    expected = round(separation * 2) / 2
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*m\s*", str(item["answer"]), re.I)
    if not match:
        raise ValueError(f"invalid distance answer: {item['answer']!r}")
    if not math.isclose(float(match.group(1)), expected, abs_tol=1e-9):
        check_answer(item, f"{expected:g}m")
    shown = rounded_vector(delta)
    shown_length = math.sqrt(sum(v * v for v in shown.values()))
    ia, ib = a["source_index"], b["source_index"]
    return (f"The Source {ia}-minus-Source {ib} displacement is {vector_text(delta)}. Its "
            f"displayed-coordinate length is sqrt(({fmt_tenth(delta['x'], True)})^2 + "
            f"({fmt_tenth(delta['y'], True)})^2 + ({fmt_tenth(delta['z'], True)})^2) = "
            f"{fmt_tenth(shown_length)} m; the unrounded metadata length is {fmt_tenth(separation)} m, "
            f"which the benchmark rounds to the nearest 0.5 m.")


def nonbinary_source(item, a, b, delta):
    words = [w.lower() for w in DIRECTION_RE.findall(item["question"])]
    if len(words) != 1:
        raise ValueError(f"ambiguous source-relation question: {item['question']!r}")
    axis = AXIS[words[0]][0]
    actual = direction_for(delta, axis)
    if actual != words[0]:
        raise ValueError(f"question relation does not match geometry: {words[0]} vs {actual}")
    answer_labels = sorted({norm_label(x) for x in item["answer"].split(";") if x.strip()})
    if answer_labels != sorted(a["labels"]):
        raise ValueError(f"source labels {a['labels']!r} do not match answer {item['answer']!r}")
    ia, ib = a["source_index"], b["source_index"]
    return (f"Subtracting Source {ib} from Source {ia} gives {vector_text(delta)}. Source {ia} is "
            f"{RELATIVE[actual]} Source {ib} on the {axis}-axis, and Source {ia} contains "
            f"{'; '.join(a['labels'])}.")


def geometry_source(audio_id, reverb_id, labels, geometry):
    p = geometry.listener_position(reverb_id)
    vector = {axis: float(p[axis]) for axis in ("x", "y", "z")}
    return {"audio_id": audio_id, "reverb_id": reverb_id, "labels": labels,
            "listener_relative_m_raw": vector, "listener_relative_m_0p1": rounded_vector(vector),
            "radial_distance_m_raw": float(p["distance_m"]),
            "radial_distance_m_0p1": round_tenth(p["distance_m"]),
            "direction": {"lr": direction_for(vector, "x"), "ud": direction_for(vector, "y"),
                          "fb": direction_for(vector, "z")}}


def render_geometry_source(s):
    d = s["direction"]
    return (f"- Source {s['source_index']}: {'; '.join(s['labels'])}. Listener-centered coordinates "
            f"{vector_text(s['listener_relative_m_raw'])}; direction: {d['lr']}, {d['ud']}, and "
            f"{d['fb']}; radial distance: {fmt_tenth(s['radial_distance_m_raw'])} m.")


def geometry_row(item, labels, geometry, split):
    """Stage-3 relational question -> coordinates of both sources + geometric analysis.

    The row is rejected unless the original answer is reproduced from the BRIR geometry
    (and the AudioSet labels for source questions)."""
    qtype = item["question_type"]
    if qtype not in STAGE3_TYPES:
        raise ValueError(f"unsupported Stage-3 question_type {qtype!r}")
    sources = []
    for audio_key, reverb_key in (("audio_id", "reverb_id"), ("audio_id2", "reverb_id2")):
        audio_id = item[audio_key]
        if not labels.get(audio_id):
            raise ValueError(f"missing AudioSet labels for {audio_id!r}")
        sources.append(geometry_source(audio_id, item[reverb_key], labels[audio_id], geometry))
    if (geometry.listener_position(item["reverb_id"])["listener"]
            != geometry.listener_position(item["reverb_id2"])["listener"]):
        raise ValueError(f"sources do not share a listener: question {item.get('question_id')}")

    # Source numbering follows observable geometry, never the arbitrary audio_id order.
    ordered = sorted(sources, key=lambda s: (s["listener_relative_m_raw"]["x"], s["listener_relative_m_raw"]["y"],
                                             s["listener_relative_m_raw"]["z"], s["radial_distance_m_raw"],
                                             tuple(s["labels"])))
    for index, source in enumerate(ordered, 1):
        source["source_index"] = index
    a, b = sources
    delta = {axis: a["listener_relative_m_raw"][axis] - b["listener_relative_m_raw"][axis] for axis in ("x", "y", "z")}
    separation = math.sqrt(sum(v * v for v in delta.values()))
    if qtype == "MIXUP_DIRECTION":
        analysis = direction_claim(item, a, b, delta)
    elif qtype == "MIXUP_DISTANCE_BOTH":
        analysis = distance_comparison(item, a, b)
    elif qtype == "MIXUP_NONBINARY_DIRECTION":
        analysis = nonbinary_direction(item, a, b, delta)
    elif qtype == "MIXUP_NONBINARY_DISTANCE":
        analysis = nonbinary_distance(item, a, b, delta, separation)
    else:
        analysis = nonbinary_source(item, a, b, delta)

    scene = "\n".join([SCENE_HEADER, COUNT_LINE.format(n=2)] + [render_geometry_source(s) for s in ordered])
    final_answer = str(item["answer"]).strip()
    row = dict(item)
    row.update({
        "key": f"{qtype}-{item['question_id']}", "raw_question_type": qtype,
        "original_narrative": item.get("narrative", final_answer),
        "scene_inventory": scene, "scene_sources": ordered,
        "pair_geometry": {
            "source_a_index": a["source_index"], "source_b_index": b["source_index"],
            "delta_a_minus_b_m_raw": delta, "delta_a_minus_b_m_0p1": rounded_vector(delta),
            "a_relative_to_b": {"lr": direction_for(delta, "x"), "ud": direction_for(delta, "y"),
                                "fb": direction_for(delta, "z")},
            "inter_source_distance_m_raw": separation,
            "inter_source_distance_m_0p1": round_tenth(separation)},
        "geometric_analysis": analysis, "final_answer": final_answer,
        "narrative": f"{scene}\nGeometric analysis:\n{analysis}\nFinal answer: {final_answer}",
        "target_format": GEOMETRY_FORMAT, "split": split,
        "codes_file": codes_file_name(item["audio_id"], item["reverb_id"], item["audio_id2"], item["reverb_id2"]),
    })
    return row


# ------------------------------------------------------------------------------- manifests
def single_source_split(ssqa_files, labels, geometry, split):
    rows, counters = [], Counter()
    for path in ssqa_files:
        for item in load_ssqa_rows(path):
            raw_qt = f"SINGLE_{item['question_type']}"
            rows.append(single_source_row(item, f"{raw_qt}-{counters[raw_qt]}", labels, geometry, split))
            counters[raw_qt] += 1
    return rows


def dual_source_split(ssqa_files, labels, geometry, split, count_line, key_order=None):
    items = [item for path in ssqa_files for item in load_ssqa_rows(path)
             if item["question_type"] in ("MIXUP_SINGLE_DOA", "MIXUP_SINGLE_CLASSIFICATION")]
    if key_order is not None:
        with gzip.open(key_order, "rt", encoding="utf-8") as stream:
            order = [line.strip() for line in stream if line.strip()]
        by_key = {f"{it['question_type']}-{it['question_id']}": it for it in items}
        if len(by_key) != len(items) or set(order) != set(by_key) or len(order) != len(by_key):
            raise SystemExit(f"{key_order} does not match the stage-2 training questions")
        items = [by_key[key] for key in order]
    return [dual_source_row(item, labels, geometry, split, count_line) for item in items]


def geometry_split(ssqa_files, labels, geometry, split):
    return [geometry_row(item, labels, geometry, split)
            for path in ssqa_files for item in load_ssqa_rows(path)
            if item.get("question_type") in STAGE3_TYPES]


def align_subset(rows, per_type=27504, seed=42):
    """Seeded subset with the same number of DOA and CLS rows, kept in manifest order."""
    rng, picked = random.Random(seed), []
    for raw_qt in ("SINGLE_DOA", "SINGLE_CLASSIFICATION"):
        picked.extend(rng.sample([i for i, r in enumerate(rows) if r["raw_question_type"] == raw_qt], per_type))
    return [rows[i] for i in sorted(picked)]


def audio_ids(rows):
    return {r[k] for r in rows for k in ("audio_id", "audio_id2") if r.get(k)}


def write_manifest(rows, path):
    tmp = path.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    tmp.replace(path)
    return digest.hexdigest()


EXPECTED_ROWS = {"train_c1_align": 55008, "train_c1": 278784, "train_c2": 514784,
                 "train_c3": 872312,
                 "eval_stage1": 29656, "eval_stage2": 21280, "eval_stage3": 32298}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssqa", type=Path, default=Path("data/SpatialSoundQA"), help="SpatialSoundQA root")
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    args = ap.parse_args()
    ce = args.ssqa / "closed-end"
    labels = load_audioset_labels(args.ssqa / "AudioSet" / "metadata")
    geometry = Geometry(args.ssqa)

    print("[1/4] single-source scene supervision (stage 1)")
    c1 = single_source_split([ce / "stage1-clsdoa" / "train.json"], labels, geometry, "curriculum_c1_single_v1")
    eval1 = single_source_split([ce / "stage1-clsdoa" / "eval-stage1-classification.json",
                                 ce / "stage1-clsdoa" / "eval-stage1-doa.json"],
                                labels, geometry, "curriculum_c1_eval_v1")
    print("[2/4] dual-source scene supervision (stage 2)")
    dual = dual_source_split([ce / "stage2-single" / "train.json"], labels, geometry,
                             "stage2_lrs_full236k_dual_scene_v1", count_line=True,
                             key_order=STAGE2_TRAIN_KEY_ORDER)
    eval2 = dual_source_split([ce / "stage2-single" / "eval-stage2-doa.json",
                               ce / "stage2-single" / "eval-stage2-classification.json"],
                              labels, geometry, "stage2_eval_full_dual_scene_v1", count_line=False)
    print("[3/4] geometric supervision (stage 3)")
    geo = geometry_split([ce / "stage3-mixup" / "train.json"], labels, geometry, "train_stage3_geometry_v1")
    eval3 = geometry_split([ce / "stage3-mixup" / f"eval-stage3-{name}.json"
                            for name in ("direction", "distance", "nonbinary")],
                           labels, geometry, "eval_stage3_geometry_v1")

    manifests = {"train_c1_align": align_subset(c1), "train_c1": c1, "train_c2": c1 + dual,
                 "train_c3": c1 + dual + geo,
                 "eval_stage1": eval1, "eval_stage2": eval2, "eval_stage3": eval3}
    for name, rows in manifests.items():
        if len(rows) != EXPECTED_ROWS[name]:
            raise SystemExit(f"{name}: {len(rows)} rows, expected {EXPECTED_ROWS[name]}")
        if len({r["key"] for r in rows}) != len(rows):
            raise SystemExit(f"{name}: duplicate keys")
    for train, evaluation in (("train_c1", "eval_stage1"), ("train_c2", "eval_stage2"), ("train_c3", "eval_stage3")):
        overlap = audio_ids(manifests[train]) & audio_ids(manifests[evaluation])
        if overlap:
            raise SystemExit(f"{train} / {evaluation} share {len(overlap)} audio clips")

    print("[4/4] writing manifests")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rows in manifests.items():
        sha = write_manifest(rows, args.out / f"{name}.jsonl")
        report[name] = {"rows": len(rows), "sha256": sha,
                        "question_types": dict(sorted(Counter(r["raw_question_type"] for r in rows).items()))}
        print(f"  {name:18s} {len(rows):>7,d} rows  sha256 {sha[:16]}")
    with open(args.out / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
