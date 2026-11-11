# SpatialSoundQA (BAT) scoring

Stand-alone implementation of the official BAT / SpatialSoundQA metrics used in the paper.
`eval.sh` calls these scripts; they can also score any prediction JSONL directly.

## Metrics

- Sound event detection (stage 1/2 classification): mAP from `text-embedding-3-small`
  similarity between the generated text and every AudioSet class name;
- DoA: exact match of the 8-octant direction (left/right × front/behind × above/below);
- Distance: DER, an answer is wrong when `|pred - gt| > 0.5 m`;
- Stage 3: Direction Binary Accuracy, Distance Binary Accuracy and their mean;
- Additional (unpublished) SpatialSoundQA splits: nonbinary direction accuracy / distance DER.

Every ground-truth item is in the denominator; a missing or unparsable prediction counts as wrong.

## Scoring a prediction file

```bash
python evaluation/evaluate.py \
  --predictions runs/eval/stage2/predictions.jsonl --subset held-out \
  --expect-records 21280 \
  --expect-count MIXUP_SINGLE_DOA=10640 --expect-count MIXUP_SINGLE_CLASSIFICATION=10640 \
  --output runs/eval/stage2/ssqa_summary.json
```

Input fields: `qt`/`question_type`, `gt`/`answer`, `pred`/`raw`, `subset`, `key`.
Duplicate `(subset, key)` pairs are rejected; `--expect-records` / `--expect-count` make a
partial prediction file fail instead of producing a complete-looking summary.

## Classification mAP

Class and prediction embeddings must come from the same model (checked from the NPZ metadata).
Both are generated with the OpenAI API (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`) via
`embeddings.py` — class names from the label CSV, predictions from the prediction file:

```bash
META=$SSQA_ROOT/AudioSet/metadata
python evaluation/embeddings.py \
  --labels-csv "$META/class_labels_indices_subset.csv" \
  --model text-embedding-3-small --output runs/eval/stage2/class_embeddings.npz
python evaluation/embeddings.py \
  --predictions runs/eval/stage2/predictions.jsonl --subset held-out \
  --model text-embedding-3-small --output runs/eval/stage2/prediction_embeddings.npz
python evaluation/evaluate.py \
  --predictions runs/eval/stage2/predictions.jsonl --subset held-out \
  --labels-csv "$META/class_labels_indices_subset.csv" \
  --class-embeddings runs/eval/stage2/class_embeddings.npz \
  --prediction-embeddings runs/eval/stage2/prediction_embeddings.npz \
  --output runs/eval/stage2/ssqa_summary.json
```

## Per-source scene attribution (stage 2)

`scene_attribution.py` scores the generated two-source scene description independently of the
final answer. Predicted and ground-truth sources are matched by label overlap (order invariant);
it reports parse rate, per-source label P/R/F1, three-axis direction, distance DER, full-tuple
accuracy, and target / distractor breakdowns.

```bash
python evaluation/scene_attribution.py \
  --manifest data/manifests/eval_stage2.jsonl \
  --predictions runs/eval/stage2/predictions.jsonl --subset held-out \
  --expect-records 21280 --output runs/eval/stage2/scene_attribution.json
```
