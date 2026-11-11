#!/usr/bin/env bash
# Formal SpatialSoundQA evaluation of one AURA checkpoint (greedy decoding, full eval sets).
#
#   bash eval.sh <ckpt_dir> <stage1|stage2|stage3|all> [out_dir]
#
#   stage1  single-source  29,656 rows  DoA accuracy / DER        max_new_tokens 192
#   stage2  dual-source    21,280 rows  DoA accuracy / DER        max_new_tokens 192
#   stage3  reasoning      32,298 rows  Direction / Distance BA   max_new_tokens 384
# The decode budgets are part of the protocol and must not be swapped between stages.
#
# Env: NUM_GPUS (default: all visible; >1 runs one shard per GPU)  BATCH_SIZE (128)
#      CODES_DIR (data/codes)  QWEN  CODEBOOKS  SSQA_ROOT
#      COMPUTE_MAP=1  also compute the classification embedding mAP (needs OPENAI_API_KEY for
#                     text-embedding-3-small embeddings of the class names and predictions)
set -euo pipefail
cd "$(dirname "$0")"

CKPT=${1:-}; WHICH=${2:-}; OUT_ROOT=${3:-}
[ -n "$CKPT" ] && [ -n "$WHICH" ] || { echo "usage: $0 <ckpt_dir> <stage1|stage2|stage3|all> [out_dir]" >&2; exit 2; }
[ -f "$CKPT/aura_model.safetensors" ] && [ -f "$CKPT/proj.pt" ] || { echo "[fatal] not an AURA checkpoint: $CKPT" >&2; exit 1; }
OUT_ROOT=${OUT_ROOT:-$CKPT/eval}
PY=${PY:-python}
CODES_DIR=${CODES_DIR:-data/codes}
QWEN=${QWEN:-models/Qwen3-8B}
CODEBOOKS=${CODEBOOKS:-models/xcodec/codebooks.pt}
SSQA_ROOT=${SSQA_ROOT:-data/SpatialSoundQA}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_GPUS=${NUM_GPUS:-$("$PY" -c 'import torch; print(max(1, torch.cuda.device_count()))')}
COMPUTE_MAP=${COMPUTE_MAP:-0}
M=data/manifests

run_stage() {
  local stage=$1 man n tokens counts=()
  case "$stage" in
    stage1) man=$M/eval_stage1.jsonl; n=29656; tokens=192
            counts=(DOA=14828 CLASSIFICATION=14828) ;;
    stage2) man=$M/eval_stage2.jsonl; n=21280; tokens=192
            counts=(MIXUP_SINGLE_DOA=10640 MIXUP_SINGLE_CLASSIFICATION=10640) ;;
    stage3) man=$M/eval_stage3.jsonl; n=32298; tokens=384
            counts=(MIXUP_DIRECTION=12002 MIXUP_DISTANCE_BOTH=10600 MIXUP_NONBINARY_DIRECTION=3232
                    MIXUP_NONBINARY_DISTANCE=3232 MIXUP_NONBINARY_SOURCE=3232) ;;
  esac
  local out=$OUT_ROOT/$stage pred
  mkdir -p "$out"; pred=$out/predictions.jsonl
  [ "$(wc -l < "$man")" -eq "$n" ] || { echo "[fatal] $man != $n rows" >&2; exit 1; }

  echo "[$stage] audio-free same-set baseline"
  "$PY" evaluation/baselines.py --manifest "$man" --output "$out/baselines.json" >/dev/null

  echo "[$stage] generate n=$n max_new_tokens=$tokens batch=$BATCH_SIZE gpus=$NUM_GPUS (greedy)"
  "$PY" -u evaluation/generate.py --ckpt "$CKPT" --manifest "$man" --codes_dir "$CODES_DIR" \
    --qwen "$QWEN" --codebooks "$CODEBOOKS" --max_new_tokens "$tokens" \
    --batch_size "$BATCH_SIZE" --num_gpus "$NUM_GPUS" --out "$pred"

  local args=(--predictions "$pred" --subset held-out --expect-records "$n" --output "$out/ssqa_summary.json")
  for c in "${counts[@]}"; do args+=(--expect-count "$c"); done
  if [ "$COMPUTE_MAP" = 1 ] && [ "$stage" != stage3 ]; then
    "$PY" evaluation/embeddings.py --predictions "$pred" --subset held-out \
      --model text-embedding-3-small --output "$out/prediction_embeddings.npz"
    "$PY" evaluation/embeddings.py --labels-csv "$SSQA_ROOT/AudioSet/metadata/class_labels_indices_subset.csv" \
      --model text-embedding-3-small --output "$out/class_embeddings.npz"
    args+=(--labels-csv "$SSQA_ROOT/AudioSet/metadata/class_labels_indices_subset.csv"
           --class-embeddings "$out/class_embeddings.npz"
           --prediction-embeddings "$out/prediction_embeddings.npz")
  fi
  echo "[$stage] BAT/SSQA metrics -> $out/ssqa_summary.json"
  "$PY" evaluation/evaluate.py "${args[@]}"
  if [ "$stage" = stage2 ]; then
    "$PY" evaluation/scene_attribution.py --manifest "$man" --predictions "$pred" \
      --subset held-out --expect-records "$n" --output "$out/scene_attribution.json"
  fi
}

case "$WHICH" in
  stage1|stage2|stage3) run_stage "$WHICH" ;;
  all) for s in stage1 stage2 stage3; do run_stage "$s"; done ;;
  *) echo "[fatal] unknown stage $WHICH" >&2; exit 2 ;;
esac
echo "[done] results in $OUT_ROOT"
