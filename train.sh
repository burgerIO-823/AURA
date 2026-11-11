#!/usr/bin/env bash
# Train one AURA curriculum stage (recipes in training/configs/, 4x H100 80GB layout).
#
#   bash train.sh <c1_align|c1_joint|c2_joint|c3_joint> [KEY=VALUE ...]
#
# Full curriculum (each stage warm-starts from the previous <out_dir>/final):
#   c1_align -> c1_joint (=stage1_c1) -> c2_joint (=stage2_c2) -> c3_joint (stage 3)
#
# Overrides:  NUM_GPUS / BS change the per-device layout; ACCUM is derived so that the
#             recipe's GLOBAL_BATCH (and thus the optimisation) stays unchanged.
#             CODES_DIR (data/codes)  QWEN (models/Qwen3-8B)  CODEBOOKS (models/xcodec/codebooks.pt)
#             OUT_DIR  INIT  WANDB=1  DRY_RUN=1 (print the command only)  SMOKE=1 (one fwd+bwd)
# Training resumes automatically from the newest complete checkpoint in OUT_DIR.
set -euo pipefail
cd "$(dirname "$0")"

STAGE=${1:-}
CFG=training/configs/${STAGE}.env
[ -n "$STAGE" ] && [ -f "$CFG" ] || {
  echo "usage: $0 <stage> [KEY=VALUE ...]; stages:" >&2
  ls training/configs/*.env | sed 's#.*/##; s#\.env$##; s#^#  #' >&2; exit 2; }
shift

ENV_NUM_GPUS=${NUM_GPUS:-}; ENV_BS=${BS:-}
# shellcheck disable=SC1090
. "$CFG"
for kv in "$@"; do
  [[ "$kv" =~ ^[A-Z_][A-Z0-9_]*= ]] || { echo "[fatal] override must be KEY=VALUE: $kv" >&2; exit 2; }
  declare "${kv?}"
  case "${kv%%=*}" in BS) ENV_BS=${kv#*=} ;; NUM_GPUS) ENV_NUM_GPUS=${kv#*=} ;; esac
done
NUM_GPUS=${ENV_NUM_GPUS:-$NUM_GPUS}
BS=${ENV_BS:-$BS}
PY=${PY:-python}
CODES_DIR=${CODES_DIR:-data/codes}
QWEN=${QWEN:-models/Qwen3-8B}
CODEBOOKS=${CODEBOOKS:-models/xcodec/codebooks.pt}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}

per_step=$((BS * NUM_GPUS))
(( GLOBAL_BATCH % per_step == 0 )) || {
  echo "[fatal] GLOBAL_BATCH=$GLOBAL_BATCH not divisible by NUM_GPUS*BS=$per_step" >&2; exit 2; }
ACCUM=$((GLOBAL_BATCH / per_step))

[ -f "$MANIFEST" ] || { echo "[fatal] missing $MANIFEST (run data_preparation/build_manifests.py)" >&2; exit 1; }
rows=$(wc -l < "$MANIFEST")
[ "$rows" -eq "$EXPECT_ROWS" ] || { echo "[fatal] $MANIFEST has $rows rows, expected $EXPECT_ROWS" >&2; exit 1; }
if [ -n "$INIT" ] && [ ! -f "$INIT/aura_model.safetensors" ]; then
  echo "[fatal] INIT=$INIT is not an AURA checkpoint (train the previous stage first," >&2
  echo "        or point INIT at a released checkpoint directory)" >&2; exit 1
fi

ARGS=(--manifest "$MANIFEST" --codes_dir "$CODES_DIR" --qwen "$QWEN" --codebooks "$CODEBOOKS"
      --seed "$SEED" --epochs 1 --bs "$BS" --accum "$ACCUM" --lr "$LR" --warmup "$WARMUP"
      --num_workers "$NUM_WORKERS")
[ "$FREEZE_LLM" = 1 ] && ARGS+=(--freeze_llm)

if [ "${SMOKE:-0}" = 1 ]; then
  [ -n "$INIT" ] && ARGS+=(--init "$INIT")
  CMD=("$PY" -u training/train.py "${ARGS[@]}" --out_dir "/tmp/aura_smoke_$STAGE" --smoke)
else
  if [ -f "$OUT_DIR/final/aura_model.safetensors" ]; then
    echo "[skip] $OUT_DIR/final already exists"; exit 0
  fi
  if ls -d "$OUT_DIR"/checkpoint-* >/dev/null 2>&1; then
    ARGS+=(--resume)                       # auto: newest complete checkpoint
  elif [ -n "$INIT" ]; then
    ARGS+=(--init "$INIT")                 # fresh optimizer / cosine schedule
  fi
  [ "${WANDB:-0}" = 1 ] && ARGS+=(--wandb --wandb_project "${WANDB_PROJECT:-aura}" --wandb_run "$STAGE")
  ARGS+=(--save_steps "$SAVE_STEPS" --out_dir "$OUT_DIR")
  if [ "$NUM_GPUS" -gt 1 ]; then
    CMD=(torchrun --standalone --nproc_per_node "$NUM_GPUS" training/train.py "${ARGS[@]}")
  else
    CMD=("$PY" -u training/train.py "${ARGS[@]}")
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1
echo "[train] stage=$STAGE rows=$rows layout=${NUM_GPUS}x${BS}x${ACCUM} (global $GLOBAL_BATCH)" \
     "lr=$LR warmup=$WARMUP expect_steps=${EXPECT_STEPS:-?} init=${INIT:-none}"
echo "+ ${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
mkdir -p "$OUT_DIR"
"${CMD[@]}"
