#!/usr/bin/env bash
# Download the frozen components AURA builds on and extract the RVQ codebooks.
#   models/Qwen3-8B                         frozen LLM backbone
#   models/xcodec/                          X-Codec general-audio checkpoint (v2) + config
#   models/xcodec/codebooks.pt              8 RVQ codebooks (8x1024x1024), the only codec part used in training
#   third_party/xcodec                      official X-Codec code (needed only to encode audio)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
huggingface-cli download ZhenYe234/xcodec xcodec_hubert_general_audio_v2.pth config_hubert_general.yaml \
  --local-dir models/xcodec
[ -d third_party/xcodec ] || git clone https://github.com/zhenye234/xcodec.git third_party/xcodec
# X-Codec loads the HuBERT general-audio semantic teacher from the Hub on first use:
#   ZhenYe234/hubert_base_general_audio

if [ ! -f models/xcodec/codebooks.pt ]; then
  "$PY" data_preparation/extract_codebooks.py \
    --ckpt models/xcodec/xcodec_hubert_general_audio_v2.pth --out models/xcodec/codebooks.pt
fi
echo "[done] models ready"
