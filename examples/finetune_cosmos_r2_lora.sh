#!/usr/bin/env bash
# LoRA fine-tune the Cosmos-Reason2-2B VLM on a VQA dataset (single H100).
# Cosmos-R2 is gated: export HF_TOKEN=hf_... first.
set -x -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to the VQA jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/cosmos_r2_lora}"
BASE_MODEL="${BASE_MODEL:-nvidia/Cosmos-Reason2-2B}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET="${LORA_TARGET:-llm}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  -m vlm_lora.train_vlm_lora
  --dataset-path "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --base-model "$BASE_MODEL"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-target "$LORA_TARGET"
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --learning-rate "$LEARNING_RATE"
)
[ -n "$IMAGE_ROOT" ] && ARGS+=( --image-root "$IMAGE_ROOT" )
[ "${USE_WANDB:-0}" = "1" ] && ARGS+=( --use-wandb )
ARGS+=( "$@" )

mkdir -p "$OUTPUT_DIR"
exec uv run python "${ARGS[@]}"
