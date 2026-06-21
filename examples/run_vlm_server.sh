#!/usr/bin/env bash
# Serve the fine-tuned VLM as AgentBot's Brain. Env: VLM_MODEL_DIR (required), PORT, CUDA_VISIBLE_DEVICES.
set -euo pipefail
: "${VLM_MODEL_DIR:?set VLM_MODEL_DIR to the merged VLM dir}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec uv run uvicorn vlm_lora.serve.openai_app:app --host 0.0.0.0 --port "${PORT:-8000}"
