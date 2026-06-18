"""Merge LoRA adapter into base Cosmos-R2 -> standalone VLM (Product A)."""

import os

import torch
import tyro
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from vlm_lora.hf_utils import resolve_model_path


def merge(adapter_dir: str, out_dir: str, base_model: str = "nvidia/Cosmos-Reason2-2B") -> None:
    assert "Cosmos-Reason2" in os.path.basename(
        out_dir.rstrip("/")
    ), "out-dir basename must contain 'Cosmos-Reason2'"
    base = resolve_model_path(base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(base, trust_remote_code=True).save_pretrained(out_dir)
    print(f"[merge] Product A (standalone VLM) saved to {out_dir}")


if __name__ == "__main__":
    tyro.cli(merge)
