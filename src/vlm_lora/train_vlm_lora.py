"""Standalone LoRA SFT of Cosmos-Reason2-2B (Qwen3-VL) on a VQA dataset."""

import os

import torch
import tyro
from peft import get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from vlm_lora.hf_utils import resolve_model_path
from vlm_lora.lora_args import VlmLoraConfig, build_lora_config
from vlm_lora.vqa_dataset import VqaCollator, VqaJsonlDataset


def main():
    cfg = tyro.cli(VlmLoraConfig)
    base = resolve_model_path(cfg.base_model)
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    if cfg.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, build_lora_config(cfg))
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=10,
        bf16=cfg.bf16,
        optim=cfg.optim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        report_to=(["wandb"] if cfg.use_wandb else []),
        remove_unused_columns=False,
    )
    if cfg.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    Trainer(
        model=model,
        args=args,
        train_dataset=VqaJsonlDataset(cfg.dataset_path, image_root=cfg.image_root),
        data_collator=VqaCollator(processor, cfg.max_seq_len),
    ).train()
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"[train] adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
