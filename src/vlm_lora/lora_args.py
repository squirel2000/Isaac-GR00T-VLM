"""CLI config + PEFT LoraConfig builder for standalone Cosmos-Reason2-2B LoRA tuning."""

from dataclasses import dataclass

from peft import LoraConfig

_DEFAULT_LLM_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class VlmLoraConfig:
    dataset_path: str
    output_dir: str
    base_model: str = "nvidia/Cosmos-Reason2-2B"
    image_root: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: str = "llm"  # 'llm' | 'all-linear' | 'q_proj,v_proj,...'
    lora_on_vision: bool = False
    max_steps: int = 2000
    learning_rate: float = 1e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_seq_len: int = 2048
    save_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 0
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "vlm-lora-cosmos-r2"


def build_lora_config(cfg: VlmLoraConfig) -> LoraConfig:
    if cfg.lora_target == "all-linear":
        targets = "all-linear"
    elif cfg.lora_target == "llm":
        targets = list(_DEFAULT_LLM_TARGETS)
        if cfg.lora_on_vision:
            targets += ["qkv", "proj", "fc1", "fc2"]
    else:
        targets = [t.strip() for t in cfg.lora_target.split(",") if t.strip()]
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
