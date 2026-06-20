# VLM LoRA Fine-Tuning — Operator 指南

對 **Cosmos-Reason2-2B** VLM 做 LoRA 微調（VQA）→ 合併成 standalone VLM → 置換進 **GR00T N1.7 VLA** → 兩方評估。獨立 repo，**不修改 `Isaac-GR00T_n1d7`**。

## 雙環境
- **本 repo（`vlm_lora_finetune`）的 uv env**：生成 VQA / LoRA 訓練 / 合併 / 置換 / standalone 推論 / VLM VQA 評估。依賴 pin 對齊 n1d7 的 cu128 wheels（`torch==2.7.1`），故 `uv sync` 直接重用快取、免下載。
- **n1d7 的 uv env**：只用於 **swapped VLA 的 open-loop 評估**（需 `gr00t`），唯讀執行既有 `examples/Openarm_LinkerHandO6/openloop_eval_openarm_o6.sh`。
- 本機（Windows）以 `python scripts/common/pegasus.py …`（plain python，含 requests/truststore）連 H100；**不是** `uv run`。

## 產物（H100，IsaacLab-GR00T 根目錄下）
| 代號 | 路徑 | 用途 |
|---|---|---|
| base | `Isaac-GR00T-VLM/artifacts/Cosmos-Reason2-2B-base/` | 微調前對照 |
| 產物 A | `artifacts/checkpoints/gr00t/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged/` | standalone VLM（可部署/分享） |
| swapped VLA | `artifacts/checkpoints/gr00t/swapped_checkpoints/N1_7_cosmosR2lora_swapped/` | action 評估 |

## H100 流程（指令）
```bash
# 1) VQA 自動生成（teacher=Qwen3-VL-30B-A3B MoE + template）
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.gen_vqa_from_lerobot \
  --dataset-path /data/VLA/datasets/OpenArm_CanSorting_MultiTask_dataset_O6_0403 \
  --out-dir artifacts/vqa --num-episodes 100 --frames-per-episode 3
# 2) LoRA 訓練
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.train_vlm_lora \
  --dataset-path artifacts/vqa/data.train.jsonl --image-root artifacts/vqa \
  --output-dir artifacts/cosmos_r2_lora --max-steps 2000 --lora-r 16 --lora-alpha 32
# 3) 合併 -> 產物 A
HF_HUB_OFFLINE=1 uv run python -m vlm_lora.merge_lora \
  --adapter-dir artifacts/cosmos_r2_lora --out-dir artifacts/Cosmos-Reason2-2B-lora-merged
# 4) 置換 -> swapped VLA
uv run python -m vlm_lora.swap_backbone \
  --vla-ckpt <baseline_vla_ckpt> --merged-vlm artifacts/Cosmos-Reason2-2B-lora-merged \
  --out-dir <swapped_out>
# 5a) VLM VQA 準確率（base vs lora）
HF_HUB_OFFLINE=1 uv run python -m vlm_lora.eval_vlm_vqa \
  --model-dir <base|merged> --val-jsonl artifacts/vqa/data.val.jsonl --image-root artifacts/vqa --out-json <out>
# 5b) VLA open-loop（n1d7 env）
CHECKPOINT_PATH=<ckpt> DATASET_PATH=<ds> EMBODIMENT_TAG=new_embodiment TRAJ_IDS="0" \
  bash examples/Openarm_LinkerHandO6/openloop_eval_openarm_o6.sh
```
長任務用 `examples/run_pegasus.py` 或 `tmp/dispatch.py`（nohup+setsid，斷線可續）。

## Troubleshooting（實際踩過的坑）
- **gated 模型離線載入**：用 bare id `nvidia/Cosmos-Reason2-2B` 會觸發 tokenizer 的 hub 探測（`is_base_mistral`），在 `HF_HUB_OFFLINE=1` 下報錯。解法：`resolve_model_path()` 解析到本地 cache snapshot 目錄再載入（`_is_local=True` 跳過探測）。支援舊式 flat cache 佈局。
- **collator image token**：messages content 必須是結構化 `[{type:image},{type:text}]`；純 `<image>` 字串 → chat template 不展開 → `tokens:0 vs features:N` 報錯。`VqaCollator.to_template_messages()` 負責轉換。
- **torch 版本**：unpinned `torch` 會解析到 2.12.1(cu13) 且下載失敗；pin `torch==2.7.1` + `pytorch-cu128` index 重用 n1d7 快取。
- **MoE teacher**：`Qwen3-VL-30B-A3B` 用 `Qwen3VLMoeForConditionalGeneration`（4.57.3 已含）；務必 `dtype=bf16`（fp32 ~124GB 會 OOM 80GB）；`device_map=auto`；teacher 只載一次（勿逐 frame 重載）。
- **detached wrapper CRLF**：Windows 寫 `.sh` 要用 LF（`write_text(..., newline="\n")`），否則 `bash` 因 `\r` 不執行。
- **commit 目標**：本 repo 是 root 下的獨立 git repo；commit 用 `git -C vlm_lora_finetune`，勿污染外層 `IsaacLab-GR00T`。

## 報告
- `docs/project_report.html` — 本專案結果（動畫架構/流程、VQA 統計、teacher 對照、評估表）。
- `docs/llm_vlm_finetuning_guide.html` — 通用 LLM/VLM 微調教學（LoRA / MoE / 多模態，含動畫）。
