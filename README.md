# vlm_lora_finetune

獨立 repo：對 **Cosmos-Reason2-2B** VLM 做 **LoRA 微調**（VQA），合併成可獨立部署的 standalone VLM，再把權重**置換進 GR00T N1.7 VLA**，並做兩方對照評估（對齊附件 PDF 的「可置換權重」流程）。

**完全不修改 `Isaac-GR00T_n1d7/`** —— 本 repo 自帶輕量 uv 環境負責生成/訓練/合併/置換/推論/VQA 評估；只有 swapped VLA 的 open-loop eval 會在 H100 的 n1d7 環境唯讀執行既有 `open_loop_eval.py`。

## 產物總表

| 代號 | 內容 | 結構 | 產生方式 | 用途 |
|---|---|---|---|---|
| **base** | 微調前 Cosmos-R2 快照 | 完整 36 層 | snapshot | VLM 微調「前」對照基準 |
| **產物 A** | Standalone VLM（LoRA 合併） | 完整 36 層 | `merge_lora.py` | 可獨立部署/分享、VQA 用；swap 來源 |
| **swapped VLA** | A 換進 VLA（保留 action head） | VLA | `swap_backbone.py` | action 評估（open-loop MSE/MAE） |

## 流程（H100 + uv）

```bash
# 0) 環境
uv sync --extra dev

# 1) 從 OpenArm LeRobot 自動生成 VQA（teacher-VLM + template）
uv run python -m vlm_lora.gen_vqa_from_lerobot \
    --dataset-path /data/VLA/datasets/OpenArm_CanSorting_MultiTask_dataset_O6_0403 \
    --out-dir artifacts/vqa --num-episodes 200

# 2) LoRA 微調 Cosmos-Reason2-2B（需 export HF_TOKEN）
uv run python -m vlm_lora.train_vlm_lora \
    --dataset-path artifacts/vqa/data.train.jsonl --image-root artifacts/vqa \
    --output-dir artifacts/cosmos_r2_lora --max-steps 2000

# 3) 合併 -> 產物 A（standalone VLM）
uv run python -m vlm_lora.merge_lora \
    --adapter-dir artifacts/cosmos_r2_lora --out-dir artifacts/Cosmos-Reason2-2B-lora-merged

# 4) 置換進 VLA -> swapped VLA
uv run python -m vlm_lora.swap_backbone \
    --vla-ckpt <baseline_vla_ckpt> --merged-vlm artifacts/Cosmos-Reason2-2B-lora-merged \
    --out-dir <swapped_vla_out>

# 5) 評估：VLM VQA（base vs A）+ VLA open-loop MSE/MAE（baseline vs swapped）
```

詳見 [docs/vlm_lora_finetune.md](docs/vlm_lora_finetune.md)。
