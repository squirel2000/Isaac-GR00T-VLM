# Isaac-GR00T-VLM

對 **Cosmos-Reason2-2B** VLM（GR00T N1.7 的 backbone）做 LoRA 微調，合併成**獨立 VLM**，再**置換回 GR00T N1.7 VLA**。與 `Isaac-GR00T_n1d7` 完全解耦——不修改其原始碼。

整條流程（VQA 生成 → LoRA → 合併 → 置換 → 評估）在**單張 GPU** 上即可跑完。本機已於 **RTX 4090（24 GB）** 全程驗證；teacher 採 dense 的 `Qwen/Qwen3-VL-8B-Instruct`（H100 可改用 30B-A3B MoE）。

## 快速開始（五步）

設定集中在 [`configs/default.yaml`](configs/default.yaml)（每段皆有註解）；CLI flag 與 `examples/finetune_cosmos_r2_lora.sh` 的環境變數同名、皆可覆蓋。

```bash
cd Isaac-GR00T-VLM
uv sync --extra dev          # 建環境（沿用 n1d7 的 cu128 wheels）
DS=/home/asus/Gits/IsaacLab-GR00T/datasets/OpenArm_CanSorting_MultiTask_Sim_dataset_O6_0403
GR=/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t

# 1) 生成 VQA：依 6 個動作階段抽幀（approach…retract），teacher 8B + template，涵蓋 7 題型
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.gen_vqa_from_lerobot \
  --dataset-path "$DS" --out-dir artifacts/vqa_6phase --num-episodes 150 \
  --teacher-model Qwen/Qwen3-VL-8B-Instruct

# 2) LoRA 微調 Cosmos-R2
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.train_vlm_lora \
  --dataset-path artifacts/vqa_6phase/data.train.jsonl --image-root artifacts/vqa_6phase \
  --output-dir artifacts/cosmos_r2_lora_6phase

# 3) 合併 → 產物 A（獨立 VLM）
uv run python -m vlm_lora.merge_lora --adapter-dir artifacts/cosmos_r2_lora_6phase \
  --out-dir "$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase"

# 4) 置換進 VLA → 產物 B（保留 action head）
uv run python -m vlm_lora.swap_backbone \
  --vla-ckpt "$GR/N1_7_fft_0614_150k_lr1e4_no_tune_visual" \
  --merged-vlm "$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase" \
  --out-dir "$GR/swapped_checkpoints/N1_7_cosmosR2lora_6phase_swapped"

# 5) 評估 VLM VQA 準確率（依題型）
uv run python -m vlm_lora.eval_vlm_vqa \
  --model-dir "$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase" \
  --val-jsonl artifacts/vqa_6phase/data.val.jsonl --image-root artifacts/vqa_6phase \
  --out-json artifacts/eval/acc_6phase_lora.json
```

> 在 H100 上改用 `python scripts/common/pegasus.py run "<cmd>"`（不是 `uv run`）；長任務用 `examples/run_pegasus.py`（detached）。

VLA 閉環評估（產物 B vs baseline 成功率，需 conda `env_isaaclab` + n1d7）走 `scripts/eval/run_eval.py`，細節見 [`docs/plans/2026-06-23-vqa-6phase-qwen3vl8b-4090.md`](docs/plans/2026-06-23-vqa-6phase-qwen3vl8b-4090.md)。

## 實測結果（4090 · 6 階段 · 8B teacher）

| 指標 | 結果 |
|---|---|
| VQA 準確率（1080 題 val） | base 20.7% → LoRA **90.1%**（Temporal 100%、Mechanics 94%、Trajectory 94%…） |
| 閉環 sim 成功率 | baseline **89%** → swapped **73%**（換 backbone 提升 VQA、但 action 略退） |
| LoRA 訓練 | r=16、2000 步、約 33 分、僅佔 ~6 GB |

## 三個產物（`artifacts/checkpoints/gr00t/` 下）

| 代號 | 路徑 | 用途 |
|---|---|---|
| 產物 A | `lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase/` | 獨立 VLM，可單獨部署、不依賴 GR00T |
| 產物 B | `swapped_checkpoints/N1_7_cosmosR2lora_6phase_swapped/` | backbone 換成 LoRA 版的 VLA |
| base | `nvidia/Cosmos-Reason2-2B`（HF 快照） | VQA 對照基準 |

## 當 AgentBot 的 Brain（tool-calling 端點）

把產物 A 起成 OpenAI 相容的 tool-calling 端點，供 `agentbot` 當「大腦」：

```bash
VLM_MODEL_DIR="$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase" \
  bash examples/run_vlm_server.sh        # 預設 :8000
```

- **端點**：`POST /v1/chat/completions`，收標準 `{messages, tools}`（支援 multimodal `image_url`）；模型輸出 `<tool_call>{"name":…,"arguments":…}</tool_call>`，server 解析成 OpenAI `tool_calls`。
- **agentbot 接線**：`config/agentbot.yaml` 設 `backend: gr00t-vlm`、`base_url: http://localhost:8000/v1`；其 `Gr00tVLMClient` 會 POST 此端點並解析成 `SkillCall`（已 e2e 驗證）。

> **範圍**：本節僅是 **serve**（把模型變成端點）。agentbot 端的部署／啟動 sim／UI 操作／監控屬 agentbot 範圍 → [`agentbot/docs/USING_VLM_BRAIN.md`](../agentbot/docs/USING_VLM_BRAIN.md)；跨 repo 交付與 4090 runbook → [`IsaacLab-GR00T/docs/DELIVERY_4090.md`](../docs/DELIVERY_4090.md)；整合計畫 → [`docs/plans/2026-06-21-vlm-brain-agentbot-integration.md`](docs/plans/2026-06-21-vlm-brain-agentbot-integration.md)。

## 文件（`docs/`）

- [`project_report.html`](docs/project_report.html) — 結果、流程動畫、VQA 原理與「7 題型 × 6 動作階段」切分、GR00T↔Cosmos-R2 抽離／置換架構圖、評估、案例分析（實際截圖）、DIY、Q&A。
- [`llm_vlm_finetuning_guide.html`](docs/llm_vlm_finetuning_guide.html) — 通用 LLM／VLM 微調概念（LoRA／MoE／多模態）＋本專案腳本逐一說明＋SFT 資料流圖。
- [`architecture_dataflow.html`](docs/architecture_dataflow.html) — 獨立 light-theme 動畫架構圖（SVG）。
- [`docs/plans/`](docs/plans/) — 實作計畫（含繁體中文 `.zh-Hant.md`）。

> 所有 commit 由維護者進行；編輯後請勿自行 `git commit` / `git push`。
