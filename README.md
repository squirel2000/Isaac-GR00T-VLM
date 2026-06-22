# Isaac-GR00T-VLM

LoRA fine-tune the **Cosmos-Reason2-2B** VLM (GR00T N1.7 的 backbone) on auto-generated VQA, merge into a **standalone VLM**, then **swap it back into the GR00T N1.7 VLA**. 與 `Isaac-GR00T_n1d7` 完全解耦（不修改其原始碼）。

## Quick start (H100 + uv)
```bash
cd Isaac-GR00T-VLM
uv sync --extra dev                 # build env (reuses n1d7's cu128 wheels)
# 所有設定都在 configs/default.yaml（已附註解）；以下 5 步即完整流程：

# 1) 生成 VQA（從 LeRobot 影片；teacher-VLM + template）
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.gen_vqa_from_lerobot \
  --dataset-path <LEROBOT_DATASET> --out-dir artifacts/vqa --num-episodes 100
# 2) LoRA 微調
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python -m vlm_lora.train_vlm_lora \
  --dataset-path artifacts/vqa/data.train.jsonl --image-root artifacts/vqa --output-dir artifacts/cosmos_r2_lora
# 3) 合併 → 產物 A（standalone VLM）
uv run python -m vlm_lora.merge_lora --adapter-dir artifacts/cosmos_r2_lora \
  --out-dir <ISAACLAB>/artifacts/checkpoints/gr00t/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged
# 4) 置換進 VLA → 產物 B（swapped VLA）
uv run python -m vlm_lora.swap_backbone --vla-ckpt <BASELINE_VLA> \
  --merged-vlm <...lora_tuned_vlm/...> --out-dir <ISAACLAB>/artifacts/checkpoints/gr00t/swapped_checkpoints/...
# 5) 評估：VLM VQA（本 repo env）+ VLA open-loop（n1d7 env，MPLBACKEND=Agg）
uv run python -m vlm_lora.eval_vlm_vqa --model-dir <A> --val-jsonl artifacts/vqa/data.val.jsonl --image-root artifacts/vqa --out-json acc.json
```
本機連 H100：`python scripts/common/pegasus.py run "<cmd>"`（不是 `uv run`）。長任務用 `examples/run_pegasus.py`（detached）。

## 設定
**所有可調設定集中在 [`configs/default.yaml`](configs/default.yaml)**（路徑、LoRA 超參、訓練、產物位置），檔內每段都有用途註解。修改該檔即可規劃微調/評估；CLI flags 與 `examples/finetune_cosmos_r2_lora.sh` 的環境變數對應同名設定並可覆蓋。

## Serve as AgentBot's Brain

Product A (the merged standalone VLM) can be served as an **OpenAI-compatible tool-calling endpoint** and consumed by the `agentbot` orchestration system as its "Brain".

**Launch the server**
```bash
VLM_MODEL_DIR=<path-to-Cosmos-Reason2-2B-lora-merged> \
  uv run uvicorn vlm_lora.serve.openai_app:app --host 0.0.0.0 --port 8000
```
Or use the convenience wrapper (sets `VLM_MODEL_DIR` from `configs/default.yaml` and passes standard uvicorn flags):
```bash
bash examples/run_vlm_server.sh
```

**I/O contract — OpenAI `/v1/chat/completions` with tool-calling**

- **Request**: standard `{model, messages, tools, tool_choice}` JSON (multimodal messages accepted — `image_url` content parts plumbed through).
- **Response**: the model emits `<tool_call>{"name":"…","arguments":{…}}</tool_call>` tokens; the server parses them into the standard OpenAI `tool_calls` array in the response `choices[0].message`.
- The served model understands agentbot's skill schema (`sort_can`, `pick`, `place`, `home`, `pour_water`) and returns a `SkillCall`-compatible argument payload.

**AgentBot configuration**

In `agentbot/configs/` set:
```yaml
vlm:
  backend: gr00t-vlm
  base_url: http://<server-host>:8000
```
`Gr00tVLMClient.complete()` will POST to that endpoint and parse `tool_calls` → `SkillCall`.  Camera frames are plumbed into the Brain via a `CAMERA` event → Monitor `state["camera"]` → multimodal message payload automatically.

詳細整合計畫：[`docs/plans/2026-06-21-vlm-brain-agentbot-integration.md`](docs/plans/2026-06-21-vlm-brain-agentbot-integration.md)。

> **Note**: All commits are the maintainer's; do not `git commit` / `git push` after editing.

## 文件（docs/）
- **`project_report.html`** — 結果、流程（動畫）、VQA 生成原理+範例、GR00T↔Cosmos-R2 抽離/整合、評估與 MSE/MAE、案例分析（實際圖像）、DIY、參數 Q&A、附錄。
- **`llm_vlm_finetuning_guide.html`** — 通用 LLM/VLM 微調概念（LoRA / MoE / 多模態，含動畫）。
- **`architecture_dataflow.html`** — 獨立 light-theme 動畫架構圖（SVG）。

## 產物（H100，`IsaacLab-GR00T/artifacts/checkpoints/gr00t/`）
| 代號 | 路徑 | 用途 |
|---|---|---|
| 產物 A | `lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged/` | standalone VLM（可獨立部署） |
| 產物 B | `swapped_checkpoints/N1_7_cosmosR2lora_swapped/` | VLM 換成 LoRA 版的 VLA |
