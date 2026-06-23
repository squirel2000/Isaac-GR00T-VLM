### Task 8: Document the `vlm_lora` scripts in the finetuning guide (user Task 3)

**Files:** Modify: `docs/llm_vlm_finetuning_guide.html` (append a new `<h2>` section before the closing footer `<p>` on line ~158; reuse existing CSS classes `.card .grid .g2 table .note code .tag`).

- [ ] **Step 1: Add "§6 · 本專案腳本逐一說明（fine-tune / eval pipeline）"** — a table mapping each script to its role + the exact transformers/peft API it uses, plus the "是否為 transformers 標準流程 / 與 GR00T N1.7 異同" answer. Content to include (one row per script):

| script | 作用 | 關鍵 API |
|---|---|---|
| `gen_vqa_from_lerobot.py` | 從 LeRobot 影片抽 6 動作階段幀 → teacher(8B)+template 生成 VQA JSONL | `AutoModelForImageTextToText`, `AutoProcessor`, `cv2` |
| `vqa_dataset.py` | JSONL→batch；`<image>`→結構化 content；label-mask 只算答案 | `processor.apply_chat_template`, `IGNORE_INDEX=-100` |
| `lora_args.py` | LoRA 超參 + target 模組（`llm` = q/k/v/o/gate/up/down） | `peft.LoraConfig(task_type="CAUSAL_LM")` |
| `train_vlm_lora.py` | 標準 HF SFT：載入→`get_peft_model`→`Trainer.train` | `Qwen3VLForConditionalGeneration`, `Trainer`, `TrainingArguments` |
| `merge_lora.py` | adapter 併回底座 → Product A | `PeftModel.merge_and_unload` |
| `swap_backbone.py` | 把 Product A 的 `backbone.model.*` 塞回 VLA | 純 `safetensors`，key 交集 |
| `eval_vlm_vqa.py` / `eval_toolcall.py` | 依題型算 VQA 準確率 / tool-call 正確率 | `model.generate` + 正規化比對 |
| `infer_vlm.py`, `serve/*` | 單張推論 / OpenAI 相容 tool-calling 端點 | FastAPI, `Qwen3VLForConditionalGeneration` |

  Add a `.note`: **「`train_vlm_lora.py` 是 transformers 標準 SFT 流程，但綁定 Qwen3-VL 類別與其 proj 名稱；換 VLM 家族需改 `AutoModelForImageTextToText` 子類與 LoRA target。與 GR00T N1.7 訓練不同 —— 後者用 GR00T 自家 trainer 訓練整個 VLA（backbone+adapter+DiT、flow-matching loss），非 HF `Trainer`、非 CausalLM loss。」**
  Also refresh the §3 stats line that still says "3 幀 / 30B-A3B teacher / 2,499 對" to reflect the new **6-phase / 8B** pipeline.

- [ ] **Step 2: Verify** — open in a browser (or `uv run python -c "import pathlib,html.parser"` sanity) and grep:

Run: `grep -c "本專案腳本逐一說明\|swap_backbone\|merge_and_unload" docs/llm_vlm_finetuning_guide.html`
Expected: ≥ 3.

- [ ] **Step 3: Stage** — `git add docs/llm_vlm_finetuning_guide.html` (suggested: `docs(guide): document vlm_lora fine-tune/eval scripts`)

---

