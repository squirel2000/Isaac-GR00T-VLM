# Unit C Report — Task 8: Document vlm_lora scripts in finetuning guide

**Status: DONE**

## What was added

### §6 · 本專案腳本逐一說明（fine-tune / eval pipeline）
Inserted immediately before the closing footer `<p class="mut" ...>` (was line 158), using only existing CSS classes (`.card`, `.note`, `table`, `th`, `td`, `code`, `h2`).

**Table contents (10 scripts, one row each):**

| Script | Verified against source |
|---|---|
| `gen_vqa_from_lerobot.py` | `AutoModelForImageTextToText`, `AutoProcessor`, `cv2.VideoCapture`; 6-phase sampling confirmed |
| `vqa_dataset.py` | `processor.apply_chat_template`, `IGNORE_INDEX=-100`, `<image>` → structured content |
| `lora_args.py` | `peft.LoraConfig(task_type="CAUSAL_LM")`, q/k/v/o/gate/up/down proj |
| `train_vlm_lora.py` | `Qwen3VLForConditionalGeneration`, `Trainer`, `TrainingArguments`, `get_peft_model` |
| `merge_lora.py` | `PeftModel.merge_and_unload()` |
| `swap_backbone.py` | `safetensors.torch.load_file/save_file`, key-intersection swap |
| `eval_vlm_vqa.py` | `model.generate`, `processor.batch_decode`, normalized substring match |
| `eval_toolcall.py` | `parse_tool_calls`, `ToolCallVLM.complete`, 4-metric scoring |
| `infer_vlm.py` | `Qwen3VLForConditionalGeneration`, single-image CLI inference |
| `serve/model.py` | `ToolCallVLM` class, `build_tool_system`, `parse_tool_calls`; FastAPI in `serve/app.py` |

**Brief correction:** `serve/model.py` does not directly import FastAPI — the FastAPI layer lives in `serve/app.py`. The description was updated to reflect this accurately.

**`.note` box added:** explains that `train_vlm_lora.py` is a standard HF SFT pipeline bound to Qwen3-VL classes, and contrasts it with GR00T N1.7 training (GR00T trainer, full VLA, flow-matching loss, not HF Trainer / CausalLM loss).

### §3 stale stats updated
- **Before:** `2,499 對（2,250 train / 249 val），300 影格` / `Qwen3-VL-30B-A3B`
- **After:** `依 episode 數而定；每集 6 幀（6 動作階段 × 1 幀）` / `Qwen3-VL-8B-Instruct（dense，4090 可跑）；6 個動作階段：approach / grasp / pick / move_to_plate / place / retract`

## Verification

```
grep -c "本專案腳本逐一說明\|swap_backbone\|merge_and_unload" docs/llm_vlm_finetuning_guide.html
→ 4   (≥ 3 ✓)
```

HTML tag balance check (Python `html.parser`): **PASS** — remaining open tags: `[]`

## Staging

`git add docs/llm_vlm_finetuning_guide.html` — **only that file was explicitly staged in this task**. (Other files already staged from prior work remain staged; no new files were added.)

## Concerns

None. All script descriptions verified directly against source code. The one brief discrepancy (FastAPI in `serve/model.py`) was corrected.
