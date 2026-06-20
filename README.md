# Isaac-GR00T-VLM

獨立 repo：對 **Cosmos-Reason2-2B**（GR00T N1.7 的 VLM backbone）做 **LoRA 微調**，合併成可獨立部署的 standalone VLM，再把權重**置換進 GR00T N1.7 VLA**，並做兩方對照評估（對齊附件 PDF 的「可置換權重」流程）。

## 用途、角色與重要性
- **用途**：把 GR00T 內的 VLM（Cosmos-Reason2-2B）抽離、用領域 VQA 做 LoRA 微調、再換回 VLA，量化「VLM 理解力 → action 表現」的影響。
- **在微調流程中的角色**：所有「VLM 端」的工作（資料生成、LoRA 訓練、合併、置換、評估）都封裝在此 repo，**與 `Isaac-GR00T_n1d7` 完全解耦**——不修改 NVIDIA 原始碼，只在評估時唯讀呼叫其 `open_loop_eval.py`。
- **重要性**：產物 A（standalone VLM）不依賴 GR00T、可獨立部署/分享；流程可重現、可換 teacher / 換資料集擴充。

## 資料夾結構（架構）
```
Isaac-GR00T-VLM/
├── src/vlm_lora/                 # 核心套件
│   ├── lora_args.py              #   VlmLoraConfig + PEFT LoraConfig builder
│   ├── gen_vqa_from_lerobot.py   #   從 OpenArm 影格生成 VQA（teacher-VLM + template）
│   ├── vqa_dataset.py            #   VQA dataset + collator（label masking、image-token）
│   ├── train_vlm_lora.py         #   LoRA SFT launcher（PEFT + HF Trainer）
│   ├── merge_lora.py             #   合併 adapter → 產物 A（standalone VLM）
│   ├── swap_backbone.py          #   把產物 A 換進 VLA → 產物 B（swapped VLA）
│   ├── eval_vlm_vqa.py           #   VLM VQA 準確率（按題型）
│   ├── infer_vlm.py              #   standalone VLM 推論範例
│   └── hf_utils.py               #   gated 模型離線載入（cache snapshot 解析）
├── examples/
│   ├── finetune_cosmos_r2_lora.sh  # env 驅動 H100 launcher
│   ├── run_pegasus.py              # detached 遠端啟動（斷線可續）
│   └── sample_vqa/                 # 合成 smoke 樣本
├── configs/default.yaml          # 路徑 + 超參數（單一來源）
├── tests/                        # CPU 單元測試（lora_args / vqa_dataset / gen_vqa / swap）
├── docs/
│   ├── project_report.html       # 本專案報告（動畫、結果、產物路徑）
│   ├── llm_vlm_finetuning_guide.html # 通用 LLM/VLM 微調教學（LoRA/MoE/多模態，動畫）
│   ├── architecture_dataflow.html    # 獨立 light-theme 動畫架構圖（SVG）
│   └── vlm_lora_finetune.md      # operator 指南 + 踩坑
└── artifacts/                    # （gitignored）vqa/、cosmos_r2_lora/、Cosmos-Reason2-2B-base/、eval/
```

## 產物總表（H100，IsaacLab-GR00T 根目錄下）
| 代號 | 路徑 | 用途 |
|---|---|---|
| **產物 A** | `artifacts/checkpoints/gr00t/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged/` | standalone VLM，純 transformers 可載入、不依賴 GR00T；swap 來源 |
| **產物 B** | `artifacts/checkpoints/gr00t/swapped_checkpoints/N1_7_cosmosR2lora_swapped/` | VLM 換成 LoRA 版的 VLA；action 評估 |
| base | `Isaac-GR00T-VLM/artifacts/Cosmos-Reason2-2B-base/` | 微調前 VLM 對照 |

## 流程（H100 + uv）
```bash
uv sync --extra dev
# 1) VQA 生成 → 2) LoRA 訓練 → 3) merge(產物A) → 4) swap(產物B) → 5) 評估(VLM VQA + VLA MSE/MAE)
```
詳見 [docs/vlm_lora_finetune.md](docs/vlm_lora_finetune.md) 與 [docs/project_report.html](docs/project_report.html)。
