"""Evaluate VQA accuracy of a VLM checkpoint by question type (PDF's VLM table)."""

import json
import os
from collections import defaultdict

import torch
import tyro
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()


def _match(pred, gold):
    p, g = _norm(pred), _norm(gold)
    return g in p or p in g


def evaluate(
    model_dir: str, val_jsonl: str, image_root: str, out_json: str, max_new_tokens: int = 64
) -> None:
    from PIL import Image

    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    rows = [json.loads(x) for x in open(val_jsonl, encoding="utf-8") if x.strip()]
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        q = r["messages"][0]["content"].replace("<image>", "").strip()
        gold = r["messages"][-1]["content"]
        imgs = [Image.open(os.path.join(image_root, p)).convert("RGB") for p in r.get("images", [])]
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=imgs or None, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=max_new_tokens)
        pred = proc.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        t = r.get("type", "Overall")
        by[t][1] += 1
        by[t][0] += int(_match(pred, gold))
    table = {t: {"correct": c, "total": n, "acc": c / n if n else 0.0} for t, (c, n) in by.items()}
    tc, tn = sum(c for c, _ in by.values()), sum(n for _, n in by.values())
    table["Overall"] = {"correct": tc, "total": tn, "acc": tc / tn if tn else 0.0}
    json.dump(table, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    tyro.cli(evaluate)
