"""Score tool-calling output vs gold: valid-parse, skills-in-vocab, name-sequence, args-exact.
evaluate() runs ToolCallVLM over a val set (GPU; mirrors eval_vlm_vqa.py)."""
from __future__ import annotations

import json
import os

import tyro

from vlm_lora.serve.toolcall import parse_tool_calls


def score(raw: str, gold_calls: list[dict], allowed: set) -> dict:
    pred = parse_tool_calls(raw)
    names = [c["name"] for c in pred]
    gold_names = [g["name"] for g in gold_calls]
    pred_pairs = [(c["name"], c.get("arguments", {})) for c in pred]
    gold_pairs = [(g["name"], g.get("arguments", {})) for g in gold_calls]
    return {
        "valid": 1 if pred else 0,
        "skills_in_vocab": 1 if pred and all(n in allowed for n in names) else 0,
        "name_ok": 1 if names == gold_names else 0,
        "args_ok": 1 if pred_pairs == gold_pairs else 0,
    }


def evaluate(model_dir: str, val_jsonl: str, image_root: str, skills_json: str,
             out_json: str, max_new_tokens: int = 512) -> None:
    from PIL import Image

    from vlm_lora.serve.model import ToolCallVLM

    tools = json.load(open(skills_json, encoding="utf-8"))
    allowed = {t["name"] for t in tools}
    vlm = ToolCallVLM(model_dir)
    rows = [json.loads(x) for x in open(val_jsonl, encoding="utf-8") if x.strip()]
    agg = {"valid": 0, "skills_in_vocab": 0, "name_ok": 0, "args_ok": 0}
    for r in rows:
        instruction = r["messages"][0]["content"].replace("<image>", "").strip()
        gold = parse_tool_calls(r["messages"][1]["content"])
        content = [{"type": "text", "text": instruction}]
        for rel in r.get("images", []):
            p = rel if os.path.isabs(rel) else os.path.join(image_root, rel)
            with open(p, "rb") as fh:
                import base64
                b64 = base64.b64encode(fh.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        out = vlm.complete([{"role": "user", "content": content}], tools, max_new_tokens)
        s = score(out["text"], gold, allowed)
        for k in agg:
            agg[k] += s[k]
    n = len(rows) or 1
    result = {k: agg[k] / n for k in agg}
    result["n"] = len(rows)
    json.dump(result, open(out_json, "w", encoding="utf-8"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    tyro.cli(evaluate)
