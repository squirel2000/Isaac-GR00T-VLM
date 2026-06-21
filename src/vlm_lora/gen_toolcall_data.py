"""Generate tool-calling training data (instruction+image -> <tool_call> sequence) over
AgentBot's real skill schemas, from OpenArm LeRobot frames. Output JSONL matches vqa_dataset's
`messages` schema so train_vlm_lora trains on it unchanged."""
from __future__ import annotations

import json
import os
import re

import tyro

_COLORS = ["orange", "green"]


def template_calls(task: str) -> list[dict]:
    """OpenArm 0403 is the can-sorting task -> a single sort_can call with the plate color."""
    for c in _COLORS:
        if re.search(rf"\b{c}\b", task.lower()):
            return [{"name": "sort_can", "arguments": {"target_color": c}}]
    return [{"name": "sort_can", "arguments": {"target_color": "orange"}}]


def _render_calls(calls: list[dict]) -> str:
    return "\n".join(f'<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>' for c in calls)


def to_training_row(image_rel: str, instruction: str, calls: list[dict]) -> dict:
    return {"images": [image_rel], "type": "ToolCall", "messages": [
        {"role": "user", "content": f"<image>\n{instruction}"},
        {"role": "assistant", "content": _render_calls(calls)}]}


def generate(dataset_path: str, out_dir: str, num_episodes: int = 100, frames_per_episode: int = 2,
             val_ratio: float = 0.1, seed: int = 42) -> None:
    import random

    from vlm_lora.gen_vqa_from_lerobot import _extract_frame, _sample_frames

    random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    eps = [json.loads(x) for x in open(os.path.join(dataset_path, "meta/episodes.jsonl"),
                                       encoding="utf-8") if x.strip()]
    random.shuffle(eps)
    eps = eps[:num_episodes]
    vpat = os.path.join(dataset_path, "videos/chunk-000/observation.images.camera/episode_{:06d}.mp4")
    rows = []
    for ep in eps:
        ei, task, length = ep["episode_index"], ep["task"], ep["length"]
        if not os.path.exists(vpat.format(ei)):
            continue
        for fidx, _phase in _sample_frames(length, frames_per_episode):
            img = _extract_frame(vpat.format(ei), fidx)
            if img is None:
                continue
            rel = f"images/ep{ei:06d}_f{fidx:06d}.png"
            img.save(os.path.join(out_dir, rel))
            rows.append(to_training_row(rel, task, template_calls(task)))
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    for name, sub in {"data.val.jsonl": rows[:n_val], "data.train.jsonl": rows[n_val:],
                      "data.jsonl": rows}.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_toolcall] {len(rows)} rows ({len(rows) - n_val} train / {n_val} val) -> {out_dir}")


if __name__ == "__main__":
    tyro.cli(generate)
