"""Auto-generate a VQA dataset from an OpenArm LeRobot v2 dataset.

teacher-VLM mode (default) captions frames with a larger VLM to cover all 7 PDF question
types; template mode adds reliable Summary/Trajectory/Attribute/Temporal pairs. Outputs
<out>/{data,data.train,data.val}.jsonl, <out>/images/*.png, <out>/stats.json.

The teacher is loaded ONCE (TeacherVLM) and reused across frames; rows are streamed to
data.jsonl as they are produced so a long/interrupted run keeps its progress.
"""

import json
import os
import re
from collections import Counter

import tyro

_COLORS = ["orange", "green", "red", "blue", "yellow", "white", "black"]
_TYPES = ["Attribute", "Mechanics", "Reasoning", "Spatial", "Summary", "Temporal", "Trajectory"]


def parse_target_color(task: str) -> str | None:
    for c in _COLORS:
        if re.search(rf"\b{c}\b", task.lower()):
            return c
    return None


def template_qa_pairs(task: str, phase: str) -> list[dict]:
    color = parse_target_color(task) or "target"
    done = phase == "late"
    return [
        {
            "type": "Summary",
            "question": "What is the dual-arm robot doing in this scene?",
            "answer": f"The robot is performing a can-sorting task: {task}.",
        },
        {
            "type": "Trajectory",
            "question": "On which plate will the can be placed?",
            "answer": f"On the {color} plate.",
        },
        {
            "type": "Attribute",
            "question": "What is the color of the target plate?",
            "answer": f"The target plate is {color}.",
        },
        {
            "type": "Temporal",
            "question": "Has the can already been placed on the plate?",
            "answer": "Yes, the can has been placed."
            if done
            else "No, the robot is still moving the can.",
        },
    ]


def compute_stats(rows: list[dict]) -> dict:
    by_type = Counter(r.get("type", "Unknown") for r in rows)
    return {
        "total": len(rows),
        "by_type": dict(by_type),
        "images": len({img for r in rows for img in r.get("images", [])}),
    }


def _sample_frames(length: int, per_episode: int) -> list[tuple[int, str]]:
    if per_episode <= 1:
        return [(length // 2, "mid")]
    out = []
    for i in range(per_episode):
        frac = i / (per_episode - 1)
        idx = min(length - 1, int(frac * (length - 1)))
        out.append((idx, "early" if frac < 0.34 else ("mid" if frac < 0.67 else "late")))
    return out


def _extract_frame(video_path: str, frame_index: int):
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = cap.read()
    cap.release()
    return Image.fromarray(np.asarray(bgr)[:, :, ::-1]) if ok else None


_TEACHER_PROMPT = (
    "This is a robot manipulation scene. Task: '{task}'. Generate 4 diverse question/answer "
    "pairs about this image covering varied types among: {types}. Keep answers short and "
    "grounded in what is visible. Reply ONLY as a JSON list of objects with keys "
    "type, question, answer."
)


class TeacherVLM:
    """A larger VLM loaded once and reused to caption frames into VQA pairs.

    Works for dense (Qwen3-VL-*-Instruct) and MoE (Qwen3-VL-30B-A3B-Instruct) checkpoints —
    AutoModelForImageTextToText maps the config to the right class (dense vs ...Moe...).
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from vlm_lora.hf_utils import resolve_model_path

        mid = resolve_model_path(model_id)
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}.get(dtype, dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            mid, trust_remote_code=True, device_map="auto", dtype=td
        ).eval()

    def generate_qa(self, image, task: str, max_new_tokens: int = 512) -> list[dict]:
        import torch

        prompt = _TEACHER_PROMPT.format(task=task, types=", ".join(_TYPES))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        dec = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        try:
            s, e = dec.index("["), dec.rindex("]") + 1
            return [qa for qa in json.loads(dec[s:e]) if {"type", "question", "answer"} <= set(qa)]
        except Exception:
            return []


def generate(
    dataset_path: str,
    out_dir: str,
    num_episodes: int = 100,
    frames_per_episode: int = 3,
    val_ratio: float = 0.1,
    teacher_model: str | None = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    teacher_dtype: str = "bfloat16",
    use_template: bool = True,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> None:
    import random

    random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    eps = [
        json.loads(x)
        for x in open(os.path.join(dataset_path, "meta/episodes.jsonl"), encoding="utf-8")
        if x.strip()
    ]
    random.shuffle(eps)
    eps = eps[:num_episodes]
    vpat = os.path.join(
        dataset_path, "videos/chunk-000/observation.images.camera/episode_{:06d}.mp4"
    )

    teacher = TeacherVLM(teacher_model, teacher_dtype) if teacher_model else None
    if teacher:
        print(f"[gen_vqa] teacher loaded: {teacher_model}", flush=True)

    data_path = os.path.join(out_dir, "data.jsonl")
    n = 0
    with open(data_path, "w", encoding="utf-8") as out:
        for i, ep in enumerate(eps):
            ei, task, length = ep["episode_index"], ep["task"], ep["length"]
            if not os.path.exists(vpat.format(ei)):
                continue
            for fidx, phase in _sample_frames(length, frames_per_episode):
                img = _extract_frame(vpat.format(ei), fidx)
                if img is None:
                    continue
                rel = f"images/ep{ei:06d}_f{fidx:06d}.png"
                img.save(os.path.join(out_dir, rel))
                qas = template_qa_pairs(task, phase) if use_template else []
                if teacher:
                    qas += teacher.generate_qa(img, task, max_new_tokens)
                for qa in qas:
                    out.write(
                        json.dumps(
                            {
                                "images": [rel],
                                "type": qa["type"],
                                "messages": [
                                    {"role": "user", "content": f"<image>\n{qa['question']}"},
                                    {"role": "assistant", "content": qa["answer"]},
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    out.flush()
                    n += 1
            if (i + 1) % 10 == 0:
                print(f"[gen_vqa] {i + 1}/{len(eps)} episodes, {n} QA pairs so far", flush=True)

    rows = [json.loads(x) for x in open(data_path, encoding="utf-8") if x.strip()]
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    for name, sub in {"data.val.jsonl": rows[:n_val], "data.train.jsonl": rows[n_val:]}.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(compute_stats(rows), open(os.path.join(out_dir, "stats.json"), "w"), indent=2)
    print(
        f"[gen_vqa] DONE {len(rows)} pairs ({len(rows) - n_val} train/{n_val} val) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    tyro.cli(generate)
