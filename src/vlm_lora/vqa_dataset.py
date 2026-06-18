"""VQA JSONL dataset + collator for Cosmos-Reason2-2B (Qwen3-VL) LoRA SFT."""

import json
import os

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def mask_prompt_labels(input_ids, prompt_len, ignore_index=IGNORE_INDEX):
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = ignore_index
    return labels


class VqaJsonlDataset(Dataset):
    def __init__(self, dataset_path, image_root=None):
        self.records = []
        for path in (p for p in dataset_path.split(os.pathsep) if p):
            with open(path, encoding="utf-8") as f:
                self.records += [json.loads(ln) for ln in f if ln.strip()]
        self.image_root = image_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        imgs = rec.get("images", []) or []
        if self.image_root:
            imgs = [im if os.path.isabs(im) else os.path.join(self.image_root, im) for im in imgs]
        return {"messages": rec["messages"], "image_paths": imgs}


class VqaCollator:
    def __init__(self, processor, max_seq_len=2048):
        self.processor = processor
        self.max_seq_len = max_seq_len

    @staticmethod
    def to_template_messages(messages):
        """Convert user turns whose content is a '<image>...' string into the structured
        content the Qwen3-VL chat template needs (so image placeholder tokens get emitted).
        Each '<image>' marker becomes one {"type": "image"} part."""
        out = []
        for m in messages:
            c = m.get("content")
            if m.get("role") == "user" and isinstance(c, str) and "<image>" in c:
                n = c.count("<image>")
                text = c.replace("<image>", "").strip()
                parts = [{"type": "image"} for _ in range(n)]
                if text:
                    parts.append({"type": "text", "text": text})
                out.append({"role": "user", "content": parts})
            else:
                out.append(m)
        return out

    def _imgs(self, paths):
        from PIL import Image

        return [Image.open(p).convert("RGB") for p in paths]

    def __call__(self, batch):
        labels_list = []
        for ex in batch:
            imgs = self._imgs(ex["image_paths"])
            msgs = self.to_template_messages(ex["messages"])
            prompt = self.processor.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True
            )
            full = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            plen = len(
                self.processor(text=[prompt], images=imgs or None, return_tensors="pt")["input_ids"][0]
            )
            fids = (
                self.processor(text=[full], images=imgs or None, return_tensors="pt")["input_ids"][0][
                    : self.max_seq_len
                ].tolist()
            )
            labels_list.append(torch.tensor(mask_prompt_labels(fids, plen), dtype=torch.long))
        texts = [
            self.processor.apply_chat_template(
                self.to_template_messages(ex["messages"]), tokenize=False, add_generation_prompt=False
            )
            for ex in batch
        ]
        all_imgs = [im for ex in batch for im in self._imgs(ex["image_paths"])]
        enc = self.processor(
            text=texts,
            images=all_imgs or None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        labels = torch.full_like(enc["input_ids"], IGNORE_INDEX)
        for i, lab in enumerate(labels_list):
            n = min(lab.shape[0], labels.shape[1])
            labels[i, :n] = lab[:n]
        enc["labels"] = labels
        return dict(enc)
