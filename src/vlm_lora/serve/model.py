"""Load the merged VLM once and generate tool-calling text for a chat request.
Decodes OpenAI multimodal image_url (data-uri or path) into PIL images for the processor."""
from __future__ import annotations

import base64
import io

from vlm_lora.serve.toolcall import build_tool_system, parse_tool_calls, split_text_and_images


def _load_image(url: str):
    from PIL import Image

    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return Image.open(url).convert("RGB")


class ToolCallVLM:
    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from vlm_lora.hf_utils import resolve_model_path

        md = resolve_model_path(model_dir)
        self.processor = AutoProcessor.from_pretrained(md, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            md, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )

    def generate(self, messages: list[dict], tools: list[dict], max_new_tokens: int = 512) -> str:
        # 1) prepend a tool-describing system message
        chat = [{"role": "system", "content": build_tool_system(tools)}]
        images = []
        for m in messages:
            text, imgs = split_text_and_images(m)
            images += imgs
            if imgs:  # keep an <image> placeholder so the processor aligns image tokens
                chat.append({"role": m["role"],
                             "content": [{"type": "image"}, {"type": "text", "text": text}]})
            else:
                chat.append({"role": m["role"], "content": text})
        pil = [_load_image(u) for u in images] or None
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[prompt], images=pil, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

    def complete(self, messages: list[dict], tools: list[dict], max_new_tokens: int = 512) -> dict:
        """Return (raw_text, parsed_calls)."""
        raw = self.generate(messages, tools, max_new_tokens)
        return {"text": raw, "calls": parse_tool_calls(raw)}
