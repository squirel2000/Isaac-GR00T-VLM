"""Standalone inference for the merged VLM (Product A). No GR00T dependency."""

import torch
import tyro
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def ask(model_dir: str, image: str, question: str, max_new_tokens: int = 128) -> str:
    from PIL import Image

    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[Image.open(image).convert("RGB")], return_tensors="pt").to(
        model.device
    )
    out = model.generate(**inp, max_new_tokens=max_new_tokens)
    ans = proc.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    print(ans)
    return ans


if __name__ == "__main__":
    tyro.cli(ask)
