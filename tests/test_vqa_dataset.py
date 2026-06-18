import json

from vlm_lora.vqa_dataset import VqaCollator, VqaJsonlDataset, mask_prompt_labels


def test_to_template_messages_converts_image_string():
    msgs = [
        {"role": "user", "content": "<image>\nWhat color?"},
        {"role": "assistant", "content": "Blue."},
    ]
    out = VqaCollator.to_template_messages(msgs)
    assert out[0]["content"] == [{"type": "image"}, {"type": "text", "text": "What color?"}]
    assert out[1] == {"role": "assistant", "content": "Blue."}


def test_reads_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps(
            {
                "images": ["a.png"],
                "messages": [
                    {"role": "user", "content": "<image>\nQ?"},
                    {"role": "assistant", "content": "A."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ds = VqaJsonlDataset(str(p), image_root=str(tmp_path))
    assert len(ds) == 1 and ds[0]["image_paths"] == [str(tmp_path / "a.png")]


def test_mask():
    assert mask_prompt_labels([10, 11, 12, 20, 21], 3) == [-100, -100, -100, 20, 21]
