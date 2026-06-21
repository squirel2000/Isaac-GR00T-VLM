from vlm_lora.gen_toolcall_data import template_calls, to_training_row

def test_template_calls_sort_can():
    calls = template_calls("place the can on the orange plate")
    assert calls == [{"name": "sort_can", "arguments": {"target_color": "orange"}}]

def test_to_training_row_emits_tool_call_tags():
    row = to_training_row("img.png", "put can on green plate",
                          [{"name": "sort_can", "arguments": {"target_color": "green"}}])
    assert row["images"] == ["img.png"]
    a = row["messages"][1]["content"]
    assert a.startswith("<tool_call>") and '"sort_can"' in a and a.rstrip().endswith("</tool_call>")
