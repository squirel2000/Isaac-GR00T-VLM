from vlm_lora.eval_toolcall import score

def test_exact_match():
    s = score('<tool_call>{"name":"sort_can","arguments":{"target_color":"orange"}}</tool_call>',
              [{"name": "sort_can", "arguments": {"target_color": "orange"}}], allowed={"sort_can"})
    assert s["valid"] == 1 and s["name_ok"] == 1 and s["args_ok"] == 1

def test_wrong_arg():
    s = score('<tool_call>{"name":"sort_can","arguments":{"target_color":"green"}}</tool_call>',
              [{"name": "sort_can", "arguments": {"target_color": "orange"}}], allowed={"sort_can"})
    assert s["name_ok"] == 1 and s["args_ok"] == 0
