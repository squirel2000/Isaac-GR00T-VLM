from vlm_lora.serve.toolcall import build_tool_system, split_text_and_images, parse_tool_calls

TOOLS = [{"name": "sort_can", "description": "place can on a colored plate",
          "input_schema": {"type": "object",
                           "properties": {"target_color": {"enum": ["orange", "green"]}},
                           "required": ["target_color"]}}]

def test_system_lists_tools_and_format():
    sys = build_tool_system(TOOLS)
    assert "sort_can" in sys and "target_color" in sys and "<tool_call>" in sys

def test_parse_single_tool_call():
    txt = 'ok\n<tool_call>{"name": "sort_can", "arguments": {"target_color": "orange"}}</tool_call>'
    calls = parse_tool_calls(txt)
    assert calls == [{"name": "sort_can", "arguments": {"target_color": "orange"}}]

def test_parse_multiple_and_ignores_prose():
    txt = ('<tool_call>{"name":"pick","arguments":{"object":"can"}}</tool_call> then '
           '<tool_call>{"name":"place","arguments":{"target":"orange plate"}}</tool_call>')
    assert [c["name"] for c in parse_tool_calls(txt)] == ["pick", "place"]

def test_parse_none_when_absent():
    assert parse_tool_calls("I cannot do that.") == []

def test_split_text_and_images_handles_multimodal():
    msg = {"role": "user", "content": [
        {"type": "text", "text": "sort it"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}]}
    text, imgs = split_text_and_images(msg)
    assert text == "sort it" and imgs == ["data:image/png;base64,aGk="]
