from fastapi.testclient import TestClient
from vlm_lora.serve import openai_app as A

class _Stub:
    def complete(self, messages, tools, max_new_tokens=512):
        return {"text": "ok", "calls": [{"name": "sort_can", "arguments": {"target_color": "orange"}}]}

def _client():
    A.STATE["model"] = _Stub()
    return TestClient(A.app)

def test_health():
    assert _client().get("/health").json()["status"] == "ok"

def test_chat_completions_returns_tool_calls():
    body = {"model": "gr00t-vlm", "messages": [{"role": "user", "content": "sort the can onto orange"}],
            "tools": [{"type": "function", "function": {"name": "sort_can", "description": "x",
                       "parameters": {"type": "object", "properties": {}, "required": []}}}]}
    r = _client().post("/v1/chat/completions", json=body).json()
    tc = r["choices"][0]["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "sort_can"
    assert '"target_color": "orange"' in tc[0]["function"]["arguments"]
