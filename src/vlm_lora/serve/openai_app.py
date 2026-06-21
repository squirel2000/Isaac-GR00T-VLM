"""Minimal OpenAI-compatible /v1/chat/completions that backs AgentBot's Gr00tVLMClient.
Launch:  VLM_MODEL_DIR=<merged_vlm> uv run uvicorn vlm_lora.serve.openai_app:app --port 8000"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="GR00T VLM (OpenAI-compatible)")
STATE: dict = {"model": None}


class ChatMessage(BaseModel):
    role: str
    content: Any = ""           # str OR list of typed parts (multimodal)


class ChatCompletionRequest(BaseModel):
    model: str = "gr00t-vlm"
    messages: list[ChatMessage]
    tools: list[dict] = Field(default_factory=list)
    tool_choice: Any = "auto"
    max_tokens: int = 512
    temperature: float = 0.0


@app.on_event("startup")
def _load():
    if STATE["model"] is not None:
        return
    from vlm_lora.serve.model import ToolCallVLM

    STATE["model"] = ToolCallVLM(os.environ["VLM_MODEL_DIR"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "gr00t-vlm", "object": "model"}]}


def _to_internal_tools(tools: list[dict]) -> list[dict]:
    """OpenAI tools [{type:function, function:{name,description,parameters}}] -> internal
    [{name, description, input_schema}]. Also accept the internal shape directly."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({"name": fn["name"], "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", fn.get("input_schema", {}))})
    return out


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [m.model_dump() for m in req.messages]
    result = STATE["model"].complete(messages, _to_internal_tools(req.tools), req.max_tokens)
    tool_calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
        for i, c in enumerate(result["calls"])
    ]
    message = {"role": "assistant", "content": result["text"] if not tool_calls else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-gr00tvlm", "object": "chat.completion", "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
    }
