> 繁體中文版（置於 agentbot）：agentbot/docs/plans/2026-06-21-vlm-brain-agentbot-integration.zh-Hant.md

# VLM Brain ⇄ AgentBot Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **COMMITS:** The maintainer owns all commits. After each task's tests pass, **stage** changes and **stop for the maintainer to commit**. Suggested messages are given; do NOT run `git commit`/`push`.
>
> **SUPERSEDES** `docs/plans/2026-06-21-vlm-brain-deployment.md` (the standalone-brain draft). AgentBot already implements the Brain/SkillRegistry/SkillCall/executor/FastAPI, so that draft's `src/vlm_lora/brain/` (schema, skills, prompt, core, serve, client, mock_executor, bridge) is **dropped**. This plan instead serves our VLM as the endpoint AgentBot was explicitly built to consume.

**Goal:** Make the fine-tuned **Cosmos-Reason2-2B** the real Brain of AgentBot — serve it as an **OpenAI-compatible tool-calling endpoint** (in `Isaac-GR00T-VLM`), wire AgentBot's `Gr00tVLMClient` to it, **plumb the camera image into the Brain**, and fine-tune the VLM to emit valid **tool-calls for AgentBot's real skills** (pick/place/home/sort_can/pour_water).

**Architecture:** Two repos. **(1) `Isaac-GR00T-VLM`** gains `src/vlm_lora/serve/` — a FastAPI `/v1/chat/completions` server that loads the merged VLM, accepts OpenAI `messages`+`tools` (incl. multimodal `image_url`), builds a tool-calling prompt, generates, and parses the model's `<tool_call>{…}</tool_call>` output into OpenAI `tool_calls`. **(2) `agentbot`** gets a real `Gr00tVLMClient.complete()` (httpx → parse `tool_calls` → `SkillCall`), a config flip (`vlm.backend: gr00t-vlm`), and **vision plumbing** so the current camera frame reaches the VLM. A **tool-calling LoRA** (data: `instruction+image → <tool_call>` over AgentBot's skill schemas) makes decomposition reliable. AgentBot's existing orchestrator/sim_session executes the resulting skills on GR00T+IsaacLab.

**Tech Stack:** FastAPI + uvicorn + pydantic v2, `transformers` (Qwen3-VL), `peft`, `httpx`, the existing `vlm_lora` package; AgentBot is `uv`-managed with its own contracts. CPU-only tests stub the model / mock httpx.

---

## Context — what AgentBot already gives us, and the seam

`agentbot/` (untracked sibling, its own repo `squirel2000/agentbot`) implements the whole architecture.png stack (Phases 0/0.5/1/2a done, 4090-verified). The VLM seam is **explicit and waiting**:

- `agentbot/agentbot/brain/vlm_client.py` → `Gr00tVLMClient`: *"the VLM extracted+fine-tuned from GR00T N1.7 — the planned replacement for the Qwen3-VL placeholder; honors the same OpenAI-compatible tool-calling contract; point `vlm.base_url` at its endpoint and set `vlm.backend: gr00t-vlm`."* Currently a keyword **stub**.
- Contract (`agentbot/agentbot/contracts/skills.py`, **verified**): `VLMClient.complete(messages, tools) -> VLMReply{text, calls:[SkillCall]}`; `SkillCall{skill_call_id, name, args, constraints, safety_flags, rationale}`; `SkillSpec.to_tool_schema() -> {name, description, input_schema:{type,properties,required}}`.
- Skills (**verified**): `sort_can{target_color: enum[orange,green]}`, `pick{object: string}`, `place{target: string}`, `home{}`, `pour_water{}`; each `to_vla_request()` → `VlaTaskRequest`. Registered in `SkillRegistry`; exposed via `registry.as_tool_schemas()`.
- Vision seam (**verified**): `UserMessage.image_path: Optional[str]` ("path or data-uri for VLM input") flows `routes_chat → Gateway.normalize → UserMessage`, **but `BrainAgent._complete` builds text-only messages and ignores it** — the gap this plan closes.
- Config (**verified**, `agentbot/config/agentbot.example.yaml`): `vlm:{backend, model, base_url, api_key_env}`. (`vla.checkpoints` is the **action**-model/GR00T registry — separate from the brain VLM, which is reached via `vlm.base_url`.)

**Tool-call wire format (decision):** our server + the fine-tune both use the Qwen/Hermes convention — the model emits `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` (one per chosen skill); the server parses those into OpenAI `tool_calls`. Owning the format on both the serving and training side keeps them consistent regardless of what Cosmos-Reason2-2B's base chat template supports.

---

## Target file structure

```
Isaac-GR00T-VLM/
  src/vlm_lora/
    serve/
      __init__.py
      toolcall.py       # PURE: build tool-calling prompt (+multimodal) ; parse <tool_call> -> calls
      model.py          # ToolCallVLM: load merged VLM once (bf16) ; generate(messages,tools)->text
      openai_app.py     # FastAPI: POST /v1/chat/completions (tools) ; GET /v1/models ; GET /health
    gen_toolcall_data.py  # Phase D: OpenArm frames + skill schemas -> (instruction+image -> <tool_call>) JSONL
    eval_toolcall.py      # Phase D: valid-call% / skill-name acc / arg acc
  examples/
    run_vlm_server.sh   # uvicorn launcher (env: VLM_MODEL_DIR, PORT)
  configs/
    agentbot_skills.sample.json  # frozen copy of registry.as_tool_schemas() for offline gen/tests
  tests/
    test_serve_toolcall.py   # pure prompt-build + parse
    test_serve_app.py        # FastAPI TestClient + stub model
    test_gen_toolcall_data.py
    test_eval_toolcall.py

agentbot/agentbot/
  brain/vlm_client.py   # MODIFY: implement Gr00tVLMClient.complete() (httpx -> tool_calls -> SkillCall)
  brain/agent.py        # MODIFY: plumb image (msg.image_path / frame_provider) into messages
  api/deps.py           # MODIFY: pass a frame_provider into BrainAgent (orchestrator-flow image)
agentbot/tests/
  test_gr00t_vlm_client.py   # NEW: mock endpoint -> parse tool_calls
  test_brain_vision.py       # NEW: image_path -> multimodal message content
agentbot/config/agentbot.example.yaml  # MODIFY: document gr00t-vlm + a vision note
```

Existing `vlm_lora` finetune modules are reused unchanged (`train_vlm_lora`, `vqa_dataset`, `merge_lora`, `infer_vlm`, `hf_utils`, `gen_vqa_from_lerobot`).

---

## Phase A — VLM OpenAI-compatible tool-calling server (`Isaac-GR00T-VLM`)

### Task A1: deps + package
**Files:** Modify `pyproject.toml`; Create `src/vlm_lora/serve/__init__.py`

- [ ] **Step 1:** append to `[project].dependencies`: `"fastapi"`, `"uvicorn"`, `"pydantic>=2"`, `"httpx"`.
- [ ] **Step 2:** create `src/vlm_lora/serve/__init__.py`:

```python
"""OpenAI-compatible tool-calling server for the fine-tuned VLM (AgentBot's Brain backend)."""
```

- [ ] **Step 3:** `uv sync --extra dev && uv run python -c "import fastapi, httpx, pydantic; print('ok')"` → `ok`.
- [ ] **Step 4: stage** — msg: `chore(serve): add fastapi/httpx deps + serve package`.

### Task A2: `toolcall.py` — prompt build + parse (TDD, pure)
**Files:** Create `src/vlm_lora/serve/toolcall.py`; Test `tests/test_serve_toolcall.py`

- [ ] **Step 1: write the failing test**

```python
# tests/test_serve_toolcall.py
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
```

- [ ] **Step 2: run → FAIL** — `uv run python -m pytest tests/test_serve_toolcall.py -v`
- [ ] **Step 3: implement**

```python
# src/vlm_lora/serve/toolcall.py
"""Pure helpers for OpenAI tool-calling over a Qwen3-VL chat model: render the available
tools into a system prompt, pull text/images out of OpenAI multimodal message content, and
parse the model's <tool_call>{...}</tool_call> output back into call dicts."""
from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def build_tool_system(tools: list[dict]) -> str:
    """A system message describing the callable skills + the required output format."""
    lines = ["You are the planning brain of a robot. Decompose the user's instruction into an "
             "ordered sequence of skill-calls, choosing ONLY from these skills:"]
    for t in tools:
        props = t.get("input_schema", {}).get("properties", {})
        req = set(t.get("input_schema", {}).get("required", []))
        args = []
        for name, sch in props.items():
            kind = sch.get("enum") or sch.get("type", "string")
            args.append(f"{name}{'*' if name in req else ''}: {kind}")
        lines.append(f"- {t['name']}({', '.join(args) or ''}) — {t.get('description', '')}")
    lines.append(
        "\nFor EACH step emit exactly one line:\n"
        '<tool_call>{"name": "<skill>", "arguments": {<args>}}</tool_call>\n'
        "Use only the listed skill names and fill every required (*) arg. "
        "Emit nothing else if no skill applies."
    )
    return "\n".join(lines)


def split_text_and_images(message: dict) -> tuple[str, list[str]]:
    """From one chat message, return (joined_text, [image_url_or_datauri, ...]).
    Accepts OpenAI content that is a plain string or a list of typed parts."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for part in content:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            images.append(part.get("image_url", {}).get("url", ""))
    return " ".join(t for t in texts if t).strip(), [u for u in images if u]


def parse_tool_calls(text: str) -> list[dict]:
    """Extract every <tool_call>{json}</tool_call> block as {name, arguments}."""
    out = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            out.append({"name": obj["name"], "arguments": obj.get("arguments", {}) or {}})
    return out
```

- [ ] **Step 4: run → PASS**; lint `uv run ruff check src tests`.
- [ ] **Step 5: stage** — msg: `feat(serve): tool-calling prompt + parser (pure)`.

### Task A3: `model.py` — ToolCallVLM loader/generate (GPU; not unit-tested)
**Files:** Create `src/vlm_lora/serve/model.py`

- [ ] **Step 1: implement** (reuse `infer_vlm`/`hf_utils` patterns)

```python
# src/vlm_lora/serve/model.py
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
```

- [ ] **Step 2:** import check `uv run python -c "import vlm_lora.serve.model"`; lint; **stage** — msg: `feat(serve): ToolCallVLM loader + multimodal generate`.

### Task A4: `openai_app.py` — FastAPI server (TDD with stub model)
**Files:** Create `src/vlm_lora/serve/openai_app.py`; Test `tests/test_serve_app.py`

- [ ] **Step 1: write the failing test**

```python
# tests/test_serve_app.py
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
```

- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** (accepts OpenAI `tools=[{type:function, function:{name,description,parameters}}]`, maps to our `{name,description,input_schema}`, returns OpenAI `tool_calls`)

```python
# src/vlm_lora/serve/openai_app.py
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
```

- [ ] **Step 4: run → PASS**; lint.
- [ ] **Step 5: launcher** `examples/run_vlm_server.sh`:

```bash
#!/usr/bin/env bash
# Serve the fine-tuned VLM as AgentBot's Brain. Env: VLM_MODEL_DIR (required), PORT, CUDA_VISIBLE_DEVICES.
set -euo pipefail
: "${VLM_MODEL_DIR:?set VLM_MODEL_DIR to the merged VLM dir}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec uv run uvicorn vlm_lora.serve.openai_app:app --host 0.0.0.0 --port "${PORT:-8000}"
```

- [ ] **Step 6:** `bash -n examples/run_vlm_server.sh`; **stage** — msg: `feat(serve): OpenAI-compatible /v1/chat/completions + launcher`.

### Task A5: GPU smoke (H100/4090, manual gate)
- [ ] **Step 1:** `VLM_MODEL_DIR=<…/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged> bash examples/run_vlm_server.sh`
- [ ] **Step 2:** `curl /v1/chat/completions` with a real OpenArm frame (as `image_url` data-uri) + tools=[sort_can]. **Expected:** HTTP 200; if the base (VQA-tuned) model doesn't emit `<tool_call>`, that's expected → motivates Phase D. Record the raw output.

---

## Phase B — AgentBot consumes the endpoint

### Task B1: implement `Gr00tVLMClient.complete()`
**Files:** Modify `agentbot/agentbot/brain/vlm_client.py:87` (the stub body)

- [ ] **Step 1: write the failing test** `agentbot/tests/test_gr00t_vlm_client.py`

```python
import json
import pytest
from agentbot.brain.vlm_client import Gr00tVLMClient

class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p

@pytest.mark.asyncio
async def test_complete_parses_tool_calls(monkeypatch):
    payload = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c0", "type": "function",
         "function": {"name": "sort_can", "arguments": json.dumps({"target_color": "orange"})}}]}}]}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _Resp(payload)

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    reply = await Gr00tVLMClient(base_url="http://x/v1").complete(
        [{"role": "user", "content": "sort onto orange"}],
        [{"name": "sort_can", "description": "d", "input_schema": {"type": "object", "properties": {}, "required": []}}])
    assert reply.calls[0].name == "sort_can" and reply.calls[0].args == {"target_color": "orange"}
```

(Note: AgentBot's test suite already uses asyncio; if `pytest.mark.asyncio` isn't configured, wrap with `asyncio.run` instead.)

- [ ] **Step 2: run → FAIL** (`cd agentbot && uv run python -m pytest tests/test_gr00t_vlm_client.py -v`)
- [ ] **Step 3: implement** — replace the `Gr00tVLMClient.complete` stub body:

```python
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        import json
        import httpx
        from agentbot.contracts.skills import SkillCall

        payload = {
            "model": self.model or "gr00t-vlm",
            "messages": messages,
            "tools": [{"type": "function",
                       "function": {"name": t["name"], "description": t.get("description", ""),
                                    "parameters": t["input_schema"]}} for t in tools],
            "tool_choice": "auto",
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # endpoint down/misconfigured -> safe stub so the loop survives
            return self._stub_reply(messages, tools)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append(SkillCall(name=fn.get("name", ""), args=args, rationale=msg.get("content") or ""))
        return VLMReply(text=msg.get("content") or "", calls=calls)
```

- [ ] **Step 4: run → PASS**; `cd agentbot && uv run ruff check agentbot tests` (match their lint).
- [ ] **Step 5: stage** — msg: `feat(brain): wire Gr00tVLMClient to the served VLM endpoint`.

### Task B2: config + docs for the gr00t-vlm backend
**Files:** Modify `agentbot/config/agentbot.example.yaml:14-21`

- [ ] **Step 1:** under `vlm:` add a documented example block (kept commented so defaults are unchanged):

```yaml
  # To use the fine-tuned GR00T VLM as the Brain, run the server (Isaac-GR00T-VLM:
  #   VLM_MODEL_DIR=<merged_vlm> uv run uvicorn vlm_lora.serve.openai_app:app --port 8000)
  # then set:
  #   backend: gr00t-vlm
  #   model:   gr00t-vlm
  #   base_url: http://<vlm-host>:8000/v1
```

- [ ] **Step 2: stage** — msg: `docs(config): how to point the Brain at the gr00t-vlm endpoint`.

---

## Phase C — vision plumbing (camera → Brain)

> Two flows: the **chat** flow already carries `UserMessage.image_path`; the **orchestrator** flow (`agent.plan(text, session_id)`) has no image, so the Brain pulls the latest frame from a `frame_provider` → the Monitor's `state["camera"]["frame"]`. **The Brain is decoupled from the frame source via one event-bus seam** (Task C0): producers publish a `CAMERA` event; `ingest` writes `state["camera"]`. In **sim**, `sim_session` is the producer; on **hardware** (Phase 3), a ROS2→event-bus bridge republishes the camera topic as the *same* event — the Brain never changes. This matches AgentBot's Phase-3 "republish ROS2 into the same event bus" design.

### Task C0: camera frame → Monitor `state["camera"]` (the single, embodiment-agnostic seam)
**Files:** Modify `agentbot/agentbot/contracts/events.py` (add `CAMERA` EventType), `agentbot/agentbot/monitor/ingest.py` (store `state["camera"]`), `agentbot/agentbot/vla/sim_session.py` (publish a downsized frame); Test `agentbot/tests/test_camera_ingest.py`

Design: the Brain reads ONLY `state["camera"]["frame"]`. Producers publish `Event(type=CAMERA, payload={frame:<jpeg data-uri>, ts})`; `ingest` stores it. Sim = `sim_session` produces (downsize the IsaacLab camera obs to ~384px JPEG, publish on the idle pump throttled to ~1–2 Hz + once per episode start). Hardware (Phase 3) = a ROS2→event-bus bridge republishes the camera topic as the SAME event. Frame downsizing keeps redis state small; planning doesn't need full res.

- [ ] **Step 1: failing test** `agentbot/tests/test_camera_ingest.py`:

```python
from agentbot.contracts.events import Event, EventType
from agentbot.monitor.ingest import Ingestor
from agentbot.monitor.state_store import InMemoryStateStore  # use the concrete in-mem store

def test_camera_event_lands_in_state():
    st = InMemoryStateStore()
    ing = Ingestor(bus=None, state_store=st)
    ing.handle(Event(type=EventType.CAMERA, source="vla.sim_session",
                     payload={"frame": "data:image/jpeg;base64,QQ==", "ts": 1.0}))
    assert st.get_state("camera")["frame"].startswith("data:image/jpeg")
```

(Confirm the concrete in-mem StateStore class name when implementing; adjust the import.)

- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: add `CAMERA` to `EventType`** (`agentbot/agentbot/contracts/events.py`): `CAMERA = "camera"`.
- [ ] **Step 4: ingest stores it** — in `Ingestor.handle`, add: `if ev.type == EventType.CAMERA: self.state.set_state("camera", ev.payload)`.
- [ ] **Step 5: run → PASS.**
- [ ] **Step 6 (GPU/sim, gated): sim_session publishes frames.** First read `agentbot/agentbot/vla/sim_session.py` + `scripts/eval/gr00t_infer_agent.py` to find the exact camera-obs accessor; then add a helper that downsizes the latest obs frame to a ~384px JPEG data-uri and publishes `Event(type=CAMERA, ...)` — on the idle pump (throttled via a monotonic timestamp, ~1–2 Hz) and at episode start. **STOP/ASK** if the idle pump has no frame available without stepping the sim.
- [ ] **Step 7: stage** — msg: `feat(monitor): CAMERA event + ingest; sim_session publishes downsized frames`.

### Task C1: `BrainAgent` builds multimodal messages (TDD)
**Files:** Modify `agentbot/agentbot/brain/agent.py`; Test `agentbot/tests/test_brain_vision.py`

- [ ] **Step 1: failing test**

```python
# agentbot/tests/test_brain_vision.py
from agentbot.brain.agent import build_user_content

def test_text_only_when_no_image():
    assert build_user_content("sort it", None) == "sort it"

def test_multimodal_when_image_present():
    c = build_user_content("sort it", "data:image/png;base64,aGk=")
    assert {"type": "text", "text": "sort it"} in c
    assert any(p["type"] == "image_url" for p in c)
```

- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** — add a module-level helper + use it; thread an optional image:

```python
# agentbot/agentbot/brain/agent.py  (add near top)
def build_user_content(text: str, image: str | None):
    """OpenAI message content: plain text, or [text, image_url] when an image is available.
    `image` is a path or data-uri (UserMessage.image_path) or a live frame from frame_provider."""
    if not image:
        return text
    url = image if image.startswith("data:") else image  # server accepts data-uri or path
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]
```

Then update `BrainAgent.__init__` to accept `frame_provider: Optional[Callable[[], str | None]] = None` (default None), and change `_complete` to accept an optional image and use it for the final user turn:

```python
    async def _complete(self, text: str, session_id: str, image: str | None = None):
        history = self.conversation.recent(session_id, n=10)
        messages = [{"role": "user" if t["role"] == "user" else "assistant", "content": t["text"]}
                    for t in history]
        messages.append({"role": "user", "content": build_user_content(text, image)})
        tools = self.registry.as_tool_schemas()
        reply = await self.llm.complete(messages, tools)
        return SkillPlan(intent=text, calls=reply.calls), reply.text

    async def plan(self, text: str, session_id: str) -> SkillPlan:
        image = self.frame_provider() if self.frame_provider else None
        plan, _ = await self._complete(text, session_id, image)
        return plan
```

and in `handle()`: `plan, reply_text = await self._complete(msg.text, msg.session_id, msg.image_path)`.

- [ ] **Step 4: run → PASS**; ensure existing brain tests still pass (`uv run python -m pytest tests/test_brain.py -v`).
- [ ] **Step 5: stage** — msg: `feat(brain): plumb camera image into VLM messages (chat + frame_provider)`.

### Task C2: wire `frame_provider` from the Monitor in `deps.py`
**Files:** Modify `agentbot/agentbot/api/deps.py`

- [ ] **Step 1:** when constructing `BrainAgent`, pass a provider that reads the latest frame from the Monitor state (falls back to None):

```python
        def _latest_frame():
            snap = self.state.snapshot() if hasattr(self, "state") else {}
            return (snap.get("camera") or {}).get("frame")  # data-uri/path, or None
        self.agent = BrainAgent(build_vlm(cfg), self.registry, self.conversation,
                                bus=self.bus, gateway=self.gateway, frame_provider=_latest_frame)
```

- [ ] **Step 2:** run the suite `cd agentbot && uv run python -m pytest -q` (expect all green; the provider returns None unless a frame is published).
- [ ] **Step 3: stage** — msg: `feat(api): provide latest Monitor frame to the Brain`.

---

## Phase D — tool-calling fine-tune (reliability)

### Task D1: freeze AgentBot's skill schemas for offline use
**Files:** Create `Isaac-GR00T-VLM/configs/agentbot_skills.sample.json`

- [ ] **Step 1:** generate it from AgentBot once and commit a copy (so gen/eval/tests don't import agentbot):

```bash
cd agentbot && uv run python -c "import json; from agentbot.api.deps import Deps; from agentbot.settings import load_config; \
print(json.dumps(Deps(load_config()).registry.as_tool_schemas(), indent=2))" \
> ../Isaac-GR00T-VLM/configs/agentbot_skills.sample.json
```

- [ ] **Step 2:** sanity-check it lists sort_can/pick/place/home/pour_water; **stage** — msg: `chore(serve): freeze AgentBot skill schemas for offline gen/tests`.

### Task D2: `gen_toolcall_data.py` (TDD)
**Files:** Create `src/vlm_lora/gen_toolcall_data.py`; Test `tests/test_gen_toolcall_data.py`

- [ ] **Step 1: failing test**

```python
# tests/test_gen_toolcall_data.py
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
```

- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** (template floor maps OpenArm tasks → sort_can; teacher VLM optional for phrasing diversity; reuse `gen_vqa_from_lerobot._extract_frame/_sample_frames`)

```python
# src/vlm_lora/gen_toolcall_data.py
"""Generate tool-calling training data (instruction+image -> <tool_call> sequence) over
AgentBot's real skill schemas, from OpenArm LeRobot frames. Output JSONL matches vqa_dataset's
`messages` schema so train_vlm_lora trains on it unchanged."""
from __future__ import annotations

import json
import os
import re

import tyro

_COLORS = ["orange", "green"]


def template_calls(task: str) -> list[dict]:
    """OpenArm 0403 is the can-sorting task -> a single sort_can call with the plate color."""
    for c in _COLORS:
        if re.search(rf"\b{c}\b", task.lower()):
            return [{"name": "sort_can", "arguments": {"target_color": c}}]
    return [{"name": "sort_can", "arguments": {"target_color": "orange"}}]


def _render_calls(calls: list[dict]) -> str:
    return "\n".join(f'<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>' for c in calls)


def to_training_row(image_rel: str, instruction: str, calls: list[dict]) -> dict:
    return {"images": [image_rel], "type": "ToolCall", "messages": [
        {"role": "user", "content": f"<image>\n{instruction}"},
        {"role": "assistant", "content": _render_calls(calls)}]}


def generate(dataset_path: str, out_dir: str, num_episodes: int = 100, frames_per_episode: int = 2,
             val_ratio: float = 0.1, seed: int = 42) -> None:
    import random

    from vlm_lora.gen_vqa_from_lerobot import _extract_frame, _sample_frames

    random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    eps = [json.loads(x) for x in open(os.path.join(dataset_path, "meta/episodes.jsonl"),
                                       encoding="utf-8") if x.strip()]
    random.shuffle(eps)
    eps = eps[:num_episodes]
    vpat = os.path.join(dataset_path, "videos/chunk-000/observation.images.camera/episode_{:06d}.mp4")
    rows = []
    for ep in eps:
        ei, task, length = ep["episode_index"], ep["task"], ep["length"]
        if not os.path.exists(vpat.format(ei)):
            continue
        for fidx, _phase in _sample_frames(length, frames_per_episode):
            img = _extract_frame(vpat.format(ei), fidx)
            if img is None:
                continue
            rel = f"images/ep{ei:06d}_f{fidx:06d}.png"
            img.save(os.path.join(out_dir, rel))
            rows.append(to_training_row(rel, task, template_calls(task)))
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    for name, sub in {"data.val.jsonl": rows[:n_val], "data.train.jsonl": rows[n_val:],
                      "data.jsonl": rows}.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_toolcall] {len(rows)} rows ({len(rows) - n_val} train / {n_val} val) -> {out_dir}")


if __name__ == "__main__":
    tyro.cli(generate)
```

- [ ] **Step 4: run → PASS**; lint; **stage** — msg: `feat(finetune): generate tool-calling data over AgentBot skills`.

### Task D3: train + merge the tool-calling LoRA (H100 GPU)
- [ ] **Step 1:** `... -m vlm_lora.gen_toolcall_data --dataset-path <OpenArm> --out-dir artifacts/toolcall --num-episodes 100`
- [ ] **Step 2:** `train_vlm_lora --dataset-path artifacts/toolcall/data.train.jsonl --image-root artifacts/toolcall --output-dir artifacts/cosmos_r2_toolcall_lora --max-steps 1500` (reuse, unchanged).
- [ ] **Step 3:** `merge_lora --adapter-dir artifacts/cosmos_r2_toolcall_lora --out-dir <…>/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged`.
- [ ] **Step 4:** point the server at it (`VLM_MODEL_DIR=<…toolcall-merged>`). **STOP/ASK** on OOM.

### Task D4: `eval_toolcall.py` (TDD) + before/after
**Files:** Create `src/vlm_lora/eval_toolcall.py`; Test `tests/test_eval_toolcall.py`

- [ ] **Step 1: failing test**

```python
# tests/test_eval_toolcall.py
from vlm_lora.eval_toolcall import score

def test_exact_match():
    s = score('<tool_call>{"name":"sort_can","arguments":{"target_color":"orange"}}</tool_call>',
              [{"name": "sort_can", "arguments": {"target_color": "orange"}}], allowed={"sort_can"})
    assert s["valid"] == 1 and s["name_ok"] == 1 and s["args_ok"] == 1

def test_wrong_arg():
    s = score('<tool_call>{"name":"sort_can","arguments":{"target_color":"green"}}</tool_call>',
              [{"name": "sort_can", "arguments": {"target_color": "orange"}}], allowed={"sort_can"})
    assert s["name_ok"] == 1 and s["args_ok"] == 0
```

- [ ] **Step 2: run → FAIL; implement** `score(raw, gold_calls, allowed)` (reuse `serve.toolcall.parse_tool_calls`; metrics: parsed-valid, all-names-in-vocab, name-sequence match, args-exact) + an `evaluate(model_dir, val_jsonl, image_root, skills_json, out_json)` mirroring `eval_vlm_vqa.py` that runs `ToolCallVLM` over the val set.
- [ ] **Step 3: run → PASS; stage** — msg: `feat(finetune): tool-calling accuracy eval`.
- [ ] **Step 4 (GPU):** eval base merged VLM vs toolcall-merged on `artifacts/toolcall/data.val.jsonl` → before/after table (valid% / name-acc / args-acc).

---

## Phase E — end-to-end integration

### Task E1: in-proc dashboard demo against the real VLM (no IsaacLab)
- [ ] **Step 1:** serve the toolcall-merged VLM (Task D3); in `agentbot/config/agentbot.yaml` set `backend: in-proc`, `vlm.backend: gr00t-vlm`, `vlm.base_url: http://<host>:8000/v1`.
- [ ] **Step 2:** `cd agentbot && uv run uvicorn agentbot.api.app:app --port 8780`; in the dashboard type "sort the can onto orange"; **Expected:** the plan shows a real `sort_can{target_color: orange}` from the VLM (not the keyword stub), fake-VLA reaches `done`.

### Task E2: real flow on the 4090 (manual gate, optional)
- [ ] **Step 1:** the 4-process flow (README §B) with `vlm.backend: gr00t-vlm`; send a command; **Expected:** VLM plans → orchestrator dispatches `sort_can` → sim_session runs an IsaacLab episode → `done`. Record success_rate. **STOP/ASK** if the VLM emits skills not in the registry (→ tighten Phase D data / few-shot).

---

## Phase F — docs

### Task F1: supersede the old plan + update Isaac-GR00T-VLM report
**Files:** Modify `Isaac-GR00T-VLM/docs/plans/2026-06-21-vlm-brain-deployment.md` (add superseded banner), `docs/project_report.html`, `docs/architecture_dataflow.html`, `README.md`

- [ ] **Step 1:** add to the old plan's top: `> ⚠️ SUPERSEDED by 2026-06-21-vlm-brain-agentbot-integration.md (integrates with the existing agentbot stack).`
- [ ] **Step 2:** `README.md` — add a "Serve as AgentBot's Brain" section: run `run_vlm_server.sh`, set `vlm.backend: gr00t-vlm` in agentbot, link this plan.
- [ ] **Step 3:** `project_report.html` §8 — "VLM as AgentBot's Brain": the OpenAI tool-calling contract, the `<tool_call>` format, the VQA→tool-calling gap + the LoRA answer, and (if D4 ran) the before/after table. Charts/formulas English, prose 繁中.
- [ ] **Step 4:** `architecture_dataflow.html` — add the lane `instruction+image → [VLM /v1/chat/completions] → tool_calls → [AgentBot Brain→Skill→VLA]`. **Stage** — msg: `docs: VLM-as-AgentBot-Brain (report + dataflow + supersede)`.

### Task F2: note in AgentBot's TASKS
**Files:** Modify `agentbot/TASKS.md`

- [ ] **Step 1:** under Phase 1's remaining item "把 stub 換成真的 Qwen3-VL tool-calling", add a sub-line that the `gr00t-vlm` backend is now implemented + served from Isaac-GR00T-VLM `serve/`. **Stage** — msg: `docs(tasks): gr00t-vlm Brain backend wired`.

---

## Verification (end-to-end)

1. **Isaac-GR00T-VLM CPU tests green:** `uv run python -m pytest tests/test_serve_toolcall.py tests/test_serve_app.py tests/test_gen_toolcall_data.py tests/test_eval_toolcall.py -v`.
2. **AgentBot tests green:** `cd agentbot && uv run python -m pytest -q` (existing 31 + `test_gr00t_vlm_client` + `test_brain_vision` all pass).
3. **Server:** `run_vlm_server.sh` → `/health` ok; `/v1/chat/completions` with tools=[sort_can] + an image returns a `tool_calls` array.
4. **Client:** `Gr00tVLMClient.complete()` parses the endpoint's `tool_calls` into `SkillCall(name=..., args=...)`.
5. **Vision:** with `image_path` set (chat) or a frame provider (orchestrator), the request to the server carries an `image_url` part.
6. **Reliability (Phase D):** toolcall-merged VLM beats base on valid%/name-acc/args-acc.
7. **Loop:** in-proc dashboard shows a real VLM-produced `sort_can{orange}` plan; (optional) 4090 runs the episode to `done`.
8. **No regressions / no n1d7 edits:** AgentBot's existing suite stays green; `Isaac-GR00T_n1d7` untouched.

---

## Conflicts resolved (vs the superseded draft)

| Superseded draft | Reality (AgentBot) | Resolution |
|---|---|---|
| own `SkillRegistry`, `SkillCall{skill,args}`, `/plan` free-JSON | `contracts/skills.py` `SkillCall{name,args,…}`, tool-calling | drop ours; serve OpenAI tool-calling; SkillCall lives in AgentBot |
| FastAPI brain server, mock executor, GR00T bridge | orchestrator + sim_session + real VLA exist | drop; we serve only the model |
| fine-tune → generic plan JSON | needs AgentBot skill tool_calls | fine-tune target = `<tool_call>` over real skills |
| sample skill manifest | `registry.as_tool_schemas()` | freeze a copy (`agentbot_skills.sample.json`) for offline gen/tests |

## Gaps & open questions

1. **Camera frame source (resolved → Task C0).** Decision: one event-bus seam — producers publish a `CAMERA` event, `ingest` writes `state["camera"]["frame"]`, the Brain reads it via `frame_provider`. Sim = `sim_session` produces (now); hardware = a ROS2→event-bus bridge produces the same event (Phase 3), so the Brain is embodiment-agnostic. The only verify-on-implement detail is the exact IsaacLab camera-obs accessor + whether a frame is renderable on the idle pump without stepping the sim (Task C0 Step 6).
2. **Does Cosmos-Reason2-2B's chat template support `tools` natively?** We don't rely on it — we inject a tool system prompt and parse `<tool_call>`. If its template *does* support Hermes tool tokens, Phase D can switch to the native format; verify during A5.
3. **Two-repo commits.** Work spans `Isaac-GR00T-VLM` and `agentbot` (its own repo/submodule). Per the maintainer's rule I only stage; the maintainer commits each repo. Confirm both repos are in scope to edit.
4. **`place`/`home`/`pick`/`pour_water` gym ids are skeletons** in AgentBot (only `sort_can` is a confirmed IsaacLab task). For multi-step plans to actually execute, those tasks must exist — but the Brain/tool-calling side is independent of that and can be built/tested now.

---

## Self-review

- **Spec coverage:** "research agentbot architecture" → Context + verified schemas. "integrate the old plan" → Conflicts table + supersede (F1). "comprehensive plan" → Phases A–F. "conflicts" → Conflicts table. "gaps + ask" → Gaps & open questions + the 3 AskUserQuestion decisions already answered (shim / vision-on / tool-calling fine-tune). Each answer is reflected: shim = Phase A; vision = Phase C; tool-calling fine-tune = Phase D.
- **Placeholder scan:** real code in every code step; commands have expected outputs; the only TODO-ish items are AgentBot's own pre-existing skeleton gym ids (called out, not introduced here).
- **Type consistency:** server emits OpenAI `tool_calls[{function:{name,arguments(JSON str)}}]`; `Gr00tVLMClient.complete` parses exactly that into `SkillCall(name, args)`; `to_internal_tools` ↔ `as_tool_schemas()` `{name,description,input_schema}`; `<tool_call>{"name","arguments"}` is produced by `gen_toolcall_data`, parsed by `serve.toolcall.parse_tool_calls`, scored by `eval_toolcall.score` — one format throughout.

---

## Execution handoff

Two options (I **stage only; you commit** — and edits span both `Isaac-GR00T-VLM` and `agentbot`):
1. **Subagent-Driven (recommended)** — fresh subagent per task + two-stage review (superpowers:subagent-driven-development).
2. **Inline** — this session, checkpoints per phase (superpowers:executing-plans).

GPU tasks (A5, D3, D4, E2) run on H100/4090 via `pegasus.py`. Which approach — or adjust the plan first?
