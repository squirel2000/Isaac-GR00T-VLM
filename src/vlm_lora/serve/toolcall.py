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
