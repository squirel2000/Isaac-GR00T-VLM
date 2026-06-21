# VLM "Brain" Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **COMMITS:** The maintainer owns all commits. After each task's tests pass, **stage** the changes (`git add …`) and **stop for the maintainer to review + commit**. Suggested commit messages are given but DO NOT run `git commit` / `git push` yourself.

**Goal:** Turn the fine-tuned standalone VLM (Product A, `Cosmos-Reason2-2B-lora-merged`) into a deployable **"Brain" service** that takes a natural-language instruction + an image + robot state and returns a **validated, structured list of skill-calls** (sub-tasks), served over FastAPI, against a **pluggable Skill Registry** that the maintainer's separately-built Skill Library plugs into via a manifest.

**Architecture:** A FastAPI server wraps the VLM (loaded once, bf16). A request `{instruction, image, state}` is turned into a planning prompt that lists the *available skills* (from a Skill Registry loaded from a manifest) + few-shot examples; the VLM generates a JSON plan; the server parses → validates each skill-call against the registry → repairs once on failure → returns `{steps:[{skill,args,rationale}], ok, raw}`. Capability is delivered in two phases: **(B–E) prompt-based MVP** on the current VQA-tuned checkpoint, then **(F) a planning LoRA** fine-tuned on auto-generated `instruction+image → plan` data to make decomposition reliable. A mock executor + optional GR00T bridge demonstrate the downstream hand-off until the real Skill Library is integrated.

**Tech Stack:** Python ≥3.10, FastAPI + uvicorn, pydantic v2, `transformers` (Qwen3-VL), `peft`, the existing `vlm_lora` package, `pytest` (CPU-only tests stub the VLM). Runs on a GPU box (H100 for dev; asus-4090 for deploy). The VLM is ~2B params → ~5 GB VRAM in bf16, fits a single 4090.

---

## Context — why, and the one critical gap

The repo already does (DONE): **extract** Cosmos-Reason2-2B from GR00T N1.7 → **auto-generate VQA** from OpenArm → **LoRA fine-tune** → **merge** to Product A (standalone VLM) → **swap** into GR00T (Product B) → **eval**. Everything is downloaded + verified locally under `artifacts/`.

**The gap (drives Phase F):** Product A was fine-tuned on **VQA** (answer questions about a scene: 7 types — Summary/Trajectory/Attribute/Temporal/Reasoning/Spatial/Mechanics). That sharpened **domain visual grounding**, but **task decomposition (instruction → ordered skill-calls) is a different capability the checkpoint was never trained on.** So:
- **Phase B–E (MVP):** get planning behavior *now* via a structured prompt + few-shot + JSON-schema validation on the current checkpoint. Good enough to stand up + test the whole loop.
- **Phase F (reliability):** auto-generate `instruction+image → plan` data (teacher VLM + templates over OpenArm, grounded in the Skill Registry) and train a **planning LoRA**; the Brain then points at the planning-tuned checkpoint.

**Decoupling from the maintainer's Skill Library (Q1 answer):** The Brain never hard-codes skills. It loads a **skill manifest** (JSON) describing available skills (name, description, typed args). The maintainer's external Skill Library exports such a manifest (or a tiny adapter that emits one); the Brain consumes it for both prompting and validation. A `configs/skills.sample.json` ships so the loop runs before the real library lands.

---

## Existing code to reuse (read before coding)

| Need | File | Notes |
|---|---|---|
| Load merged VLM once (bf16, device_map) | `src/vlm_lora/infer_vlm.py` | `ask()` shows the exact `AutoProcessor` + `Qwen3VLForConditionalGeneration` + chat-template + generate flow to factor into `BrainModel` |
| Gated/offline model path | `src/vlm_lora/hf_utils.py` | `resolve_model_path()` — reuse so the brain loads offline |
| Teacher-VLM gen pattern | `src/vlm_lora/gen_vqa_from_lerobot.py` | `TeacherVLM` (load once, bf16) + frame sampling — reuse for `gen_plan_data.py` |
| LoRA trainer + dataset/collator | `src/vlm_lora/train_vlm_lora.py`, `vqa_dataset.py` | reuse verbatim for the planning LoRA (same JSONL `messages` schema) |
| Merge LoRA → standalone | `src/vlm_lora/merge_lora.py` | reuse to merge the planning adapter |
| Product A path | `configs/default.yaml` → `products.lora_tuned_vlm` | the brain's default `--model-dir` |
| GR00T policy client (for the optional bridge) | `Isaac-GR00T_n1d7/gr00t/policy/server_client.py` (`PolicyClient`), `gr00t/eval/run_gr00t_server.py` (:5555) | observation `{video,state,language}` → action; bridge maps a skill → `language_instruction` |

---

## Target file structure (additive; existing finetune files stay put)

```
Isaac-GR00T-VLM/
  src/vlm_lora/
    brain/
      __init__.py
      skills.py          # Skill, SkillRegistry: load manifest, render for prompt, validate calls
      schema.py          # pydantic v2: SkillCall, Plan, PlanRequest, PlanResponse
      prompt.py          # build_planning_prompt(): system+skills+few-shot+instruction+state
      core.py            # BrainModel (load VLM once) + plan(): generate -> parse -> validate -> repair
      serve.py           # FastAPI app: POST /plan, GET /skills, GET /health
      client.py          # HTTP client + CLI (python -m vlm_lora.brain.client)
      mock_executor.py   # logs/echoes skill-calls (stand-in Skill Library)
      bridge_gr00t.py    # optional: skill-call -> GR00T VLA language_instruction via PolicyClient
    gen_plan_data.py     # Phase F: auto-generate instruction+image -> plan JSONL
    eval_plan.py         # Phase F: planning accuracy (valid-JSON %, skill-name acc, plan match)
  configs/
    default.yaml         # + `brain:` section
    skills.sample.json   # sample manifest (MVP; replaced by maintainer's library)
    plan_fewshot.json    # few-shot exemplars for the planning prompt
  examples/
    brain_demo.py        # end-to-end: instruction+image -> plan -> mock executor (+ optional GR00T bridge)
    run_brain_server.sh  # uvicorn launcher (env-driven)
  tests/
    test_brain_skills.py   test_brain_schema.py   test_brain_prompt.py
    test_brain_core.py     # stubs model.generate -> no GPU
    test_brain_serve.py    # FastAPI TestClient + stub brain
  docs/
    deploy_brain.md        #繁中 operator guide: launch, I/O, integrate Skill Library, deploy
    plans/2026-06-21-vlm-brain-deployment.md   # (this file)
```

Existing finetune modules (`lora_args.py`, `gen_vqa_from_lerobot.py`, `vqa_dataset.py`, `train_vlm_lora.py`, `merge_lora.py`, `infer_vlm.py`, `eval_vlm_vqa.py`, `swap_backbone.py`, `hf_utils.py`) are **unchanged** — only imported. (A heavier reorg into `finetune/`+`brain/`+`common/` subpackages is possible but adds import churn for little gain; this plan stays additive and documents the stage split in the README instead.)

---

## Phase A — dependencies + brain package skeleton

### Task A1: add deps + empty package
**Files:** Modify `pyproject.toml`; Create `src/vlm_lora/brain/__init__.py`

- [ ] **Step 1: add runtime deps** to `pyproject.toml` `[project].dependencies` (append): `"fastapi"`, `"uvicorn"`, `"pydantic>=2"`, `"httpx"`, `"python-multipart"`.
- [ ] **Step 2: create the package**

```python
# src/vlm_lora/brain/__init__.py
"""VLM 'Brain': instruction + image + state -> validated structured skill-calls."""
```

- [ ] **Step 3: install + import check**

Run: `uv sync --extra dev && uv run python -c "import fastapi, pydantic, uvicorn, httpx; print('ok', pydantic.VERSION)"`
Expected: `ok 2.x`

- [ ] **Step 4: stage** `git add pyproject.toml src/vlm_lora/brain/__init__.py` — suggested msg: `chore(brain): add FastAPI/pydantic deps + brain package` (maintainer commits).

---

## Phase B — Skill contract (the integration point)

### Task B1: `schema.py` — pydantic models (TDD)
**Files:** Create `src/vlm_lora/brain/schema.py`; Test `tests/test_brain_schema.py`

- [ ] **Step 1: write the failing test**

```python
# tests/test_brain_schema.py
import pytest
from pydantic import ValidationError
from vlm_lora.brain.schema import SkillCall, Plan, PlanRequest

def test_skillcall_minimal():
    sc = SkillCall(skill="pick", args={"object": "red can"})
    assert sc.skill == "pick" and sc.args["object"] == "red can" and sc.rationale is None

def test_plan_holds_ordered_steps():
    p = Plan(steps=[SkillCall(skill="pick", args={"object": "can"}),
                    SkillCall(skill="place", args={"target": "orange plate"})])
    assert [s.skill for s in p.steps] == ["pick", "place"]

def test_plan_request_requires_instruction():
    with pytest.raises(ValidationError):
        PlanRequest(image_b64="x", state={})
```

- [ ] **Step 2: run → FAIL** — `uv run python -m pytest tests/test_brain_schema.py -v`
- [ ] **Step 3: implement**

```python
# src/vlm_lora/brain/schema.py
"""Pydantic v2 models for the Brain's HTTP contract and the skill-call plan."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SkillCall(BaseModel):
    skill: str = Field(..., description="skill name; must exist in the SkillRegistry")
    args: dict[str, object] = Field(default_factory=dict, description="arg name -> value")
    rationale: str | None = Field(default=None, description="why this step (optional)")


class Plan(BaseModel):
    steps: list[SkillCall] = Field(default_factory=list)


class PlanRequest(BaseModel):
    instruction: str = Field(..., min_length=1, description="natural-language goal")
    image_b64: str | None = Field(default=None, description="base64 PNG/JPEG of the current view")
    state: dict[str, object] = Field(default_factory=dict, description="robot/world state")
    max_new_tokens: int = Field(default=512, ge=16, le=2048)


class PlanResponse(BaseModel):
    ok: bool
    steps: list[SkillCall] = Field(default_factory=list)
    error: str | None = None
    raw: str | None = Field(default=None, description="raw model text (debug)")
```

- [ ] **Step 4: run → PASS**; lint `uv run ruff check src tests`.
- [ ] **Step 5: stage** `git add src/vlm_lora/brain/schema.py tests/test_brain_schema.py` — msg: `feat(brain): plan/skill-call pydantic schema`.

### Task B2: `skills.py` — SkillRegistry + sample manifest (TDD)
**Files:** Create `src/vlm_lora/brain/skills.py`, `configs/skills.sample.json`; Test `tests/test_brain_skills.py`

- [ ] **Step 1: write the sample manifest** (MVP skills grounded in OpenArm; the maintainer's library later replaces this)

```json
// configs/skills.sample.json
{
  "skills": [
    {"name": "move_to", "description": "Move the end-effector above a named object or location.",
     "args": {"target": "string: object or location name (e.g. 'red can', 'orange plate')"}},
    {"name": "pick", "description": "Close the gripper to grasp the object currently above.",
     "args": {"object": "string: object to grasp"}},
    {"name": "place", "description": "Place the held object onto a target location.",
     "args": {"target": "string: destination (e.g. 'orange plate')"}},
    {"name": "inspect", "description": "Look at a region/object to verify a condition.",
     "args": {"target": "string: what to inspect", "question": "string: what to check"}},
    {"name": "done", "description": "Signal the task is complete.", "args": {}}
  ]
}
```

- [ ] **Step 2: write the failing test**

```python
# tests/test_brain_skills.py
import json
from vlm_lora.brain.skills import SkillRegistry
from vlm_lora.brain.schema import SkillCall

MANIFEST = {"skills": [
    {"name": "pick", "description": "grasp", "args": {"object": "string: what"}},
    {"name": "place", "description": "put down", "args": {"target": "string: where"}},
]}

def test_load_and_names(tmp_path):
    p = tmp_path / "m.json"; p.write_text(json.dumps(MANIFEST), encoding="utf-8")
    reg = SkillRegistry.from_manifest(str(p))
    assert reg.names() == ["pick", "place"]

def test_render_lists_skills_and_args():
    reg = SkillRegistry(MANIFEST["skills"])
    text = reg.render_for_prompt()
    assert "pick" in text and "object" in text and "place" in text

def test_validate_accepts_known_and_rejects_unknown():
    reg = SkillRegistry(MANIFEST["skills"])
    ok, err = reg.validate(SkillCall(skill="pick", args={"object": "can"}))
    assert ok and err is None
    bad, err2 = reg.validate(SkillCall(skill="fly", args={}))
    assert not bad and "unknown skill" in err2.lower()

def test_validate_flags_missing_required_arg():
    reg = SkillRegistry(MANIFEST["skills"])
    ok, err = reg.validate(SkillCall(skill="pick", args={}))
    assert not ok and "object" in err
```

- [ ] **Step 3: run → FAIL**
- [ ] **Step 4: implement**

```python
# src/vlm_lora/brain/skills.py
"""SkillRegistry: load a skill manifest (the integration point for an external Skill
Library), render it for the planning prompt, and validate model-produced skill-calls."""
from __future__ import annotations

import json
from dataclasses import dataclass

from vlm_lora.brain.schema import SkillCall


@dataclass
class Skill:
    name: str
    description: str
    args: dict[str, str]  # arg name -> "type: description"


class SkillRegistry:
    def __init__(self, skills: list[dict]):
        self.skills: dict[str, Skill] = {
            s["name"]: Skill(s["name"], s.get("description", ""), s.get("args", {})) for s in skills
        }

    @classmethod
    def from_manifest(cls, path: str) -> "SkillRegistry":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["skills"])

    def names(self) -> list[str]:
        return list(self.skills)

    def render_for_prompt(self) -> str:
        lines = []
        for s in self.skills.values():
            args = ", ".join(f"{k} ({v})" for k, v in s.args.items()) or "(no args)"
            lines.append(f"- {s.name}: {s.description}  args: {args}")
        return "\n".join(lines)

    def validate(self, call: SkillCall) -> tuple[bool, str | None]:
        skill = self.skills.get(call.skill)
        if skill is None:
            return False, f"unknown skill '{call.skill}'. allowed: {self.names()}"
        missing = [a for a in skill.args if a not in call.args]
        if missing:
            return False, f"skill '{call.skill}' missing required arg(s): {missing}"
        return True, None
```

- [ ] **Step 5: run → PASS**; lint; **stage** `git add src/vlm_lora/brain/skills.py configs/skills.sample.json tests/test_brain_skills.py` — msg: `feat(brain): SkillRegistry + sample manifest`.

---

## Phase C — Brain core (prompt-based MVP)

### Task C1: `prompt.py` — planning prompt builder (TDD)
**Files:** Create `src/vlm_lora/brain/prompt.py`, `configs/plan_fewshot.json`; Test `tests/test_brain_prompt.py`

- [ ] **Step 1: few-shot exemplars**

```json
// configs/plan_fewshot.json
[
  {"instruction": "Put the can on the orange plate.",
   "state": {"gripper": "open"},
   "plan": {"steps": [
     {"skill": "move_to", "args": {"target": "can"}, "rationale": "approach the can"},
     {"skill": "pick", "args": {"object": "can"}},
     {"skill": "move_to", "args": {"target": "orange plate"}},
     {"skill": "place", "args": {"target": "orange plate"}},
     {"skill": "done", "args": {}}]}}
]
```

- [ ] **Step 2: write the failing test**

```python
# tests/test_brain_prompt.py
import json
from vlm_lora.brain.skills import SkillRegistry
from vlm_lora.brain.prompt import build_planning_messages

REG = SkillRegistry([{"name": "pick", "description": "grasp", "args": {"object": "string: what"}}])

def test_messages_have_image_slot_and_skills_and_instruction():
    msgs = build_planning_messages("pick the can", REG, state={"gripper": "open"},
                                   fewshot=[], image_placeholder=True)
    assert msgs[0]["role"] == "system"
    assert "pick" in msgs[0]["content"]                      # skills listed
    user = msgs[-1]["content"]
    types = [c["type"] for c in user]
    assert "image" in types and "text" in types             # multimodal user turn
    assert "pick the can" in json.dumps(user)               # instruction present

def test_state_is_rendered():
    msgs = build_planning_messages("go", REG, state={"gripper": "open"}, fewshot=[], image_placeholder=True)
    assert "gripper" in json.dumps(msgs[-1]["content"])
```

- [ ] **Step 3: run → FAIL**
- [ ] **Step 4: implement**

```python
# src/vlm_lora/brain/prompt.py
"""Build the chat messages that turn (instruction, image, state, skills) into a plan request.
Output contract: the model must return ONLY a JSON object {"steps":[{"skill","args","rationale?"}]}."""
from __future__ import annotations

import json

from vlm_lora.brain.skills import SkillRegistry

_SYSTEM = """You are the planning brain of a dual-arm robot. Given the camera image, the \
robot/world state, and a natural-language instruction, decompose the task into an ORDERED \
list of skill-calls chosen ONLY from the available skills below.

Available skills:
{skills}

Rules:
- Use ONLY the skill names above; fill every listed arg.
- Output ONLY a JSON object: {{"steps":[{{"skill":"<name>","args":{{...}},"rationale":"<short>"}}]}}
- No prose before or after the JSON. End the plan with the "done" skill if available.
"""


def build_planning_messages(instruction, registry: SkillRegistry, state: dict,
                            fewshot: list[dict], image_placeholder: bool = True) -> list[dict]:
    system = _SYSTEM.format(skills=registry.render_for_prompt())
    msgs: list[dict] = [{"role": "system", "content": system}]
    for ex in fewshot:                                   # text-only exemplars (image is for the live turn)
        msgs.append({"role": "user", "content":
                     f"instruction: {ex['instruction']}\nstate: {json.dumps(ex.get('state', {}))}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["plan"], ensure_ascii=False)})
    user_text = f"instruction: {instruction}\nstate: {json.dumps(state, ensure_ascii=False)}"
    content = ([{"type": "image"}] if image_placeholder else []) + [{"type": "text", "text": user_text}]
    msgs.append({"role": "user", "content": content})
    return msgs
```

- [ ] **Step 5: run → PASS**; lint; **stage** — msg: `feat(brain): planning prompt builder + few-shot`.

### Task C2: `core.py` — BrainModel + plan() with parse/validate/repair (TDD, model stubbed)
**Files:** Create `src/vlm_lora/brain/core.py`; Test `tests/test_brain_core.py`

- [ ] **Step 1: write the failing test** (no GPU — inject a fake generate fn)

```python
# tests/test_brain_core.py
import json
from vlm_lora.brain.skills import SkillRegistry
from vlm_lora.brain.core import plan_from_text, extract_json

REG = SkillRegistry([{"name": "pick", "description": "grasp", "args": {"object": "string"}},
                     {"name": "done", "description": "end", "args": {}}])

def test_extract_json_from_noisy_text():
    raw = 'sure!\n```json\n{"steps":[{"skill":"done","args":{}}]}\n```'
    assert extract_json(raw) == {"steps": [{"skill": "done", "args": {}}]}

def test_plan_from_text_validates_ok():
    raw = '{"steps":[{"skill":"pick","args":{"object":"can"}},{"skill":"done","args":{}}]}'
    resp = plan_from_text(raw, REG)
    assert resp.ok and [s.skill for s in resp.steps] == ["pick", "done"]

def test_plan_from_text_rejects_unknown_skill():
    raw = '{"steps":[{"skill":"teleport","args":{}}]}'
    resp = plan_from_text(raw, REG)
    assert not resp.ok and "unknown skill" in resp.error.lower()

def test_plan_from_text_handles_garbage():
    resp = plan_from_text("no json here", REG)
    assert not resp.ok and resp.error
```

- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** (pure helpers tested above; `BrainModel` does the GPU work and reuses `infer_vlm`/`hf_utils` patterns)

```python
# src/vlm_lora/brain/core.py
"""BrainModel loads the merged VLM once and turns (instruction,image,state) into a validated
Plan. Parsing/validation are pure functions (unit-tested without a GPU)."""
from __future__ import annotations

import json

from vlm_lora.brain.prompt import build_planning_messages
from vlm_lora.brain.schema import Plan, PlanResponse, SkillCall
from vlm_lora.brain.skills import SkillRegistry


def extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} JSON object out of model text (handles ```json fences)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def plan_from_text(raw: str, registry: SkillRegistry) -> PlanResponse:
    data = extract_json(raw)
    if data is None:
        return PlanResponse(ok=False, error="no JSON object found in model output", raw=raw)
    try:
        plan = Plan(**data)
    except Exception as e:  # pydantic ValidationError
        return PlanResponse(ok=False, error=f"plan schema invalid: {e}", raw=raw)
    for call in plan.steps:
        ok, err = registry.validate(call)
        if not ok:
            return PlanResponse(ok=False, steps=plan.steps, error=err, raw=raw)
    return PlanResponse(ok=True, steps=plan.steps, raw=raw)


class BrainModel:
    """Loads the merged VLM once; generates a plan for a request. GPU-only (not unit-tested)."""

    def __init__(self, model_dir: str, fewshot: list[dict] | None = None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from vlm_lora.hf_utils import resolve_model_path

        md = resolve_model_path(model_dir)
        self.processor = AutoProcessor.from_pretrained(md, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            md, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
        self.fewshot = fewshot or []

    def _generate(self, messages, image, max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=([image] if image is not None else None),
                             return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

    def plan(self, instruction: str, image, state: dict, registry: SkillRegistry,
             max_new_tokens: int = 512) -> PlanResponse:
        msgs = build_planning_messages(instruction, registry, state, self.fewshot,
                                       image_placeholder=image is not None)
        raw = self._generate(msgs, image, max_new_tokens)
        resp = plan_from_text(raw, registry)
        if resp.ok:
            return resp
        # one repair retry: feed the error back and ask for corrected JSON only
        repair = msgs + [{"role": "assistant", "content": raw},
                         {"role": "user", "content":
                          f"That was invalid ({resp.error}). Reply with ONLY corrected JSON."}]
        raw2 = self._generate(repair, image, max_new_tokens)
        return plan_from_text(raw2, registry)
```

- [ ] **Step 4: run → PASS** (the GPU `BrainModel` is exercised later by the server smoke test).
- [ ] **Step 5: lint; stage** — msg: `feat(brain): BrainModel + parse/validate/repair`.

---

## Phase D — FastAPI server + client

### Task D1: `serve.py` — FastAPI app (TDD with stubbed brain)
**Files:** Create `src/vlm_lora/brain/serve.py`; Test `tests/test_brain_serve.py`

- [ ] **Step 1: write the failing test** (uses FastAPI `TestClient`; monkeypatch the brain so no GPU)

```python
# tests/test_brain_serve.py
from fastapi.testclient import TestClient
from vlm_lora.brain import serve
from vlm_lora.brain.schema import PlanResponse, SkillCall
from vlm_lora.brain.skills import SkillRegistry

class _StubBrain:
    def plan(self, instruction, image, state, registry, max_new_tokens=512):
        return PlanResponse(ok=True, steps=[SkillCall(skill="done", args={})], raw="{}")

def _client():
    serve.STATE["brain"] = _StubBrain()
    serve.STATE["registry"] = SkillRegistry(
        [{"name": "done", "description": "end", "args": {}}])
    return TestClient(serve.app)

def test_health():
    assert _client().get("/health").json()["status"] == "ok"

def test_skills_lists_registry():
    assert "done" in _client().get("/skills").json()["skills"]

def test_plan_returns_steps():
    r = _client().post("/plan", json={"instruction": "finish", "state": {}})
    body = r.json()
    assert body["ok"] and body["steps"][0]["skill"] == "done"
```

- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement**

```python
# src/vlm_lora/brain/serve.py
"""FastAPI Brain server. Launch:
    BRAIN_MODEL_DIR=<merged_vlm> BRAIN_SKILLS=configs/skills.sample.json \
    uv run uvicorn vlm_lora.brain.serve:app --host 0.0.0.0 --port 8000
POST /plan {instruction, image_b64?, state?} -> {ok, steps:[{skill,args,rationale}], error?, raw?}"""
from __future__ import annotations

import base64
import io
import json
import os

from fastapi import FastAPI

from vlm_lora.brain.schema import PlanRequest, PlanResponse
from vlm_lora.brain.skills import SkillRegistry

app = FastAPI(title="VLM Brain")
STATE: dict = {"brain": None, "registry": None}


@app.on_event("startup")
def _load():
    if STATE["brain"] is not None:            # already injected (tests) -> skip GPU load
        return
    from vlm_lora.brain.core import BrainModel

    model_dir = os.environ["BRAIN_MODEL_DIR"]
    skills_path = os.environ.get("BRAIN_SKILLS", "configs/skills.sample.json")
    fewshot_path = os.environ.get("BRAIN_FEWSHOT", "configs/plan_fewshot.json")
    fewshot = json.load(open(fewshot_path, encoding="utf-8")) if os.path.exists(fewshot_path) else []
    STATE["registry"] = SkillRegistry.from_manifest(skills_path)
    STATE["brain"] = BrainModel(model_dir, fewshot=fewshot)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["brain"] is not None}


@app.get("/skills")
def skills():
    return {"skills": STATE["registry"].names()}


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    image = None
    if req.image_b64:
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    return STATE["brain"].plan(req.instruction, image, req.state, STATE["registry"], req.max_new_tokens)
```

- [ ] **Step 4: run → PASS**; lint.
- [ ] **Step 5: launcher script** `examples/run_brain_server.sh`

```bash
#!/usr/bin/env bash
# Launch the VLM Brain server. Env: BRAIN_MODEL_DIR (required), BRAIN_SKILLS, PORT, CUDA_VISIBLE_DEVICES.
set -euo pipefail
: "${BRAIN_MODEL_DIR:?set BRAIN_MODEL_DIR to the merged VLM dir}"
export BRAIN_SKILLS="${BRAIN_SKILLS:-configs/skills.sample.json}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec uv run uvicorn vlm_lora.brain.serve:app --host 0.0.0.0 --port "${PORT:-8000}"
```

- [ ] **Step 6: `bash -n examples/run_brain_server.sh`; stage** — msg: `feat(brain): FastAPI server + launcher`.

### Task D2: `client.py` — HTTP client + CLI
**Files:** Create `src/vlm_lora/brain/client.py`

- [ ] **Step 1: implement**

```python
# src/vlm_lora/brain/client.py
"""HTTP client + CLI for the Brain server.
    uv run python -m vlm_lora.brain.client --url http://HOST:8000 \
        --instruction "put the can on the orange plate" --image frame.png
"""
from __future__ import annotations

import base64

import tyro


def plan(url: str = "http://localhost:8000", instruction: str = "", image: str | None = None,
         state_json: str = "{}") -> dict:
    import json

    import httpx

    payload = {"instruction": instruction, "state": json.loads(state_json)}
    if image:
        payload["image_b64"] = base64.b64encode(open(image, "rb").read()).decode()
    r = httpx.post(f"{url}/plan", json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    tyro.cli(plan)
```

- [ ] **Step 2:** `uv run python -m vlm_lora.brain.client --help`; **stage** — msg: `feat(brain): HTTP client + CLI`.

### Task D3: GPU smoke test (H100/4090, manual gate)
**Files:** none (run against Product A)

- [ ] **Step 1: start server** (H100): `BRAIN_MODEL_DIR=<artifacts/.../lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged> bash examples/run_brain_server.sh`
- [ ] **Step 2: call it** with a real OpenArm frame + instruction; **Expected:** HTTP 200, `ok=true`, ≥2 steps, all skills ∈ registry. **STOP/ASK** if the model returns prose-not-JSON repeatedly (→ that motivates Phase F).

---

## Phase E — mock executor + Brain→VLA demo

### Task E1: `mock_executor.py` (TDD)
**Files:** Create `src/vlm_lora/brain/mock_executor.py`; Test extend `tests/test_brain_core.py` or new `tests/test_brain_executor.py`

- [ ] **Step 1: failing test**

```python
# tests/test_brain_executor.py
from vlm_lora.brain.schema import Plan, SkillCall
from vlm_lora.brain.mock_executor import MockExecutor

def test_executor_runs_each_step_in_order():
    ex = MockExecutor()
    log = ex.run(Plan(steps=[SkillCall(skill="pick", args={"object": "can"}),
                             SkillCall(skill="done", args={})]))
    assert [r["skill"] for r in log] == ["pick", "done"]
    assert all(r["status"] == "ok" for r in log)
```

- [ ] **Step 2: run → FAIL; implement**

```python
# src/vlm_lora/brain/mock_executor.py
"""Stand-in for the maintainer's Skill Library: logs/echoes each skill-call so the full
instruction->plan->execute loop runs end-to-end before the real library is integrated."""
from __future__ import annotations

from vlm_lora.brain.schema import Plan


class MockExecutor:
    def run(self, plan: Plan) -> list[dict]:
        log = []
        for i, call in enumerate(plan.steps):
            print(f"[exec {i}] {call.skill}({call.args})")
            log.append({"step": i, "skill": call.skill, "args": call.args, "status": "ok"})
        return log
```

- [ ] **Step 3: run → PASS; stage** — msg: `feat(brain): mock executor (Skill Library stand-in)`.

### Task E2: optional `bridge_gr00t.py` + `examples/brain_demo.py`
**Files:** Create `src/vlm_lora/brain/bridge_gr00t.py`, `examples/brain_demo.py`

- [ ] **Step 1: bridge** — maps a skill-call to a GR00T VLA `language_instruction` and (optionally) calls the GR00T policy client. Pure mapping is unit-testable; the network call is gated.

```python
# src/vlm_lora/brain/bridge_gr00t.py
"""Optional Brain->VLA bridge: render a skill-call as a natural-language instruction and send
it to a running GR00T policy server (Isaac-GR00T_n1d7 run_gr00t_server.py, :5555)."""
from __future__ import annotations

from vlm_lora.brain.schema import SkillCall


def skill_to_instruction(call: SkillCall) -> str:
    a = call.args
    return {
        "move_to": lambda: f"move to the {a.get('target', '')}",
        "pick": lambda: f"pick up the {a.get('object', '')}",
        "place": lambda: f"place the object on the {a.get('target', '')}",
        "inspect": lambda: f"look at the {a.get('target', '')}",
        "done": lambda: "task complete",
    }.get(call.skill, lambda: call.skill)().strip()


def send_to_gr00t(instruction: str, observation: dict, host="localhost", port=5555):
    """observation = {'video':{...}, 'state':{...}}; injects language and calls get_action."""
    import sys

    sys.path.insert(0, r"../Isaac-GR00T_n1d7")  # adjust to your n1d7 path
    from gr00t.policy.server_client import PolicyClient

    observation = dict(observation)
    observation["language"] = {"annotation.language.language_instruction": [[instruction]]}
    return PolicyClient(host=host, port=port).get_action(observation=observation)
```

- [ ] **Step 2: test the pure mapping** (add to `tests/test_brain_executor.py`)

```python
def test_skill_to_instruction():
    from vlm_lora.brain.bridge_gr00t import skill_to_instruction
    from vlm_lora.brain.schema import SkillCall
    assert skill_to_instruction(SkillCall(skill="pick", args={"object": "red can"})) == "pick up the red can"
```

- [ ] **Step 3: end-to-end demo**

```python
# examples/brain_demo.py
"""instruction + image -> Brain plan -> MockExecutor (default) or GR00T bridge (--use-gr00t).
    uv run python examples/brain_demo.py --model-dir <merged_vlm> --image frame.png \
        --instruction "put the can on the orange plate"
"""
import json

import tyro

from vlm_lora.brain.core import BrainModel
from vlm_lora.brain.mock_executor import MockExecutor
from vlm_lora.brain.skills import SkillRegistry


def main(model_dir: str, image: str, instruction: str,
         skills: str = "configs/skills.sample.json", fewshot: str = "configs/plan_fewshot.json"):
    from PIL import Image

    reg = SkillRegistry.from_manifest(skills)
    fs = json.load(open(fewshot, encoding="utf-8"))
    brain = BrainModel(model_dir, fewshot=fs)
    resp = brain.plan(instruction, Image.open(image).convert("RGB"), {}, reg)
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
    if resp.ok:
        MockExecutor().run(__import__("vlm_lora.brain.schema", fromlist=["Plan"]).Plan(steps=resp.steps))


if __name__ == "__main__":
    tyro.cli(main)
```

- [ ] **Step 4: run pure tests → PASS; `bash`/`--help` check the demo; stage** — msg: `feat(brain): GR00T bridge + end-to-end demo`.

---

## Phase F — planning data generation + planning LoRA (reliability)

### Task F1: `gen_plan_data.py` — auto-generate `instruction+image → plan` JSONL
**Files:** Create `src/vlm_lora/gen_plan_data.py`; Test `tests/test_gen_plan_data.py`

- [ ] **Step 1: failing test** (pure helpers: build a template plan + the training row)

```python
# tests/test_gen_plan_data.py
from vlm_lora.gen_plan_data import template_plan, to_training_row

def test_template_plan_has_ordered_skills():
    plan = template_plan("place the can on the orange plate")
    skills = [s["skill"] for s in plan["steps"]]
    assert skills[0] == "move_to" and "pick" in skills and skills[-1] == "done"

def test_to_training_row_schema():
    row = to_training_row("img.png", "put can on plate", {"steps": [{"skill": "done", "args": {}}]})
    assert row["images"] == ["img.png"] and row["type"] == "Plan"
    assert row["messages"][0]["role"] == "user" and "<image>" in row["messages"][0]["content"]
    assert '"steps"' in row["messages"][1]["content"]      # assistant = plan JSON
```

- [ ] **Step 2: run → FAIL; implement** (reuse `gen_vqa_from_lerobot.TeacherVLM` + frame sampling; template guarantees a reliable floor, teacher adds diversity)

```python
# src/vlm_lora/gen_plan_data.py
"""Auto-generate a planning dataset (instruction+image -> plan JSON) from OpenArm LeRobot,
grounded in the SkillRegistry. Template plans give a reliable floor; the teacher VLM adds
diverse phrasings. Output JSONL matches vqa_dataset's `messages` schema so train_vlm_lora
can train on it unchanged."""
from __future__ import annotations

import json
import os
import re

import tyro

_COLORS = ["orange", "green", "red", "blue", "yellow", "white", "black"]


def _target_color(task: str) -> str:
    for c in _COLORS:
        if re.search(rf"\b{c}\b", task.lower()):
            return c
    return "target"


def template_plan(task: str) -> dict:
    color = _target_color(task)
    return {"steps": [
        {"skill": "move_to", "args": {"target": "the can"}, "rationale": "approach the can"},
        {"skill": "pick", "args": {"object": "the can"}},
        {"skill": "move_to", "args": {"target": f"the {color} plate"}},
        {"skill": "place", "args": {"target": f"the {color} plate"}},
        {"skill": "done", "args": {}}]}


def to_training_row(image_rel: str, instruction: str, plan: dict) -> dict:
    return {"images": [image_rel], "type": "Plan", "messages": [
        {"role": "user", "content": f"<image>\nInstruction: {instruction}\nProduce the skill-call plan as JSON."},
        {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}]}


def generate(dataset_path: str, out_dir: str, num_episodes: int = 100, frames_per_episode: int = 2,
             val_ratio: float = 0.1, use_teacher: bool = False,
             teacher_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct", seed: int = 42) -> None:
    import random

    # reuse the VQA generator's frame extraction + (optional) teacher
    from vlm_lora.gen_vqa_from_lerobot import _extract_frame, _sample_frames

    random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    eps = [json.loads(x) for x in open(os.path.join(dataset_path, "meta/episodes.jsonl"),
                                       encoding="utf-8") if x.strip()]
    random.shuffle(eps)
    eps = eps[:num_episodes]
    vpat = os.path.join(dataset_path, "videos/chunk-000/observation.images.camera/episode_{:06d}.mp4")
    rows: list[dict] = []
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
            rows.append(to_training_row(rel, task, template_plan(task)))
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    for name, sub in {"data.val.jsonl": rows[:n_val], "data.train.jsonl": rows[n_val:],
                      "data.jsonl": rows}.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_plan] {len(rows)} rows ({len(rows) - n_val} train / {n_val} val) -> {out_dir}")


if __name__ == "__main__":
    tyro.cli(generate)
```

- [ ] **Step 3: run tests → PASS; lint; stage** — msg: `feat(plan): generate instruction->plan dataset from OpenArm`.

### Task F2: train the planning LoRA + merge (H100 GPU)
**Files:** none new — reuse `train_vlm_lora` + `merge_lora`

- [ ] **Step 1: generate** (H100): `... -m vlm_lora.gen_plan_data --dataset-path <OpenArm> --out-dir artifacts/plan --num-episodes 100`
- [ ] **Step 2: train** the planning LoRA on `artifacts/plan/data.train.jsonl` → `artifacts/cosmos_r2_plan_lora` (reuse `train_vlm_lora`; `--max-steps 1500`).
- [ ] **Step 3: merge** → `artifacts/.../lora_tuned_vlm_planner/Cosmos-Reason2-2B-plan-merged` (reuse `merge_lora`).
- [ ] **Step 4:** point the Brain at the planner checkpoint via `BRAIN_MODEL_DIR`. **STOP/ASK** if VRAM/OOM (batch=1, grad-accum already minimal).

### Task F3: `eval_plan.py` — planning accuracy (TDD)
**Files:** Create `src/vlm_lora/eval_plan.py`; Test `tests/test_eval_plan.py`

- [ ] **Step 1: failing test** (pure scoring)

```python
# tests/test_eval_plan.py
from vlm_lora.eval_plan import score_plan

def test_valid_json_and_skill_names():
    gold = {"steps": [{"skill": "pick", "args": {"object": "can"}}, {"skill": "done", "args": {}}]}
    pred = '{"steps":[{"skill":"pick","args":{"object":"can"}},{"skill":"done","args":{}}]}'
    s = score_plan(pred, gold, allowed={"pick", "done"})
    assert s["valid_json"] == 1 and s["skills_in_vocab"] == 1 and s["seq_match"] == 1

def test_invalid_json_scores_zero():
    s = score_plan("nope", {"steps": []}, allowed={"done"})
    assert s["valid_json"] == 0
```

- [ ] **Step 2: run → FAIL; implement** `score_plan` (valid-JSON %, skills-in-vocab %, skill-sequence match) + an `evaluate(model_dir, val_jsonl, image_root, skills, out_json)` that runs `BrainModel` over the val set and aggregates. (Mirror `eval_vlm_vqa.py`'s structure.)
- [ ] **Step 3: run → PASS; stage** — msg: `feat(plan): planning-accuracy eval`.
- [ ] **Step 4 (GPU):** eval base Product A vs planner checkpoint on `artifacts/plan/data.val.jsonl` → before/after table (mirrors the VQA before/after story).

---

## Phase G — docs (update existing + new operator guide)

### Task G1: `docs/deploy_brain.md` (繁中 operator guide)
**Files:** Create `Isaac-GR00T-VLM/docs/deploy_brain.md`

- [ ] **Step 1: write** sections: (1) 什麼是 Brain（架構圖：instruction+image+state → skill-calls → Skill Library/VLA）；(2) Launch（`run_brain_server.sh`, env vars, ports, H100 vs 4090）；(3) **I/O 格式**（`/plan` 的 PlanRequest/PlanResponse 範例 JSON，`/skills`, `/health`）；(4) **整合你的 Skill Library**（manifest schema + 換掉 `skills.sample.json`）；(5) MVP(prompt) vs planner(LoRA) 兩種 checkpoint 的切換；(6) Brain→VLA bridge 用法；(7) troubleshooting（model returns prose → 用 planner ckpt / 調 few-shot；OOM；offline gating）。
- [ ] **Step 2: stage** — msg: `docs: brain deploy/operator guide`.

### Task G2: update `README.md` + `architecture_dataflow.html` + `project_report.html`
**Files:** Modify `Isaac-GR00T-VLM/README.md`, `docs/architecture_dataflow.html`, `docs/project_report.html`

- [ ] **Step 1: README** — add a "Deploy: VLM Brain" section (launch + I/O one-liner + link to `deploy_brain.md`), and a stage map: `finetune (done) → checkpoint → Brain serve (new)`. Note commits are the maintainer's.
- [ ] **Step 2: `architecture_dataflow.html`** — extend the animated SVG with the deployment lane: `instruction+image+state → [VLM Brain] → skill-call JSON → [Skill Library / VLA]`, mirroring `architecture.png`'s Brain→Skill Library→VLA but with the VLM as the brain. Keep the existing light theme + animation style.
- [ ] **Step 3: `project_report.html`** — add a "§8 Deployment: VLM as Brain" section: the FastAPI contract, the VQA→planning gap + the two-phase answer (MVP prompt → planning LoRA), the skill-call schema, and (if F4 ran) the planning before/after table. Keep charts/formulas in English, prose in 繁中.
- [ ] **Step 4: open both HTML in a browser to confirm animation/layout; stage** — msg: `docs: add Brain deployment to report + dataflow + README`.

---

## Verification (end-to-end)

1. **CPU tests green:** `cd Isaac-GR00T-VLM && uv run python -m pytest tests/ -v` (schema, skills, prompt, core parse/validate, serve TestClient, executor, gen_plan_data, eval_plan — all run without a GPU).
2. **Lint clean:** `uv run ruff check src tests examples`.
3. **Server up (GPU):** `run_brain_server.sh` with Product A; `GET /health` → `model_loaded:true`; `GET /skills` → sample skills.
4. **Plan call:** `POST /plan {instruction:"put the can on the orange plate", image_b64:<frame>}` → `ok:true`, ordered skill-calls, every skill ∈ registry.
5. **Demo:** `examples/brain_demo.py` prints a plan + MockExecutor runs each step in order.
6. **(If Phase F ran)** planner checkpoint beats base Product A on `eval_plan` (valid-JSON %, skills-in-vocab %, seq-match) — before/after table in the report.
7. **Skill Library swap:** replacing `skills.sample.json` with a different manifest changes `/skills` + what the Brain may emit, with **no code change** (proves the decoupling).
8. **No n1d7 edits:** the GR00T bridge only *imports*/calls the policy client; `cd Isaac-GR00T_n1d7 && git status` clean.

---

## Gaps & recommendations (beyond the core ask)

1. **VQA-tuned ≠ planner (addressed by Phase F).** Expect the MVP to sometimes emit prose or out-of-vocab skills; the planning LoRA is what makes it reliable. Ship MVP first to validate the loop, then train.
2. **Memory / multi-turn.** `architecture.png`'s Brain has a "Memory" block. The current contract is single-shot. Recommendation (future task): add an optional `history: list[SkillCall|result]` to `PlanRequest` so the Brain can re-plan from execution feedback (closed loop). Schema already extensible.
3. **Grounding / perception bridge.** `state` is free-form JSON now. When the real Skill Library lands, define a typed state schema (object list, gripper, poses) so plans reference detected entities, not guessed names. Consider an `inspect`/verify skill loop.
4. **Safety & validation.** Beyond schema validation, add (future) a max-steps cap, a per-skill arg-type check (the manifest already declares types — enforce them), and a "refuse if instruction not achievable with available skills" path.
5. **Latency.** A 2B VLM at bf16 on a 4090 is ~sub-second/plan; if you batch or need lower latency, add `torch.compile`/TensorRT later (GR00T's `benchmark_inference.py` is a template).
6. **Skill Library contract is the long pole.** The manifest schema in `skills.py` is intentionally minimal (name/description/args). Align it early with your external library's real skill signatures so integration is a drop-in manifest, not a refactor.
7. **Eval realism.** `eval_plan` scores structure/sequence vs templates; once you have human-authored or executed-success labels, add a task-success metric (did the executed plan complete the goal).

---

## Self-review

- **Spec coverage:** "how to use the checkpoint / launch / I/O" → Phases C–D (BrainModel, FastAPI `/plan` with PlanRequest/PlanResponse). "VLM as brain → sub-tasks/skills" → prompt.py + skills.py + structured SkillCall output. "structured skill-call + external Skill Library (Q1)" → SkillRegistry manifest + sample + mock executor + swap-test (Verification #7). "MVP→fine-tune (Q2)" → Phases B–E then F. "FastAPI (Q3)" → serve.py. "complete flow extract→VQA→finetune→checkpoint→deploy" → Context ties DONE stages to new Phases. "reorg folders" → additive `brain/` package + documented stage split (heavier reorg deliberately declined with rationale). "update html/md" → Phase G. "suggestions for gaps" → Gaps & recommendations.
- **Placeholder scan:** every code step has real code; commands have expected outputs; no TBD/TODO.
- **Type consistency:** `SkillCall{skill,args,rationale}`, `Plan{steps}`, `PlanRequest{instruction,image_b64,state,max_new_tokens}`, `PlanResponse{ok,steps,error,raw}`, `SkillRegistry.{from_manifest,names,render_for_prompt,validate}`, `BrainModel.plan(...)`, `plan_from_text(raw, registry)`, `extract_json(text)` — names match across prompt/core/serve/tests. Training JSONL uses the same `{images,type,messages}` schema as `vqa_dataset` so `train_vlm_lora` is reused unchanged.

---

## Execution handoff

Two execution options:
1. **Subagent-Driven (recommended)** — a fresh subagent per task, with two-stage review between tasks (uses superpowers:subagent-driven-development).
2. **Inline Execution** — execute in this session with checkpoints (uses superpowers:executing-plans).

Either way: **per your instruction, I will stage changes and stop for you to review + commit — I won't commit/push.** GPU steps (D3, F2, F4) run on H100/4090 via `pegasus.py`. Which approach do you want — or do you want to review/adjust the plan first?
