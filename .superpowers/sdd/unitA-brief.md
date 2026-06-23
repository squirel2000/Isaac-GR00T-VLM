## Global Constraints

- **Dataset (verified):** `/home/asus/Gits/IsaacLab-GR00T/datasets/OpenArm_CanSorting_MultiTask_Sim_dataset_O6_0403` — 2000 episodes, 450 frames @ 30 fps, single camera `observation.images.camera`, tasks `place the can on the {orange|green} plate`. Video pattern `videos/chunk-000/observation.images.camera/episode_{:06d}.mp4`.
- **GPU:** one RTX 4090, 24 GB. Teacher 8B bf16 ≈ 17 GB (fits); 30B-A3B MoE ≈ 62 GB (does NOT fit — never use on 4090). `Qwen/Qwen3-VL-4B-Instruct` is already cached as a lighter fallback.
- **Baseline VLA for swap (verified):** `…/artifacts/checkpoints/gr00t/N1_7_fft_0614_150k_lr1e4_no_tune_visual` — `select_layer=16`, `vl_self_attention_cfg.num_layers=4`, `diffusion_model_cfg.num_layers=32`, `backbone_embedding_dim=2048`.
- **7 question types (verbatim):** `Attribute, Mechanics, Reasoning, Spatial, Summary, Temporal, Trajectory` (`gen_vqa_from_lerobot.py:19`).
- **6 phases (verbatim, user-specified):** `approach, grasp, pick, move_to_plate, place, retract`.
- **No `git commit`/`push`.** Stage only.

---

### Task 1: Phase-based frame sampling

**Files:**
- Modify: `src/vlm_lora/gen_vqa_from_lerobot.py` (add `_PHASES`, `_PHASE_FRACS`, `_PHASE_ORDER`, `_sample_phase_frames`)
- Test: `tests/test_phase_sampling.py`

**Interfaces:**
- Produces: `_PHASES: tuple[str,...]` (the 6 names, in order); `_sample_phase_frames(length: int, phases=_PHASES) -> list[tuple[int,str]]` returning `(frame_index, phase_name)`, indices non-decreasing and in `[0, length-1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase_sampling.py
from vlm_lora.gen_vqa_from_lerobot import _PHASES, _sample_phase_frames

def test_six_phases_in_order_and_in_range():
    frames = _sample_phase_frames(450)
    assert [p for _, p in frames] == list(_PHASES)
    assert len(_PHASES) == 6
    idxs = [i for i, _ in frames]
    assert idxs == sorted(idxs)              # non-decreasing across the trajectory
    assert all(0 <= i <= 449 for i in idxs)  # in range
    assert idxs[0] < idxs[-1]                # approach earlier than retract

def test_short_episode_does_not_crash():
    frames = _sample_phase_frames(3)
    assert len(frames) == 6
    assert all(0 <= i <= 2 for i, _ in frames)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T-VLM && uv run pytest tests/test_phase_sampling.py -v`
Expected: FAIL — `ImportError: cannot import name '_PHASES'`.

- [ ] **Step 3: Add the sampling code**

Insert after `_TYPES = [...]` (line 19) in `gen_vqa_from_lerobot.py`:

```python
# Six distinct manipulation phases of the can-sort episode, in temporal order.
_PHASES = ("approach", "grasp", "pick", "move_to_plate", "place", "retract")
_PHASE_FRACS = {
    "approach": 0.12, "grasp": 0.30, "pick": 0.45,
    "move_to_plate": 0.62, "place": 0.80, "retract": 0.96,
}
_PHASE_ORDER = {p: i for i, p in enumerate(_PHASES)}


def _sample_phase_frames(length: int, phases: tuple[str, ...] = _PHASES) -> list[tuple[int, str]]:
    """Map each named manipulation phase to one representative frame index."""
    out = []
    for ph in phases:
        frac = _PHASE_FRACS.get(ph, 0.5)
        idx = min(length - 1, max(0, int(frac * (length - 1))))
        out.append((idx, ph))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_phase_sampling.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Stage**

```bash
git add src/vlm_lora/gen_vqa_from_lerobot.py tests/test_phase_sampling.py
# suggested: feat(vqa): add 6-phase frame sampling (approach…retract)
```

---

### Task 2: Phase-aware QA templates covering all 7 types

**Files:**
- Modify: `src/vlm_lora/gen_vqa_from_lerobot.py` (`_next_action_answer`, rewrite `template_qa_pairs`)
- Test: `tests/test_phase_sampling.py` (extend)

**Interfaces:**
- Consumes: `_PHASE_ORDER` (Task 1).
- Produces: `template_qa_pairs(task: str, phase: str) -> list[dict]` where the **union of `type` values across all 6 phases == the 7 types**; per-phase logic sets `placed`/`grasped`/`moving` correctly.

- [ ] **Step 1: Write the failing test**

```python
def test_templates_cover_all_seven_types_across_phases():
    from vlm_lora.gen_vqa_from_lerobot import _PHASES, _TYPES, template_qa_pairs
    seen = set()
    for ph in _PHASES:
        for qa in template_qa_pairs("place the can on the green plate", ph):
            assert {"type", "question", "answer"} <= set(qa)
            seen.add(qa["type"])
    assert set(_TYPES) <= seen   # all 7 PDF types present from templates alone

def test_placed_flag_follows_phase():
    from vlm_lora.gen_vqa_from_lerobot import template_qa_pairs
    def temporal_answer(ph):
        qs = [q for q in template_qa_pairs("place the can on the orange plate", ph)
              if q["question"].startswith("Has the can already been placed")]
        return qs[0]["answer"].lower()
    assert "yes" in temporal_answer("place")
    assert "yes" in temporal_answer("retract")
    assert "no" in temporal_answer("approach")
    assert "no" in temporal_answer("pick")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase_sampling.py -k templates_cover -v`
Expected: FAIL — current `template_qa_pairs` only emits 4 types (Summary/Trajectory/Attribute/Temporal).

- [ ] **Step 3: Replace `template_qa_pairs` (lines 29-55) with the phase-aware version**

```python
_NEXT = {
    "approach": "Reach toward the can and prepare to grasp it.",
    "grasp": "Close the dexterous hand around the can.",
    "pick": "Lift the can off the table.",
    "move_to_plate": "Carry the can toward the {color} plate.",
    "place": "Lower and release the can onto the {color} plate.",
    "retract": "Withdraw the arm back to its home pose.",
}


def _next_action_answer(phase: str, color: str) -> str:
    return _NEXT.get(phase, "Continue the task.").format(color=color)


def template_qa_pairs(task: str, phase: str) -> list[dict]:
    color = parse_target_color(task) or "target"
    i = _PHASE_ORDER.get(phase, 0)
    grasped = i >= _PHASE_ORDER["grasp"]
    placed = i >= _PHASE_ORDER["place"]
    moving = _PHASE_ORDER["pick"] <= i < _PHASE_ORDER["place"]
    phase_human = phase.replace("_", " ")
    return [
        {"type": "Summary",
         "question": "What is the dual-arm robot doing in this scene?",
         "answer": f"The robot is performing a can-sorting task: {task}."},
        {"type": "Trajectory",
         "question": "On which plate will the can be placed?",
         "answer": f"On the {color} plate."},
        {"type": "Attribute",
         "question": "What is the color of the target plate?",
         "answer": f"The target plate is {color}."},
        {"type": "Temporal",
         "question": "Which stage is shown: approach, grasp, pick, move, place, or retract?",
         "answer": f"The {phase_human} stage."},
        {"type": "Temporal",
         "question": "Has the can already been placed on the plate?",
         "answer": "Yes, the can has been placed." if placed
                   else "No, the can has not been placed yet."},
        {"type": "Mechanics",
         "question": "Is the dexterous hand currently holding the can?",
         "answer": ("No, the can has been released." if placed
                    else "Yes, the hand is grasping the can." if grasped
                    else "No, the hand has not grasped the can yet.")},
        {"type": "Reasoning",
         "question": "What should the robot do next?",
         "answer": _next_action_answer(phase, color)},
        {"type": "Spatial",
         "question": "Where is the can relative to the target plate?",
         "answer": ("The can is resting on the plate." if placed
                    else "The can is being carried toward the plate." if moving
                    else "The can is in front of the robot, away from the plate.")},
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase_sampling.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Stage**

```bash
git add src/vlm_lora/gen_vqa_from_lerobot.py tests/test_phase_sampling.py
# suggested: feat(vqa): phase-aware templates covering all 7 question types
```

---

### Task 3: Qwen3-VL-8B teacher + phase-grounded prompt + CLI wiring

**Files:**
- Modify: `src/vlm_lora/gen_vqa_from_lerobot.py` (`_TEACHER_PROMPT`, `TeacherVLM.generate_qa` signature, `generate()` defaults + phase loop)
- Test: `tests/test_phase_sampling.py` (extend — CPU-only, no model load)

**Interfaces:**
- Consumes: `_sample_phase_frames`, `template_qa_pairs` (Tasks 1-2).
- Produces: `generate(..., teacher_model="Qwen/Qwen3-VL-8B-Instruct", phase_mode=True, ...)`; `TeacherVLM.generate_qa(image, task, phase, max_new_tokens)`.

- [ ] **Step 1: Write the failing test** (asserts new defaults via signature introspection — no GPU needed)

```python
def test_generate_defaults_to_8b_teacher_and_phase_mode():
    import inspect
    from vlm_lora.gen_vqa_from_lerobot import generate
    sig = inspect.signature(generate)
    assert sig.parameters["teacher_model"].default == "Qwen/Qwen3-VL-8B-Instruct"
    assert sig.parameters["phase_mode"].default is True

def test_teacher_prompt_mentions_phase():
    from vlm_lora.gen_vqa_from_lerobot import _TEACHER_PROMPT
    assert "{phase}" in _TEACHER_PROMPT and "{task}" in _TEACHER_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase_sampling.py -k "teacher or defaults" -v`
Expected: FAIL — default is `Qwen/Qwen3-VL-30B-A3B-Instruct`; `phase_mode` param does not exist; `{phase}` not in prompt.

- [ ] **Step 3: Apply the edits**

3a. Replace `_TEACHER_PROMPT` (lines 90-95):

```python
_TEACHER_PROMPT = (
    "This is a robot manipulation scene. Task: '{task}'. This frame is captured during the "
    "'{phase}' stage of a can-sorting episode (stages in order: approach, grasp, pick, "
    "move_to_plate, place, retract). Generate 4 diverse question/answer pairs about THIS image "
    "covering varied types among: {types}. Keep answers short and grounded in what is visible. "
    "Reply ONLY as a JSON list of objects with keys type, question, answer."
)
```

3b. Update `TeacherVLM.generate_qa` signature + format call (line 118 / 121):

```python
    def generate_qa(self, image, task: str, phase: str, max_new_tokens: int = 512) -> list[dict]:
        import torch

        prompt = _TEACHER_PROMPT.format(task=task, phase=phase, types=", ".join(_TYPES))
```

3c. In `generate()` change the signature defaults (lines 142-143) and the sampling/teacher loop (lines 175-183):

```python
    teacher_model: str | None = "Qwen/Qwen3-VL-8B-Instruct",
    teacher_dtype: str = "bfloat16",
    use_template: bool = True,
    phase_mode: bool = True,
    max_new_tokens: int = 512,
```

```python
            frames = (
                _sample_phase_frames(length)
                if phase_mode
                else _sample_frames(length, frames_per_episode)
            )
            for fidx, phase in frames:
                img = _extract_frame(vpat.format(ei), fidx)
                if img is None:
                    continue
                rel = f"images/ep{ei:06d}_f{fidx:06d}_{phase}.png"
                img.save(os.path.join(out_dir, rel))
                qas = template_qa_pairs(task, phase) if use_template else []
                if teacher:
                    qas += teacher.generate_qa(img, task, phase, max_new_tokens)
```

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest tests/test_phase_sampling.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Stage**

```bash
git add src/vlm_lora/gen_vqa_from_lerobot.py tests/test_phase_sampling.py
# suggested: feat(vqa): Qwen3-VL-8B teacher with phase-grounded prompts
```

---

### Task 4: Update `configs/default.yaml` for the 4090 run

**Files:** Modify: `configs/default.yaml`

- [ ] **Step 1: Edit the `data` and `models` blocks** so the documented defaults match the 4090 layout (CLI flags still override):

```yaml
models:
  base_vlm: nvidia/Cosmos-Reason2-2B
  teacher_vlm: Qwen/Qwen3-VL-8B-Instruct      # 4090: dense 8B (~17GB) — 30B-A3B MoE does NOT fit
  teacher_vlm_fallback: Qwen/Qwen3-VL-4B-Instruct  # already cached, ~9GB

data:
  lerobot_dataset: /home/asus/Gits/IsaacLab-GR00T/datasets/OpenArm_CanSorting_MultiTask_Sim_dataset_O6_0403
  vqa_out: artifacts/vqa_6phase
  num_episodes: 150
  phase_mode: true            # six action phases (approach…retract) per episode
  val_ratio: 0.1
```

- [ ] **Step 2: Verify it still parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('configs/default.yaml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Stage** — `git add configs/default.yaml`  (suggested: `chore(config): point pipeline at 4090 dataset + 8B teacher`)

---

