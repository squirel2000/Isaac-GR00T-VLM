# Unit A Implementation Report

**Date:** 2026-06-23
**Branch:** develop (feature/LoRA_VLM outer repo)
**Implementer:** Claude Sonnet 4.6 (agent)

---

## Summary

All four tasks completed in strict TDD order. 6/6 tests pass.

---

## Task 1: Phase-based frame sampling

**What changed in `src/vlm_lora/gen_vqa_from_lerobot.py`:**
- Added `_PHASES = ("approach", "grasp", "pick", "move_to_plate", "place", "retract")` (tuple, 6 names in temporal order).
- Added `_PHASE_FRACS` dict mapping each phase to a fractional position in the episode.
- Added `_PHASE_ORDER` reverse-lookup dict `{phase: index}` used by Task 2.
- Added `_sample_phase_frames(length, phases=_PHASES) -> list[tuple[int,str]]` — maps each phase to one frame index, clamped to `[0, length-1]`.

**TDD evidence:**
- RED: `uv run --extra dev pytest tests/test_phase_sampling.py -v` → `ImportError: cannot import name '_PHASES'` (exit 2).
- GREEN after edit: `test_six_phases_in_order_and_in_range PASSED`, `test_short_episode_does_not_crash PASSED`.

---

## Task 2: Phase-aware QA templates covering all 7 types

**What changed in `src/vlm_lora/gen_vqa_from_lerobot.py`:**
- Replaced old `template_qa_pairs` (4 types: Summary/Trajectory/Attribute/Temporal) with phase-aware version returning 8 QA pairs covering all 7 types.
- Added `_NEXT` dict with per-phase next-action strings (with `{color}` placeholder).
- Added `_next_action_answer(phase, color)` helper.
- Per-phase flags: `grasped = i >= PHASE_ORDER["grasp"]`, `placed = i >= PHASE_ORDER["place"]`, `moving = pick <= i < place` — drive Mechanics, Spatial, and Temporal (placed) answers.

**TDD evidence (still under first RED run):**
- RED: `test_templates_cover_all_seven_types_across_phases` and `test_placed_flag_follows_phase` both failed at the same collection point (ImportError on `_PHASES`).
- GREEN after edit: both tests PASSED.

---

## Task 3: Qwen3-VL-8B teacher + phase-grounded prompt + CLI wiring

**What changed in `src/vlm_lora/gen_vqa_from_lerobot.py`:**
- `_TEACHER_PROMPT`: added `'{phase}' stage` context and stage order enumeration; format call updated to pass `phase=`.
- `TeacherVLM.generate_qa`: added `phase: str` parameter; prompt format call now includes `phase`.
- `generate()` signature: `teacher_model` default changed from `"Qwen/Qwen3-VL-30B-A3B-Instruct"` to `"Qwen/Qwen3-VL-8B-Instruct"`; added `phase_mode: bool = True`.
- Inner loop: replaces old `_sample_frames(length, frames_per_episode)` with `_sample_phase_frames(length) if phase_mode else _sample_frames(...)`. Filename now includes phase (`ep…_f…_{phase}.png`). `teacher.generate_qa` call now passes `phase`.

**TDD evidence:**
- RED: `test_generate_defaults_to_8b_teacher_and_phase_mode` failed (default was 30B; `phase_mode` param absent); `test_teacher_prompt_mentions_phase` failed (`{phase}` not in prompt).
- GREEN after edit: both tests PASSED.

---

## Task 4: Update `configs/default.yaml` for the 4090 run

**What changed:**
- `models.teacher_vlm`: `Qwen/Qwen3-VL-30B-A3B-Instruct` → `Qwen/Qwen3-VL-8B-Instruct` with comment explaining 4090 fit.
- `models.teacher_vlm_compare` key renamed to `models.teacher_vlm_fallback` → `Qwen/Qwen3-VL-4B-Instruct`.
- `data.lerobot_dataset`: H100 path → local `4090` path `/home/asus/Gits/IsaacLab-GR00T/datasets/OpenArm_CanSorting_MultiTask_Sim_dataset_O6_0403`.
- `data.vqa_out`: `artifacts/vqa` → `artifacts/vqa_6phase`.
- `data.num_episodes`: 100 → 150.
- `data.frames_per_episode: 3` removed; `data.phase_mode: true` added.

**Verification:** `uv run python -c "import yaml; yaml.safe_load(open('configs/default.yaml')); print('ok')"` → `ok`.

---

## Final test run

```
uv run --extra dev pytest tests/test_phase_sampling.py -v

============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0
collected 6 items

tests/test_phase_sampling.py::test_six_phases_in_order_and_in_range PASSED [ 16%]
tests/test_phase_sampling.py::test_short_episode_does_not_crash PASSED   [ 33%]
tests/test_phase_sampling.py::test_templates_cover_all_seven_types_across_phases PASSED [ 50%]
tests/test_phase_sampling.py::test_placed_flag_follows_phase PASSED      [ 66%]
tests/test_phase_sampling.py::test_generate_defaults_to_8b_teacher_and_phase_mode PASSED [ 83%]
tests/test_phase_sampling.py::test_teacher_prompt_mentions_phase PASSED  [100%]

============================== 6 passed in 0.03s ===============================
```

---

## Files changed

1. `src/vlm_lora/gen_vqa_from_lerobot.py` — Tasks 1, 2, 3.
2. `configs/default.yaml` — Task 4.
3. `tests/test_phase_sampling.py` — New file, 6 tests.

---

## Self-review

**Completeness:** All 4 tasks implemented; all 7 interfaces from the brief are present.

**YAGNI:** No extra functionality added. `_PHASE_ORDER` is used by `template_qa_pairs` and could be considered an internal detail; it is also exported so tests can import it if needed. The `_NEXT` dict and `_next_action_answer` helper cleanly separate phase→text logic from the larger `template_qa_pairs` function.

**Names:** Match the brief verbatim for all public names. `phase_human` is a one-line local in `template_qa_pairs`; fine.

**Test quality:** Tests are pure-Python, CPU-only (no model load), use only public module imports. `test_short_episode_does_not_crash` covers the edge case of very short videos. `test_placed_flag_follows_phase` validates the phase-boundary logic via representative phases (approach, pick below threshold; place, retract above).

**Concerns:**

1. **`frames_per_episode` parameter is now dead code in the default path** (`phase_mode=True` skips it). It remains in the signature for backward compatibility when `--phase-mode false` is passed, but the config no longer exposes it. Not a bug; intentional.

2. **`_NEXT["approach"]` and `_NEXT["grasp"]` contain no `{color}` placeholder.** `_next_action_answer` calls `.format(color=color)` unconditionally on all values. This works fine (no-op when `{color}` is absent), but a future editor could accidentally add `{color}` to one of those strings and not notice the branch-conditional behavior.

3. **`test_short_episode_does_not_crash` with `length=3`**: all 6 phases map to indices in `{0,1,2}` and the monotonicity property (`idxs == sorted(idxs)`) holds because `_PHASE_FRACS` are strictly increasing (0.12 < 0.30 < … < 0.96). The `_sample_phase_frames` function correctly uses `min(length-1, max(0, int(frac * (length-1))))`. For length=3 the indices are `[0, 0, 0, 1, 2, 2]` — non-decreasing but with ties, which the test allows (it checks `==sorted`, not `strictly increasing`).

4. **`configs/default.yaml` key rename** (`teacher_vlm_compare` → `teacher_vlm_fallback`) may break any downstream script that reads `models.teacher_vlm_compare`. No such consumer was found in the repo, but this should be checked at integration time.

---

## Status

DONE — all tasks complete, 6/6 tests passing, `git add -A` staged.

---

## Fix pass (2026-06-23)

Applied 5 small fixes (I-1, I-2, M-1 + length<=0 coverage, M-3, M-4).

```
uv run pytest tests/test_phase_sampling.py -v

============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0
collected 8 items

tests/test_phase_sampling.py::test_six_phases_in_order_and_in_range PASSED [ 12%]
tests/test_phase_sampling.py::test_short_episode_does_not_crash PASSED   [ 25%]
tests/test_phase_sampling.py::test_zero_length_returns_empty PASSED      [ 37%]
tests/test_phase_sampling.py::test_templates_cover_all_seven_types_across_phases PASSED [ 50%]
tests/test_phase_sampling.py::test_placed_flag_follows_phase PASSED      [ 62%]
tests/test_phase_sampling.py::test_generate_defaults_to_8b_teacher_and_phase_mode PASSED [ 75%]
tests/test_phase_sampling.py::test_teacher_prompt_mentions_phase PASSED  [ 87%]
tests/test_phase_sampling.py::test_legacy_late_phase_is_placed PASSED    [100%]

============================== 8 passed in 0.03s ===============================
```

`uv run python -c "import yaml; yaml.safe_load(open('configs/default.yaml')); print('ok')"` → `ok`

`uv.lock` was rewritten by `uv run`; unstaged with `git restore --staged uv.lock`.
