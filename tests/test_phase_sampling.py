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
    idxs = [i for i, _ in frames]
    assert idxs == sorted(idxs)


def test_zero_length_returns_empty():
    assert _sample_phase_frames(0) == []


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


def test_generate_defaults_to_8b_teacher_and_phase_mode():
    import inspect
    from vlm_lora.gen_vqa_from_lerobot import generate
    sig = inspect.signature(generate)
    assert sig.parameters["teacher_model"].default == "Qwen/Qwen3-VL-8B-Instruct"
    assert sig.parameters["phase_mode"].default is True


def test_teacher_prompt_mentions_phase():
    from vlm_lora.gen_vqa_from_lerobot import _TEACHER_PROMPT
    assert "{phase}" in _TEACHER_PROMPT and "{task}" in _TEACHER_PROMPT


def test_legacy_late_phase_is_placed():
    from vlm_lora.gen_vqa_from_lerobot import template_qa_pairs
    ans = [q["answer"].lower() for q in template_qa_pairs("place the can on the green plate", "late")
           if q["question"].startswith("Has the can already been placed")][0]
    assert "yes" in ans
