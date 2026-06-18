from vlm_lora.gen_vqa_from_lerobot import compute_stats, parse_target_color, template_qa_pairs


def test_parse_color():
    assert parse_target_color("place the can on the orange plate") == "orange"
    assert parse_target_color("no color") is None


def test_template_types_and_phase():
    early = template_qa_pairs("place the can on the green plate", phase="early")
    late = template_qa_pairs("place the can on the green plate", phase="late")
    assert {"Summary", "Trajectory", "Attribute", "Temporal"}.issubset(
        {qa["type"] for qa in early}
    )
    te = next(q for q in early if q["type"] == "Temporal")["answer"]
    tl = next(q for q in late if q["type"] == "Temporal")["answer"]
    assert te != tl and all(q["question"] and q["answer"] for q in early)


def test_stats_counts_by_type():
    s = compute_stats([{"type": "Summary"}, {"type": "Summary"}, {"type": "Temporal"}])
    assert s["total"] == 3 and s["by_type"]["Summary"] == 2 and s["by_type"]["Temporal"] == 1
