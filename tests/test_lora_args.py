from vlm_lora.lora_args import VlmLoraConfig, build_lora_config


def test_defaults_target_llm_projections():
    lc = build_lora_config(VlmLoraConfig(dataset_path="x", output_dir="/tmp/o"))
    assert lc.r == 16 and lc.lora_alpha == 32 and lc.task_type == "CAUSAL_LM"
    assert {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(set(lc.target_modules))


def test_all_linear_and_custom():
    assert (
        build_lora_config(VlmLoraConfig("x", "/tmp/o", lora_target="all-linear")).target_modules
        == "all-linear"
    )
    assert set(
        build_lora_config(VlmLoraConfig("x", "/tmp/o", lora_target="q_proj,v_proj")).target_modules
    ) == {"q_proj", "v_proj"}
