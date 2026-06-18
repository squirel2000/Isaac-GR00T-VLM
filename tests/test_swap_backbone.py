import torch

from vlm_lora.swap_backbone import build_swapped_state_dict


def test_copies_matching_only():
    vla = {
        "backbone.model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "backbone.model.visual.patch_embed.proj.weight": torch.zeros(2, 2),
        "action_head.diffusion.layer.weight": torch.ones(3, 3),
    }
    merged = {
        "language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "language_model.layers.5.self_attn.q_proj.weight": torch.ones(4, 4),
        "visual.patch_embed.proj.weight": torch.ones(2, 2),
    }
    new_sd, st = build_swapped_state_dict(vla, merged, prefix="backbone.model.")
    assert torch.equal(
        new_sd["backbone.model.language_model.layers.0.self_attn.q_proj.weight"], torch.ones(4, 4)
    )
    assert torch.equal(new_sd["action_head.diffusion.layer.weight"], torch.ones(3, 3))
    assert st["copied"] == 2 and st["missing_in_merged"] == 0


def test_shape_mismatch_skipped():
    new_sd, st = build_swapped_state_dict(
        {"backbone.model.x.weight": torch.zeros(4, 4)},
        {"x.weight": torch.ones(2, 2)},
        prefix="backbone.model.",
    )
    assert torch.equal(new_sd["backbone.model.x.weight"], torch.zeros(4, 4)) and st[
        "shape_mismatch"
    ] == 1
