"""Swap merged VLM (Product A) into a trained VLA checkpoint -> swapped VLA.

Keeps action head; replaces only backbone (VLM). Respects select_layer pruning by key
intersection. Pure safetensors (no GR00T import).
"""

import shutil
from pathlib import Path

import tyro


def build_swapped_state_dict(vla_sd, merged_sd, prefix="backbone.model."):
    new_sd = dict(vla_sd)
    st = {"copied": 0, "shape_mismatch": 0, "missing_in_merged": 0, "backbone_keys": 0}
    for k in vla_sd:
        if not k.startswith(prefix):
            continue
        st["backbone_keys"] += 1
        sub = k[len(prefix) :]
        src = merged_sd.get(sub, merged_sd.get("model." + sub))
        if src is None:
            st["missing_in_merged"] += 1
        elif src.shape != vla_sd[k].shape:
            st["shape_mismatch"] += 1
        else:
            new_sd[k] = src.to(vla_sd[k].dtype)
            st["copied"] += 1
    return new_sd, st


def _load_sd(ckpt_dir):
    from safetensors.torch import load_file

    files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {ckpt_dir}")
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


def swap(vla_ckpt: str, merged_vlm: str, out_dir: str) -> None:
    new_sd, st = build_swapped_state_dict(_load_sd(vla_ckpt), _load_sd(merged_vlm))
    print(f"[swap] stats: {st}")
    assert st["copied"] > 0, "no backbone weights copied — check prefixes/shapes"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(vla_ckpt).iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)
    from safetensors.torch import save_file

    save_file(new_sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    idx = out / "model.safetensors.index.json"
    if idx.exists():
        idx.unlink()
    print(f"[swap] swapped VLA written to {out_dir}")


if __name__ == "__main__":
    tyro.cli(swap)
