"""Resolve a HF model id to its local cached snapshot path.

Loading a gated model (e.g. nvidia/Cosmos-Reason2-2B) by bare id triggers transformers'
tokenizer hub probes (is_base_mistral -> model_info), which 401 without a token or raise
under HF_HUB_OFFLINE. Loading from the local snapshot *directory* sets _is_local=True and
skips those probes — so the cached model works with no token and no network.

Handles both the standard hub cache (<root>/hub/models--org--name/snapshots/<hash>) and the
legacy flat TRANSFORMERS_CACHE/HF_HOME layout (<root>/models--org--name/snapshots/<hash>).
"""

import glob
import os


def _cache_snapshot(name_or_path: str) -> str | None:
    folder = "models--" + name_or_path.replace("/", "--")
    roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
    ]
    seen = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        for base in (os.path.join(root, "hub", folder), os.path.join(root, folder)):
            for snap in sorted(glob.glob(os.path.join(base, "snapshots", "*"))):
                if os.path.isfile(os.path.join(snap, "config.json")):
                    return snap
    return None


def resolve_model_path(name_or_path: str) -> str:
    """Return a local dir for name_or_path.

    - If it is already a directory, return it unchanged.
    - If it is a HF id cached locally (standard or legacy layout), return the snapshot dir.
    - Otherwise return the id unchanged (will require network / a token to download).
    """
    if os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name_or_path, local_files_only=True)
    except Exception:
        pass
    snap = _cache_snapshot(name_or_path)
    return snap if snap else name_or_path
