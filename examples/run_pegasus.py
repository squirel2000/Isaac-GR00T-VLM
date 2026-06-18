#!/usr/bin/env python3
"""Detached launcher: run examples/finetune_cosmos_r2_lora.sh on the H100 (Pegasus),
surviving disconnect via setsid+nohup. Reuses the main repo's scripts/common/pegasus.py.

Run from a machine that has the main IsaacLab-GR00T checkout (for pegasus.py + creds):
    python vlm_lora_finetune/examples/run_pegasus.py \
        --remote-repo /data/VLA/tingying/vlm_lora_finetune \
        --dataset-path artifacts/vqa/data.train.jsonl --image-root artifacts/vqa \
        --output-dir artifacts/cosmos_r2_lora --max-steps 2000 --hf-token "$HF_TOKEN"

Tail the log afterwards:
    python scripts/common/pegasus.py run "tail -n 200 -f <state>/run.log"
"""
import argparse
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
for _up in _HERE.parents:  # find the main repo's scripts/common/pegasus.py
    if (_up / "scripts" / "common" / "pegasus.py").exists():
        sys.path.insert(0, str(_up / "scripts" / "common"))
        break
import pegasus as pg  # noqa: E402

WRAPPER = """#!/bin/bash
STATE="__STATE__"
echo RUNNING > "$STATE/status"
exec >> "$STATE/run.log" 2>&1
echo "[wrapper] $(date '+%F %T') start"
cd "__REPO__" || { echo FAILED_CD > "$STATE/status"; exit 12; }
__ENV__bash examples/finetune_cosmos_r2_lora.sh
rc=$?
echo "[wrapper] exit=$rc"
if [ $rc -eq 0 ]; then echo DONE > "$STATE/status"; else echo "FAILED:$rc" > "$STATE/status"; fi
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-repo", required=True, help="repo dir on the H100")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--image-root", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--state-root", default=None)
    args = ap.parse_args()

    s = pg.connect()
    runid = args.output_dir.strip("/").replace("/", "_")
    state = f"{args.state_root or (args.remote_repo + '/artifacts/runs')}/{runid}"
    pg.run(s, f"mkdir -p {state}")

    env = ""
    if args.hf_token:
        env += f"HF_TOKEN={args.hf_token} "
    env += f"CUDA_VISIBLE_DEVICES={args.cuda} DATASET_PATH={args.dataset_path} "
    if args.image_root:
        env += f"IMAGE_ROOT={args.image_root} "
    env += f"OUTPUT_DIR={args.output_dir} MAX_STEPS={args.max_steps} "

    wrapper = WRAPPER.replace("__STATE__", state).replace("__REPO__", args.remote_repo).replace(
        "__ENV__", env
    )
    tmp = _HERE.parent / "_wrapper.sh"
    tmp.write_text(wrapper, encoding="utf-8")
    pg.put_file(s, str(tmp), f"{state}/wrapper.sh")
    pg.run(
        s,
        f"cd {state} && : > run.log && nohup setsid bash wrapper.sh >/dev/null 2>&1 & "
        "echo started detached",
    )
    print(f"[run_pegasus] launched (detached). state={state}")
    print(f'[run_pegasus] tail: python scripts/common/pegasus.py run "tail -n 200 -f {state}/run.log"')


if __name__ == "__main__":
    main()
