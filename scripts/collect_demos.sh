#!/bin/bash
# Collect a small dataset for testing the pipeline end-to-end.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
MUJOCO_GL=egl PYTHONPATH=. python -m rvt_lerobot.data.collect \
    --num "${NUM:-3}" \
    --out "${OUT:-data/demos}" \
    --task "${TASK:-put_block_in_box}" \
    --lang "${LANG_GOAL:-put the block in the box}"
