#!/bin/bash
# Render the multi-view hero figure from the first collected episode.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
EPISODE="${1:-data/demos/train/put_block_in_box/all_variations/episodes/episode0}"
OUT="${2:-assets/virtual_views_hero.png}"
PYTHONPATH=. python -m rvt_lerobot.visualize.virtual_views \
    --episode "$EPISODE" --frame 0 --out "$OUT"
