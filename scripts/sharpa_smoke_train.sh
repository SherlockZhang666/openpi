#!/bin/bash
# 200 步冒烟：只验管线能跑通，不验性能。
# 需要一块真 GPU —— 用 scripts/sharpa_smoke_train.sbatch 投，或者在已经 salloc 到的
# GPU 节点上直接跑本脚本。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python

exec "$PY" scripts/train.py sharpa_egg \
    --exp-name=smoke \
    --num-train-steps=200 \
    --batch-size=2 \
    --save-interval=200 \
    --no-wandb-enabled \
    --overwrite
