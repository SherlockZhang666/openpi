#!/bin/bash
#SBATCH -J sharpa_smoke
# ⚠️ 卡型必须钉死，否则会抢到 nvidia_a100_3g.20gb 的 MIG 切片。
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 0-01:00:00
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_smoke_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_smoke_%j.err

# 200 步冒烟。只验「pi05_base 能加载 + 数据能喂进去 + loss 有限且在降 + checkpoint 落得下」，
# 不验性能。被抢占就重投，不做 resume（200 步而已）。

set -euo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
OPENPI=$ROOT/code/Tactile/openpi

echo "===== attempt at $(date -Is) on $(hostname) (job $SLURM_JOB_ID, restarts=${SLURM_RESTART_COUNT:-0}) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$OPENPI"
export OPENPI_DATA_HOME=$ROOT/.cache/openpi
export HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1

exec ./scripts/sharpa_smoke_train.sh
