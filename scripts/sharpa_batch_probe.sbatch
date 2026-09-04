#!/bin/bash
#SBATCH -J sharpa_probe
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH -c 32
#SBATCH --mem 256G
#SBATCH -t 0-01:30:00
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_probe_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_probe_%j.err

# 找 2x H100 上放得下的最大 batch size。
#
# ⚠️ 必须 XLA_PYTHON_CLIENT_PREALLOCATE=false。JAX 默认开机就吃掉 75% 显存，
#    nvidia-smi 会一直显示 ~60GB，跟实际用量无关 —— 不关掉这个，探测出来的数字全是假的。
#
# 每个 batch size 跑 30 步，后台每 2 秒采一次 nvidia-smi 取峰值。
# OOM 的那档会崩，用 || true 兜住继续试下一档。

set -uo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
OPENPI=$ROOT/code/Tactile/openpi
PY=$OPENPI/.venv/bin/python
SIZES=${*:-32 64 96 128}

cd "$OPENPI"
export OPENPI_DATA_HOME=$ROOT/.cache/openpi
export HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "===== batch probe at $(date -Is) on $(hostname) (job $SLURM_JOB_ID) ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for BS in $SIZES; do
    echo
    echo "########## batch_size=$BS ##########"
    LOG=$(mktemp)
    # 采样器：只记两块卡各自的峰值
    ( while :; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$LOG" &
    SAMPLER=$!

    timeout 900 "$PY" scripts/train.py sharpa_egg \
        --exp-name="probe_$BS" \
        --batch-size="$BS" \
        --num-train-steps=30 \
        --save-interval=1000000 \
        --no-wandb-enabled \
        --overwrite 2>&1 | grep -E "^Step |it/s|s/it|RESOURCE_EXHAUSTED|Out of memory|OOM" | tail -5
    RC=${PIPESTATUS[0]}

    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    echo "  exit=$RC  峰值显存: $(awk -F', ' '{if($2>m[$1])m[$1]=$2} END{for(i in m) printf "GPU%s=%dMiB ", i, m[i]}' "$LOG")"
    rm -f "$LOG"
    rm -rf "$ROOT/code/Tactile/ckpt/sharpa_pi05/sharpa_egg/probe_$BS"
done

echo
echo "===== done at $(date -Is) ====="
