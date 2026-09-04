#!/bin/bash
#SBATCH -J sharpa_convert
# ⚠️ CPU 分区，**不要**加 --gres=gpu。这一步只做 mjpeg 解码 + 视频重编码。
#    走 shared 而非 serial_requeue：本作业不能 checkpoint，被抢占就要从头再来。
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-08:00:00
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_convert_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_convert_%j.err

# 自采 teleop HDF5 → LeRobot 数据集。
# 详见 examples/sharpa/convert_sharpa_data_to_lerobot.py 的模块 docstring。
#
# 用法：  sbatch scripts/sharpa_convert.sbatch <train|tactile|both> [task]
#
# train 和 tactile 是两个独立的数据集，可以同时投两个作业并行跑 ——
# 段选取和重采样网格都是 HDF5 的纯函数，两边算出来的 episode/帧顺序必然一致。
#
# 原始 HDF5 只读，不会被改动。

set -euo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
OPENPI=$ROOT/code/Tactile/openpi
PY=$OPENPI/.venv/bin/python

ONLY=${1:-both}
TASK=${2:-pick_up_the_egg}
# 100 条里只有 38 条两路数据都完好，见转换器的 UnusableEpisode docstring。
# 换任务时用 EXPECT=<n> 覆盖，或 EXPECT=0 关掉这个断言（不推荐）。
EXPECT=${EXPECT:-38}

echo "===== convert $TASK ($ONLY) at $(date -Is) on $(hostname) (job ${SLURM_JOB_ID:-none}) ====="
cd "$OPENPI"

"$PY" examples/sharpa/convert_sharpa_data_to_lerobot.py \
    --raw-root "$ROOT/data/tactile-steering-data/$TASK" \
    --out-root "$ROOT/data/sharpa_lerobot" \
    --task-dir "$TASK" \
    --only "$ONLY" \
    --expect-episodes "$EXPECT" \
    --image-writer-processes 12 \
    --image-writer-threads 4 \
    --overwrite

# 训练集转完就顺手把 norm stats 算掉 —— 同样是纯 CPU 活，而且必须在 $OPENPI 下跑：
# TrainConfig.assets_dirs 是 "./assets/<name>"，相对 cwd，train.py 也是从这里读的。
if [ "$ONLY" != "tactile" ]; then
    echo; echo "===== norm stats ====="
    JAX_PLATFORMS=cpu OPENPI_DATA_HOME=$ROOT/.cache/openpi \
        "$PY" scripts/compute_norm_stats.py --config-name sharpa_egg
fi

echo "===== done at $(date -Is) ====="
du -sh "$ROOT/data/sharpa_lerobot/$TASK" "$ROOT/data/sharpa_lerobot/${TASK}_tactile" 2>/dev/null || true
