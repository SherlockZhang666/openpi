#!/bin/bash
#SBATCH -J sharpa_validate
# ⚠️ CPU 分区，不要加 --gres=gpu。norm stats 和数据管线检查都不需要 GPU（JAX_PLATFORMS=cpu）。
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-02:00:00
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_validate_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_validate_%j.err

# 用 3 条 episode 把「转换 → 校验 → norm stats → 数据管线」整条链路走一遍，
# 再决定要不要花几个小时转全量 100 条。
#
# ⚠️ 登录节点 nproc=1（cgroup 限死），负载常年 20+，在上面测速度或者跑转换都是白费 ——
#    这个作业存在的理由就是把这一步搬到真有 CPU 的节点上。
#
# 产物全部写进一个临时目录并在开头清掉，不碰 $ROOT/data/sharpa_lerobot 的正式数据。
# SHARPA_DATASET_ROOT 指向那个临时目录，所以 sharpa_egg 配置、compute_norm_stats
# 和校验脚本看到的都是这 3 条数据。

set -euo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
OPENPI=$ROOT/code/Tactile/openpi
PY=$OPENPI/.venv/bin/python
TMP=$ROOT/data/sharpa_lerobot_validate
TASK=${1:-pick_up_the_egg}
N=${2:-3}

export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME=$ROOT/.cache/openpi
export HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1
export SHARPA_DATASET_ROOT=$TMP

echo "===== validate $TASK ($N episodes) at $(date -Is) on $(hostname), nproc=$(nproc) ====="
rm -rf "$TMP"
cd "$OPENPI"

echo; echo "########## 1. 转换 ##########"
time "$PY" examples/sharpa/convert_sharpa_data_to_lerobot.py \
    --raw-root "$ROOT/data/tactile-steering-data/$TASK" \
    --out-root "$TMP" \
    --task-dir "$TASK" \
    --only both \
    --limit-episodes "$N" \
    --image-writer-processes 12 \
    --image-writer-threads 4 \
    --overwrite

echo; echo "########## 2. norm stats ##########"
# 3 条数据的 q01/q99 当然不代表全量，这一步验的是「算得出来、不报错」。
# ⚠️ TrainConfig.assets_dirs 是 "./assets/<name>"，相对 **cwd**。所以这里换到 $TMP 再跑，
#    免得这 3 条数据算出来的 norm_stats.json 覆盖掉 $OPENPI/assets 里全量的那份。
cd "$TMP"
"$PY" "$OPENPI/scripts/compute_norm_stats.py" --config-name sharpa_egg

echo; echo "########## 3. 校验（schema / 腕部 round-trip / 数据管线 / norm stats）##########"
"$PY" "$OPENPI/examples/sharpa/verify_sharpa_dataset.py" \
    --root "$TMP/$TASK" \
    --raw-root "$ROOT/data/tactile-steering-data/$TASK" \
    --round-trip-episodes "$N"

echo; echo "########## 4. 体积（用来外推全量 100 条）##########"
du -sh "$TMP/$TASK" "$TMP/${TASK}_tactile"
echo "===== done at $(date -Is) ====="
