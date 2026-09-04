#!/bin/bash
#SBATCH -J sharpa_train
# ⚠️ 卡型必须钉死。只写 --gres=gpu:1 会抢到 nvidia_a100_3g.20gb 的 MIG 切片，
#    20GB 装不下 pi0.5 的 LoRA 微调。
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 16
#SBATCH --mem 128G
# gpu_requeue 的上限就是 3 天（PreemptMode=REQUEUE）。顶满，靠 --resume 跨重排续跑。
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_train_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_train_%j.err

# pi05_base → sharpa_egg 微调。
#
# 用法：  sbatch scripts/sharpa_train.sbatch <exp_name> [额外的 train.py 参数...]
# 例：    sbatch scripts/sharpa_train.sbatch egg_10k --num-train-steps=10000
#
# gpu_requeue 会被抢占，所以必须能续跑。openpi **不会**自动续跑：
# checkpoints.initialize_checkpoint_dir 在目录已存在且既没 --overwrite 也没 --resume 时
# 直接 FileExistsError。所以本脚本恒传 --resume ——
#   目录不存在（首次投递）     -> 正常新建，--resume 无副作用
#   目录存在但还没落过 checkpoint -> 脚本自己 "Aborting resume"，从头开始
#   目录存在且有 checkpoint（被抢占后重排）-> 接着最新的 checkpoint 跑
# 要推倒重来就换一个 exp_name，或者手动删 <ckpt_root>/sharpa_egg/<exp_name>。
#
# 单卡。两卡在 gpu_requeue 上排不动，而且 batch 32 单卡装得下的话本来也没必要拆。
#
# ⚠️ XLA_PYTHON_CLIENT_PREALLOCATE=false：JAX 默认一启动就吃掉 75% 显存，
#    不关掉的话 nvidia-smi 永远显示 ~60GB，看不出真实用量。关掉后按需分配，
#    下面的采样器读到的就是真数字 —— 这是「显存有没有浪费、能不能加大 batch」
#    唯一能回答的方式，代价是极小的分配开销（训练循环形状固定，不会反复重分配）。
#
# 卡型备选（H100 排不上时换成这一行）：
#   #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

set -euo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
OPENPI=$ROOT/code/Tactile/openpi
PY=$OPENPI/.venv/bin/python

EXP=${1:?usage: sbatch scripts/sharpa_train.sbatch <exp_name> [train.py args...]}
shift

echo "===== attempt at $(date -Is) on $(hostname) (job $SLURM_JOB_ID, restarts=${SLURM_RESTART_COUNT:-0}) ====="
cd "$OPENPI"
export OPENPI_DATA_HOME=$ROOT/.cache/openpi
export HF_HOME=$ROOT/hf_home
export XLA_PYTHON_CLIENT_PREALLOCATE=false

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 显存采样器：每 60 秒报一次当前/峰值，用来判断还能不能加大 batch。
( PEAK=0
  while :; do
      U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$U" -gt "$PEAK" ] && PEAK=$U
      echo "[mem] $(date +%H:%M:%S) used=${U}MiB peak=${PEAK}MiB"
      sleep 60
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# save_interval / keep_period 由调用方传入（见下面的 epoch 对齐说明）。
"$PY" scripts/train.py sharpa_egg --exp-name="$EXP" --resume \
    --no-wandb-enabled "$@"
