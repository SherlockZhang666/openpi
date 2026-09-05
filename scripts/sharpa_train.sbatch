#!/bin/bash
#SBATCH -J sharpa_train
# ⚠️ 卡型必须钉死。只写 --gres=gpu:1 会抢到 nvidia_a100_3g.20gb 的 MIG 切片，
#    20GB 装不下 pi0.5 的 LoRA 微调。
#
# ⚠️ 用 gpu（PreemptMode=OFF）而不是 gpu_requeue。2026-09-03 实测：在 gpu_requeue 上
#    5h45m 被抢占 21 次，只推进了 2430 步（纯计算约 1.8h），~3.2 倍开销 —— 每次重排都要重新
#    花 4-5 分钟加载模型 + JIT 编译，抢占间隔常常比这还短。
#
# ⚠️ 本账号没有非抢占的 H100。全集群 5 个带 H100 的分区里，kempner_h100 /
#    kempner_requeue / kempner_gpu_priority 都投不进去（unix 组里有 kempner_ydu_lab，
#    但 SLURM 没有对应 association），剩下 gpu_requeue 和 serial_requeue 都是抢占式。
#    非抢占的只有 gpu(A100-80GB) 和 gpu_h200(H200)。
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 16
#SBATCH --mem 128G
# gpu 上限 3 天。--resume 仍然保留：作业超时或手动重投时能接着跑。
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
# 多卡是纯数据并行（fsdp_devices=1 ⇒ mesh (N,1)），batch 64 拆成每卡 64/N。
# 实测这是 compute-bound 任务（batch 32→64 步时正好翻倍，说明单卡算力已饱和），
# 所以多卡接近线性加速。A100 单卡 7.1 s/it，两卡预计 ~3.6 s/it。
# ⚠️ batch_size 必须能被卡数整除。64 对 1/2/4 都可以。
# ⚠️ gpu 分区的 A100 很紧张（140 张卡常年只剩个位数空闲），卡数越多排队越久。
#
# ⚠️ XLA_PYTHON_CLIENT_PREALLOCATE=false：JAX 默认一启动就吃掉 75% 显存，
#    不关掉的话 nvidia-smi 永远显示 ~60GB，看不出真实用量。关掉后按需分配，
#    下面的采样器读到的就是真数字 —— 这是「显存有没有浪费、能不能加大 batch」
#    唯一能回答的方式，代价是极小的分配开销（训练循环形状固定，不会反复重分配）。
#
# 分区/卡型备选：
#   #SBATCH -p gpu_requeue  + --gres=gpu:nvidia_h100_80gb_hbm3:1   （H100，但会被抢占，见上面的实测）

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
