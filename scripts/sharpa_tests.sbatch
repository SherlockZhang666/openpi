#!/bin/bash
#SBATCH -J sharpa_tests
#SBATCH -p gpu_requeue
# ⚠️ 必须钉死卡型：gpu_requeue 里混着 H200 / H100 / A100-80 / A100-40 / A40 /
#    V100 / RTX A6000，以及 MIG 切片（1g.5gb / 1g.10gb / 3g.20gb）。
#    不钉死会落到装不下的切片上。这些测试本身是 CPU 的，但 import jax 会初始化
#    设备，钉住卡型让"能不能起来"这件事可复现。
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 0-01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_tests_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_tests_%j.err

# Sharpa × π0.5 集成的测试 + 变异扫描（sharpa-pi05-plan.md Task 0/2/3/4）。
#
# 为什么要 sbatch：openpi 的套件光 import jax 就要 1–2 分钟，变异扫描要跑十几遍，
# 登录节点不跑计算（HANDOFF.md §11）。
#
# 抢占安全：本作业只读代码、跑测试，产物只有日志；变异扫描每次都从 /tmp 的原件
# 还原，被踢了重跑不会留下改坏的源文件。⚠️ 但如果作业在变异中途被 SIGKILL，
# 工作区里会残留一个被变异的源文件 —— 收尾的 `git diff --stat` 就是用来发现这件事的，
# 输出非空就 `git checkout` 掉。

set -uo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
OPENPI=$ROOT/code/Tactile/openpi
TS=$ROOT/code/Tactile/tactile_steering
PY=$OPENPI/.venv/bin/python

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1   # 不让 data_loader_test 去够 HF hub

echo "===== attempt at $(date -Is) on $(hostname) (job $SLURM_JOB_ID, restarts=${SLURM_RESTART_COUNT:-0}) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$OPENPI" || exit 1

echo
echo "########## 1. openpi 侧：Sharpa 测试 + 上游回归 ##########"
# test_with_real_dataset 会从 HF hub 下载 aloha_sim_transfer_cube_human，
# 与本改动无关且需要网络，deselect 掉。
timeout 1800 "$PY" -m pytest \
    src/openpi/training/sharpa_configs_test.py \
    src/openpi/policies/sharpa_policy_test.py \
    src/openpi/training/data_loader_test.py \
    src/openpi/transforms_test.py \
    -q -p no:cacheprovider -k "not real_dataset"
echo "openpi pytest exit=$?"

echo
echo "########## 2. tactile_steering 侧：全量 ##########"
cd "$TS" || exit 1
timeout 900 env -u PYTHONPATH \
    $ROOT/Downloads/envs/trex-quickstart/bin/python -m pytest tests/ -q -p no:cacheprovider
echo "tactile_steering pytest exit=$?"

echo
echo "########## 3. 变异扫描（护栏必须能失败，HANDOFF.md §12）##########"
bash "$OPENPI/scripts/sharpa_mutation_sweep.sh"
echo "mutation sweep exit=$?"

echo
echo "########## 4. 收尾：工作区必须是干净的 ##########"
echo "--- openpi git status ---"
git -C "$OPENPI" status --porcelain
echo "--- tactile_steering git status (只看本计划碰过的文件) ---"
git -C "$TS" status --porcelain -- data/actions.py dp2/ tests/test_actions.py tests/test_sharpa_env_bridge.py
echo "===== done at $(date -Is) ====="
