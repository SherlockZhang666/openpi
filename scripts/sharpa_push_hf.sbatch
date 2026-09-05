#!/bin/bash
#SBATCH -J sharpa_push_hf
# ⚠️ CPU 分区。登录节点 cgroup 限死 1 核，传 12 GB 会很慢甚至被 kill。
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-04:00:00
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_push_hf_%j.out
#SBATCH -e /n/netscratch/ydu_lab/Lab/hangxing/logs/sharpa_push_hf_%j.err

# 把训好的 checkpoint 推到 HF（private），供部署机 hf download。
#
# 用法： sbatch scripts/sharpa_push_hf.sbatch [exp_name] [step ...]
# 例：   sbatch scripts/sharpa_push_hf.sbatch egg_70ep_b64 8640 15120
#
# ⚠️ 只传 params/ 和 assets/。train_state/ 是优化器状态，推理用不上、每个 3.1 GB，
#    但这也意味着传上去的 checkpoint **不能用来续训**，续训得回集群拿原件。
#
# ⚠️ 仓库是 private。任何时候都不要去掉 --private / 把它改成 public ——
#    这是未发表的实验室数据训出来的模型，公开了收不回来。

set -euo pipefail
ROOT=/n/netscratch/ydu_lab/Lab/hangxing
CKPT=$ROOT/code/Tactile/ckpt/sharpa_pi05/sharpa_egg
REPO=zhx-tactile-steering/sharpa-pi05
HF=/n/home04/hangxing/.local/bin/hf

EXP=${1:-egg_70ep_b64}
shift || true
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(8640 15120)

echo "===== push $EXP steps=${STEPS[*]} -> $REPO at $(date -Is) on $(hostname) ====="
"$HF" auth whoami

# 模型卡
if [ -f "$ROOT/tmp/sharpa_pi05_README.md" ]; then
    echo "--- README.md ---"
    "$HF" upload "$REPO" "$ROOT/tmp/sharpa_pi05_README.md" README.md \
        --type model --commit-message "model card"
fi

for S in "${STEPS[@]}"; do
    SRC=$CKPT/$EXP/$S
    [ -d "$SRC/params" ] || { echo "!!! $SRC/params 不存在，跳过"; continue; }
    echo
    echo "--- $EXP/$S ($(du -sh "$SRC/params" | cut -f1)) ---"
    # 逐个 checkpoint 一次提交；--exclude 挡住 train_state，即使以后有人改了目录结构。
    "$HF" upload "$REPO" "$SRC" "$EXP/$S" \
        --type model \
        --exclude "train_state/*" \
        --commit-message "$EXP step $S (params + norm stats, inference only)"
done

echo
echo "===== done at $(date -Is) ====="
"$HF" models list "$REPO" --tree --recursive 2>/dev/null | head -30
