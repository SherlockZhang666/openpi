#!/bin/bash
# 变异测试：证明 Task 0/2/3/4 的护栏真的能失败（HANDOFF.md §12）。
set -u
REPO=/n/netscratch/ydu_lab/Lab/hangxing/code/Tactile/openpi
PY=$REPO/.venv/bin/python
cd "$REPO" || exit 1

CFG=src/openpi/training/config.py
DL=src/openpi/training/data_loader.py
POL=src/openpi/policies/sharpa_policy.py
SC=src/openpi/training/sharpa_configs.py

for f in "$CFG" "$DL" "$POL" "$SC"; do cp "$f" "/tmp/$(basename "$f").orig"; done
restore () { for f in "$CFG" "$DL" "$POL" "$SC"; do cp "/tmp/$(basename "$f").orig" "$f"; done; }

run_mutation () {
  local name="$1" file="$2" old="$3" new="$4"
  restore
  # WARNING do not use `grep -c` here: with a multi-line pattern grep treats each
  # line as its own pattern and counts matching LINES, so a multi-line anchor comes
  # back as 2/3/5 and the mutation is silently skipped. The first version of this
  # script did exactly that -- 6 of 11 mutations never actually ran, and the log
  # showed only a SKIP line.
  python3 - "$file" "$old" "$new" <<'PYEOF'
import io, sys
f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = io.open(f, encoding="utf-8").read()
n = s.count(old)
if n != 1:
    sys.stderr.write("ANCHOR-COUNT=%d\n" % n)
    sys.exit(3)
io.open(f, "w", encoding="utf-8").write(s.replace(old, new, 1))
PYEOF
  if [ $? -ne 0 ]; then
    echo "[$name] SKIP <-- BAD: anchor did not appear exactly once"
    BAD=$((BAD + 1))
    return
  fi
  local out
  out=$(timeout 400 $PY -m pytest src/openpi/training/sharpa_configs_test.py \
        src/openpi/policies/sharpa_policy_test.py -q -p no:cacheprovider 2>&1 | tail -1)
  # A mutation is only useful if it makes a test FAIL. An `error` means the mutation
  # broke the code rather than changing its behaviour -- that proves nothing about the
  # guardrail, so it counts as BAD too.
  case "$out" in
    *failed*)  echo "[$name] CAUGHT   $out" ;;
    *error*)   echo "[$name] BAD <-- mutation broke the code, not its behaviour: $out"
               BAD=$((BAD + 1)) ;;
    *)         echo "[$name] SURVIVED <-- BAD: guardrail did not catch it: $out"
               BAD=$((BAD + 1)) ;;
  esac
}

BAD=0

# --- Task 0: 上游补丁 ---
run_mutation "M1 LeRobotDataset 不传 root" "$DL" \
  "        root=root,
        delta_timestamps={" \
  "        delta_timestamps={"

run_mutation "M2 Metadata 不传 root" "$DL" \
  "lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)" \
  "lerobot_dataset.LeRobotDatasetMetadata(repo_id)"

# --- Task 3: create() 的两处转发（原计划就是漏了这两行）---
run_mutation "M3 漏传 action_sequence_keys" "$CFG" \
  "            action_sequence_keys=self.action_sequence_keys,
            root=self.root if self.root is not tyro.MISSING else None," \
  "            root=self.root if self.root is not tyro.MISSING else None,"

run_mutation "M4 漏传 root" "$CFG" \
  "            action_sequence_keys=self.action_sequence_keys,
            root=self.root if self.root is not tyro.MISSING else None," \
  "            action_sequence_keys=self.action_sequence_keys,"

run_mutation "M5 repack 相机串位" "$CFG" \
  '                        "base": self.head_img_name,
                        "wrist": self.wrist_img_name,' \
  '                        "base": self.wrist_img_name,
                        "wrist": self.head_img_name,'

# --- Task 2: transform ---
run_mutation "M6 输出不切回 28" "$POL" \
  'return {"actions": np.asarray(data["actions"][..., :SHARPA_ACTION_DIM])}' \
  'return {"actions": np.asarray(data["actions"])}'

run_mutation "M7 掩码分支写成 != PI0 (PI05 下静默翻转)" "$POL" \
  "                if self.model_type == _model.ModelType.PI0_FAST" \
  "                if self.model_type != _model.ModelType.PI0"

run_mutation "M8 相机槽位互换" "$POL" \
  '                "base_0_rgb": head_image,
                "left_wrist_0_rgb": wrist_image,' \
  '                "base_0_rgb": wrist_image,
                "left_wrist_0_rgb": head_image,'

run_mutation "M9 state 维度校验被删" "$POL" \
  '            raise ValueError(
                f"expected state dim {SHARPA_STATE_DIM} (arm 7 + hand 22), got {state.shape[-1]}"
            )' \
  "            pass"

# --- Task 4: 配置 ---
run_mutation "M10 discrete_state_input 悄悄改成 False" "$SC" \
  '    "pi05": True,' \
  '    "pi05": True,
    "discrete_state_input": False,'

run_mutation "M11 freeze_filter 用了不配套的 model" "$SC" \
  "        freeze_filter=model.get_freeze_filter()," \
  "        freeze_filter=pi0_config.Pi0Config(pi05=True).get_freeze_filter(),"

restore
echo "--- 已还原，确认基线仍然全绿 ---"
timeout 400 $PY -m pytest src/openpi/training/sharpa_configs_test.py \
  src/openpi/policies/sharpa_policy_test.py -q -p no:cacheprovider 2>&1 | tail -1
git -C "$REPO" diff --stat

echo "--- 变异扫描判决 ---"
if [ $BAD -ne 0 ]; then
  echo "MUTATION SWEEP FAILED: $BAD mutation(s) skipped, survived, or malformed."
  echo "A sweep that silently does nothing is worse than no sweep -- fix the anchors."
  exit 1
fi
echo "MUTATION SWEEP OK: every mutation was applied and caught."
