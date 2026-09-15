#!/usr/bin/env bash
# Serve a trained Sharpa pi0.5 checkpoint to the rollout client.
#
#   scripts/sharpa_serve.sh <ckpt-dir> [config] [extra serve_policy args...]
#
# Get a checkpoint first (private repo -- `hf auth login` or export HF_TOKEN):
#
#   hf download zhx-tactile-steering/sharpa-pi05 \
#       --include "egg_70ep_b64/15120/*" --local-dir ./ckpt
#   scripts/sharpa_serve.sh ./ckpt/egg_70ep_b64/15120
#
#   scripts/sharpa_serve.sh ./ckpt/egg_70ep_b64/08640 sharpa_egg --port 8001
#
# The config only has to name the right TrainConfig; its dataset root is never opened at
# inference time, so this runs on a machine that has no copy of the LeRobot dataset.
#
# What it DOES need is the checkpoint's own assets. create_trained_policy loads norm stats
# from <ckpt>/assets/<asset_id>/norm_stats.json, and asset_id is "local_repo" for every
# sharpa config. A checkpoint copied without that directory produces an UNNORMALISED
# policy -- no error, just an arm that goes somewhere else -- so it is checked here.
set -euo pipefail

CKPT=${1:?usage: sharpa_serve.sh <ckpt-dir> [config] [extra args...]}
CONFIG=${2:-sharpa_egg}
[[ $# -ge 2 ]] && shift 2 || shift 1

CKPT=$(readlink -f "$CKPT")
[[ -d $CKPT/params ]] || { echo "no params/ under $CKPT -- is this a step directory?" >&2; exit 1; }

NORM=$CKPT/assets/local_repo/norm_stats.json
if [[ ! -f $NORM ]]; then
    echo "missing $NORM" >&2
    echo "The policy would run unnormalised. Copy the checkpoint's assets/ directory" >&2
    echo "alongside params/ -- it is written by the training job, not derivable here." >&2
    exit 1
fi

cd "$(dirname "$0")/.."

# `uv run` re-resolves the environment on every invocation, and openpi pulls `lerobot`
# straight from git. That checkout runs git's lfs filter, so on a machine without
# `git-lfs` installed it dies with "the remote end hung up unexpectedly" -- an error that
# names neither lfs nor lerobot. Neutralise the filter for these child git processes only
# (GIT_CONFIG_* env vars, not the user's global config). The lfs payloads in lerobot are
# test assets; nothing on the serving path reads them. Harmless when git-lfs IS installed.
export GIT_CONFIG_COUNT=4 \
  GIT_CONFIG_KEY_0=filter.lfs.smudge   GIT_CONFIG_VALUE_0=cat \
  GIT_CONFIG_KEY_1=filter.lfs.clean    GIT_CONFIG_VALUE_1=cat \
  GIT_CONFIG_KEY_2=filter.lfs.process  GIT_CONFIG_VALUE_2= \
  GIT_CONFIG_KEY_3=filter.lfs.required GIT_CONFIG_VALUE_3=false

# Extra args go BEFORE the `policy:checkpoint` subcommand: they are serve_policy.py's own
# top-level options (--port, --num-candidates, --noise-temperature, ...), and tyro applies an
# argument to the subcommand directly preceding it -- after it they are "unrecognized".
exec uv run scripts/serve_policy.py "$@" policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$CKPT"
