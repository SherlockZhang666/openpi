# Egg tactile guidance inside OpenPI

This example integrates the existing experimental F6/proprioception action-compatibility guide into the frozen JAX `sharpa_egg` policy. No separate source overlay or repository is required. The base policy still uses RGB; tactile history enters the external guide.

The guide compares two normalized embeddings: 16 frames of five-finger F6 plus the current 29D state, and a 30-by-28 action sequence plus flow time. It does **not** predict future tactile sequences or calibrated task-success probabilities. It is a contact-score adaptation, not a full TouchGuide or DynaGuide reproduction.

## Setup and files

Use this branch's `pyproject.toml` / `uv.lock` and Python 3.11. From the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Set `OPENPI_DATA_HOME` to a writable cache containing `big_vision/paligemma_tokenizer.model`, or allow OpenPI to fetch that tokenizer. The full local policy step directory must contain `params/` and `assets/local_repo/norm_stats.json`.

Guide weights and recorded data remain external files; this commit does not upload demonstrations, RGB fixtures, policy checkpoints or trained guide weights. Copy your existing guide output directory, or train it using the next section. A guide output contains `full.npz`, `no_touch.npz`, their Torch `.pt` equivalents, `validation_windows.npz` and training metadata. `full.npz` includes the guide's training normalization; do not replace it with statistics fitted on test data.

Runtime modules are `src/openpi/guidance/tactile.py`, `compatibility.py` and `tactile_history.py`. The optional `guidance_fn` callback lives directly in `Pi0.sample_actions`; leaving it unset preserves the original Euler update.

## Train the compatibility guide

The cache needs a `manifest.json` and `episodes/ep_XXXXXX.npz` containing `f6`, `proprio` and `action28`. Demonstration entries use `source=demo` and `split=train` or `val`; optional `pi05_heldout` entries provide a separate distribution-shift probe. Whole episodes must remain disjoint. The raw-cache alignment and calibration are prerequisites; this command does not convert raw HDF5/video.

```bash
uv run python examples/sharpa/tactile_guidance/train.py \
  --cache /path/to/cache/rig_egg \
  --checkpoint /path/to/egg/15120 \
  --out /path/to/guide_output --epochs 40
```

Training is CPU-based and defaults to eight Torch threads. It trains full and independently fitted no-touch controls using the same architecture, InfoNCE objective and seed. Observation normalization uses training demos only; action normalization follows the base checkpoint quantiles. Half of training samples are clean, the rest interpolate clean actions with Gaussian noise at flow time in [0, 0.4]. Cross-episode mismatches are retrieval negatives, not labels of physically impossible actions. Validation selects the best checkpoint and is not an independent test set.

## Serve the policy

```bash
uv run python examples/sharpa/tactile_guidance/serve.py \
  --checkpoint /path/to/egg/15120 \
  --guide /path/to/guide_output/full.npz \
  --scale 1 --port 8001
```

The default host is loopback; pass `--host 0.0.0.0` if another machine must connect. Use `--scale 0` for the exact original sampler path, or `--guide /path/to/guide_output/no_touch.npz` for the matched control. `--num-steps` sets the base and guided sampler step count together. The server returns actions; it never sends robot commands.

A standard Sharpa observation has `base` / `wrist` RGB, raw `state[29]` and `prompt`. Add `guidance/f6_history` with shape `[16,5,6]`, float32, thumb/index/middle/ring/pinky, [Fx,Fy,Fz,Mx,My,Mz]. Forces are N and torques N m in the calibrated rig convention. Subtract free-space bias before calling the client buffer. Do not subtract a rolling contact mean.

```python
from openpi.guidance.tactile_history import TactileHistory

history = TactileHistory()
# At EVERY control tick, including while executing an action chunk:
history.append(calibrated_f6, timestamp=control_timestamp)
# After 16 fresh 30 Hz frames, when making a policy request:
request = history.attach(observation, now=control_timestamp)
# client.infer(request)
# On every episode boundary:
history.reset()
```

Do not collect one tactile frame per slow inference request. The buffer rejects incomplete, stale or discontinuous history. The runtime uses the raw 29D state and checks deployed action unnormalization against the exported guide statistics.

## Sampling rule and limits

Flow time decreases from 1 (noise) to 0 (action). With K Euler steps, the guide adds `scale/K * t/(1-t) * grad_action(score)` to the ordinary next action sample, only for `0 < t <= 0.4`. The correction's normalized RMS across the 30-by-28 physical action coordinates is capped at `--step-rms` (default 0.03); the four native padding coordinates receive no direct guidance. This cap does not bound final physical force or full-trajectory deviation. Strength is experimental and has not been selected using closed-loop success.

## Verification

Small synthetic numerical/regression checks need no policy checkpoint or data:

```bash
uv run pytest -q src/openpi/guidance/tactile_test.py
```

Check actual trained Torch/JAX scores, gradients, clipping and inactive steps using external weights and validation windows:

```bash
uv run python examples/sharpa/tactile_guidance/verify_runtime.py \
  --guide-dir /path/to/guide_output --out /path/to/new_verification
```

Check actual frozen base-policy inference using an external `inference_inputs.npz` with arrays `base`, `wrist`, `state`, `f6`, `episode`, and `t`:

```bash
uv run python examples/sharpa/tactile_guidance/verify_inference.py \
  --checkpoint /path/to/egg/15120 \
  --guide-dir /path/to/guide_output \
  --inputs /path/to/inference_inputs.npz --out /path/to/new_verification
```

The public scale-zero path is compared exactly against the original sampler at fixed noise. Nonzero intervention is compared against a zero-strength reference on the **same compiled guided graph**: bfloat16 compilation can otherwise create numerical differences unrelated to guidance. Both verification scripts write new results under `--out` and leave supplied weights/data untouched.

Existing recorded-input checks used two observations held out from guide training but included in base-policy training. They validate inference plumbing, not generalization. On the separate policy-rollout retrieval probe, the full tactile guide was weaker than no-touch (61.3% versus 71.0% for matched score above a nearest-state cross-episode mismatch). No physical or simulated closed-loop task-success benefit has been established.
