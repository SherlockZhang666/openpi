#!/usr/bin/env python3
"""Exercise the policy server end to end without the robot.

    python examples/sharpa/rollout/ping_policy.py --host 127.0.0.1 --port 8000

Answers the three questions you would otherwise only find out with the arm engaged:

  1. Does the server load, and does the whole serving path -- checkpoint, norm stats,
     camera slots, tokeniser, 28-d slicing -- actually produce an action chunk?
  2. How long does a forward pass take on THIS machine? That number sets `--infer-lead`:
     the budget is `infer_lead / fps` seconds, 133 ms at the defaults, and if inference
     does not fit the control loop stalls at every chunk boundary.
  3. Are the actions in the range the policy was trained in? With synthetic input the
     answer is meaningless as behaviour, but a chunk full of NaN, the wrong width, or
     metre-scale wrist steps is a wiring fault and shows up immediately.

Synthetic observations by default. `--replay FILE.npz` (a `--record` file from a previous
rollout) feeds real states instead, which is the closest you can get to a real rollout
with the robot switched off.

This is a client: it needs `packages/openpi-client` and nothing else. No JAX, no GPU.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import rig

STATE_DIM = 29
ACTION_DIM = 28
# The per-step maxima in the training data, from the checkpoint's action q01/q99.
TRAIN_MAX_STEP_LIN = 0.0046   # m
TRAIN_MAX_STEP_ANG = 0.0229   # rad
# Native camera resolutions downscaled to short side 256, exactly as cameras.py serves them.
HEAD_HW = (256, 410)
WRIST_HW = (256, 456)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", default="pick up the egg")
    p.add_argument("--n", type=int, default=20, help="inference calls to time")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--infer-lead", type=int, default=8,
                   help="the value you intend to pass to main.py; the latency budget is "
                        "this many steps at --fps")
    p.add_argument("--replay", default=None, metavar="NPZ",
                   help="a --record file; its states and actions are used instead of "
                        "synthetic ones (images are still synthetic)")
    p.add_argument("--rig-root", default=None)
    return p.parse_args(argv)


def synthetic_obs(rng, prompt: str) -> dict:
    """Shapes and dtypes exactly as the rollout client sends them."""
    return {
        "state": rng.normal(scale=0.3, size=STATE_DIM).astype(np.float32),
        "base": rng.integers(0, 256, size=(*HEAD_HW, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 256, size=(*WRIST_HW, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def check_chunk(chunk: np.ndarray) -> list[str]:
    """Wiring faults only. This says nothing about whether the policy is any good."""
    problems = []
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        return [f"chunk is {chunk.shape}, expected (horizon, {ACTION_DIM})"]
    if not np.isfinite(chunk).all():
        problems.append(f"{int((~np.isfinite(chunk)).sum())} non-finite values")
    lin = np.linalg.norm(chunk[:, 22:25], axis=1)
    ang = np.linalg.norm(chunk[:, 25:28], axis=1)
    over_lin = int((lin > 4 * TRAIN_MAX_STEP_LIN).sum())
    over_ang = int((ang > 4 * TRAIN_MAX_STEP_ANG).sum())
    if over_lin:
        problems.append(f"{over_lin}/{len(chunk)} steps move >4x the training max "
                        f"(worst {lin.max() * 1000:.1f} mm vs {TRAIN_MAX_STEP_LIN * 1000:.1f})")
    if over_ang:
        problems.append(f"{over_ang}/{len(chunk)} steps rotate >4x the training max "
                        f"(worst {np.rad2deg(ang.max()):.2f} deg vs "
                        f"{np.rad2deg(TRAIN_MAX_STEP_ANG):.2f})")
    return problems


def main(argv=None) -> int:
    a = parse_args(argv)
    rig.add_paths(a.rig_root)
    rig.check_client_deps()

    from openpi_client import websocket_client_policy

    states = None
    if a.replay:
        rec = np.load(pathlib.Path(a.replay))
        states = np.asarray(rec["state"], dtype=np.float32)
        print(f"replaying {len(states)} recorded states from {a.replay}")

    print(f"connecting to ws://{a.host}:{a.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)
    print("server metadata:", client.get_server_metadata())

    rng = np.random.default_rng(0)
    latencies, problems, horizon = [], [], None
    lin_steps, ang_steps = [], []
    for i in range(a.n):
        obs = synthetic_obs(rng, a.prompt)
        if states is not None:
            obs["state"] = states[i % len(states)]
        t0 = time.monotonic()
        chunk = np.asarray(client.infer(obs)["actions"], dtype=np.float64)
        latencies.append(time.monotonic() - t0)
        horizon = chunk.shape[0]
        lin_steps.append(np.linalg.norm(chunk[:, 22:25], axis=1))
        ang_steps.append(np.linalg.norm(chunk[:, 25:28], axis=1))
        problems.extend(f"call {i}: {msg}" for msg in check_chunk(chunk))

    # The FIRST call is dropped by position, not by rank. It pays for XLA compilation and
    # is tens of seconds; sorting first and then dropping lat[0] would discard the fastest
    # call and leave the 36-second one sitting in p95 and max, which is exactly backwards.
    warmup = latencies[0]
    steady = sorted(latencies[1:]) or sorted(latencies)
    p50 = statistics.median(steady)
    p95 = steady[min(len(steady) - 1, int(0.95 * len(steady)))]
    budget = a.infer_lead / a.fps

    print(f"\nchunk shape        ({horizon}, {ACTION_DIM})")
    print(f"warm-up call       {warmup * 1e3:7.0f} ms   (XLA compilation; happens once)")
    print(f"steady   p50       {p50 * 1e3:7.0f} ms   over {len(steady)} calls")
    print(f"         p95       {p95 * 1e3:7.0f} ms")
    print(f"         max       {steady[-1] * 1e3:7.0f} ms")
    print(f"budget at --infer-lead {a.infer_lead} @ {a.fps:g} Hz: {budget * 1e3:.0f} ms")

    lin = np.concatenate(lin_steps)
    ang = np.concatenate(ang_steps)
    print(f"\nwrist step |dp|    median {np.median(lin) * 1000:5.2f} mm  max "
          f"{lin.max() * 1000:5.2f} mm   (training max {TRAIN_MAX_STEP_LIN * 1000:.1f})")
    print(f"wrist step |dr|    median {np.rad2deg(np.median(ang)):5.2f} deg max "
          f"{np.rad2deg(ang.max()):5.2f} deg  (training max "
          f"{np.rad2deg(TRAIN_MAX_STEP_ANG):.2f})")

    if problems:
        print("\nCHUNK PROBLEMS (these are wiring faults, not policy quality):")
        for msg in problems[:10]:
            print("  " + msg)
        return 1

    # Size the lead off p95 rather than max: one outlier costs a single stalled boundary,
    # which the loop reports and recovers from, whereas sizing off max spends chunk budget
    # on a tail that may not recur. The +1 is a rounding cushion, not superstition.
    need = int(np.ceil(p95 * a.fps)) + 1
    if p95 > budget:
        print(f"\nlatency does not fit the budget. Use --infer-lead {need}, and keep "
              f"--infer-lead < --chunk-steps with their sum <= {30}: e.g. "
              f"--infer-lead {need} --chunk-steps {min(30 - need, 2 * need)}.")
        return 2

    print(f"\nOK -- p95 inference fits in {budget * 1e3:.0f} ms with "
          f"{(budget - p95) * 1e3:.0f} ms to spare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
