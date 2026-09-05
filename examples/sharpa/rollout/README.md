# Sharpa rollout — running the trained pi0.5 policy on the real rig

Client for the OpenArm(7) + Sharpa Wave left-hand(22) rig. The policy runs in an openpi
policy server; this directory is the robot-side client that turns its 28-d actions into
hand commands and arm targets.

**This file is the design: what the pieces are and why they are the way they are.
[`RUNBOOK.md`](RUNBOOK.md) is the procedure: what to type at the robot, in order.**

```
  openpi venv, any machine with a GPU
  scripts/serve_policy.py policy:checkpoint --policy.config=sharpa_egg --policy.dir=<ckpt>
        │  websocket + msgpack, :8000
        ▼
  <rig>/.venv + ROS sourced        examples/sharpa/rollout/main.py
        ├─ observation  /joint_states (left arm 7) + hand broadcast :50000 (22) + 2 cameras
        ├─ hand         SharpaWave SDK  set_joint_position(22 absolute)
        └─ arm          UDP :9873 → openarm_vr_teleop.py → IK, safety, controller
```

**The arm is driven through `openarm_vr_teleop.py`, not through the controller.** The
policy impersonates the Quest bridge, so every safety layer that exists for
teleoperation — the soft joint-limit envelope, the per-joint rate limit, the deviation
trip, RAMP_BACK, the gravity feed-forward — is still between the policy and the hardware.
Publishing joint commands directly would mean reimplementing all of them on live
hardware.

## The checkpoint

`zhx-tactile-steering/sharpa-pi05` on the Hub, private. Two steps of one run, fine-tuned
from `pi05_base` on 38 `pick_up_the_egg` episodes:

| dir | step | epochs |
|---|---|---|
| `egg_70ep_b64/8640` | 8640 | 40 — try this if 70ep looks overfit |
| `egg_70ep_b64/15120` | 15120 | 70 — final |

⚠️ The model card writes the first one `08640`; **on the Hub it is `8640`, unpadded.** That
matters because `hf download --include` with a pattern that matches nothing **exits 0 and
downloads nothing** — no warning, no error, and the next thing you see is the server
failing to find a checkpoint. Check the file count, not the exit status.

```bash
hf download zhx-tactile-steering/sharpa-pi05 \
    --include "egg_70ep_b64/15120/*" --local-dir ./checkpoints/sharpa-pi05
```

`checkpoints/` is in the repo's `.gitignore`, so this does not have to live outside the
tree.

Each is `params/` (6.0 GB, orbax OCDBT) plus `assets/local_repo/norm_stats.json`. **Both
directories are required.** `create_trained_policy` resolves norm stats from the
checkpoint's own `assets/`, and a checkpoint copied without it loads without complaint and
runs unnormalised. `scripts/sharpa_serve.sh` refuses to start in that case.

There is **no validation set** — all 38 episodes went into training, and every number on
the model card is training error. Pick a checkpoint by testing on the robot.

## What has to be true before any of this runs

| | |
|---|---|
| The checkpoint, **with its `assets/`** | see above |
| openpi venv on the serving machine, **with network on first run** | `uv sync` in the repo root. The first server start also fetches `gs://big_vision/paligemma_tokenizer.model` (4 MB) into `~/.cache/openpi`; after that it is offline-capable. pi0.5 in bf16 is ~6 GB of weights. ⚠️ `uv sync` pulls `lerobot` from git, whose checkout runs the git-lfs filter; without `git-lfs` installed it fails with `the remote end hung up unexpectedly`. Either install git-lfs, or neutralise the filter for that one command — see the `GIT_CONFIG_*` block in `scripts/sharpa_serve.sh`. |
| The hand's command units, **measured** | `probe_hand_units.py`. See below; guessing is a factor of 57.3. |
| The rig stack up | hand → Quest-less arm bring-up → home → teleop node. `collect_up.sh` is the reference, but do **not** start `quest_bridge` or `udp_fanout` — this client takes their place on :9873. |
| `websockets>=12.0` in the rig venv | `<rig>/.venv/bin/python -m pip install 'websockets>=12.0'` — **already installed**. The venv inherits 10.4 from the system and `websockets.sync`, the blocking client, only exists from 12.0. Nothing else in the rig imports websockets. **Do not also upgrade msgpack**: openpi-client's metadata asks for ≥1.0.5, the system's 1.0.3 round-trips the real payload fine, and `python-can 4.3.1` pins `msgpack~=1.0.0`. |

The client is not installed. It runs from the source tree with the rig's interpreter, and
puts the rig modules and `packages/openpi-client/src` on `sys.path` itself (`rig.py`) —
deliberately, so nothing pip-installs into the working robot venv. openpi-client declares
`numpy<2.0` and that venv runs numpy 2.5.

## Running it

**The step-by-step operating procedure is [`RUNBOOK.md`](RUNBOOK.md)** — terminal layout,
bring-up order, pass criteria for every step, shutdown, a troubleshooting table, and the
protocol for choosing between the two checkpoints. It is in Chinese, matching the rig docs
it has to be read alongside (`vr/README.md`, `vr/collect/README.md`).

The shape of it:

```bash
scripts/sharpa_serve.sh ./checkpoints/sharpa-pi05/egg_70ep_b64/15120   # 1. server
python examples/sharpa/rollout/ping_policy.py                          # 2. check it, no robot
python examples/sharpa/rollout/probe_hand_units.py --i-am-watching     # 3. hand units, once
#    ... arm bring-up, home, teleop node with --scale 1.0 ...
python examples/sharpa/rollout/main.py --prompt "pick up the egg"      # 4. dry run
python examples/sharpa/rollout/main.py --prompt "..." --enable-hand --enable-arm   # 5. go
```

**Nothing moves without `--enable-arm` / `--enable-hand`.** A dry run exercises the whole
pipeline at full rate — cameras, state, inference, action decoding, the integration chain,
the safety envelopes — and prints what it would have commanded. Do it every time.

## Latency, measured

`laptoplingfeng` (RTX 5090 Laptop 24 GB), checkpoint `15120`, via `ping_policy.py`:

```
chunk shape        (30, 28)
warm-up call     36471 ms   XLA compilation, once per server start
steady   p50       188 ms
         p95       227 ms
budget at --infer-lead 8 @ 30 Hz: 267 ms      -> OK, 40 ms to spare
wrist step |dp|    median 0.84 mm  max 3.56 mm   (training max 4.6)
wrist step |dr|    median 0.23 deg max 0.89 deg  (training max 1.31)
```

This is what sets the defaults. **`--infer-lead 4` is not enough on this hardware** — the
133 ms it buys is below p50, so the loop would stall at every chunk boundary; hence
`--chunk-steps 15 --infer-lead 8`. Re-measure on whatever machine actually serves;
`ping_policy.py` prints the value to pass.

The second half of that output is the more interesting one. The wrist steps come back
*inside* the training distribution even on random-noise images, which is the strongest
single piece of evidence that unnormalisation is wired correctly end to end: a norm-stats
mismatch produces magnitudes that are wrong by orders of magnitude, not subtly off.


## How the 28-d action is executed

```
[0:22]  hand joints, ABSOLUTE, radians   → rate-limited, clipped, set_joint_position
[22:25] palm-frame translation delta (m) ┐ integrated into a commanded wrist pose,
[25:28] palm-frame rotation delta (rad)  ┘ then re-expressed as a delta since engage
```

The asymmetry — fingers absolute, wrist incremental — is the dataset's, not a choice made
here. `actions.py` carries the integration and `actions_test.py` pins it against the
converter's own formula by round trip; if that test fails the policy is being replayed in
a frame it was never trained in.

Two conventions that fail **silently** if you get them wrong, and are pinned by tests
rather than by care:

* **Quaternions are w-first `[x,y,z,qw,qx,qy,qz]`**, everywhere on this rig
  (`DATA_SCHEMA_v1.md`, `collect_core._pose7_wxyz`, `openarm_vr_teleop.se3_to_pose7`).
  scipy is xyzw. `actions.py` never touches scipy and works in matrices internally;
  `actions_test.py` round-trips against the converter's scipy path, which is where the
  `[4,5,6,3]` reindex lives. Get this backwards and the arm goes the wrong way with no
  error anywhere.
* **`PalmChain.step` integrates every delta it is given — there is no sentinel skip.**
  The training-side inverse (`tactile_steering.integrate_palm_centric`) *does* skip index
  0, correctly, because `palm_centric` fabricates a zero there for the first frame of an
  episode. A policy chunk has no such frame: all 30 actions are real increments. Calling
  the training-side function on a chunk, or "simplifying" `PalmChain` to match it, drops
  the first action of every chunk — no error, just a small deficit at every boundary that
  accumulates into drift. `loop_test.py` and `schedule_test.py` both assert that N control
  steps produce exactly N integrations.

Two more consequences worth knowing before debugging a bad rollout:

* **The chain is open loop.** It is seeded once, from the node's telemetry at engage, and
  never re-seeded from a measurement. That is deliberate: the dataset differenced
  `action/wrist_pose_b`, the *commanded* target, and re-seeding from the measured pose
  would fold the servo's tracking error (8.2 mm median) into the command every step.
* **Chunk boundaries are not free.** A new chunk is consumed from index `--infer-lead`,
  not 0, because its observation is that many steps old. Starting at 0 would re-integrate
  motion that already happened. `schedule_test.py` pins the index sequence.

## Observation assembly

| field | source | matches training via |
|---|---|---|
| `state[0:7]` | `/joint_states`, `openarm_left_joint1..7` | `collect/sources.py:RosSources`, the same reader that wrote `obs/joint_pos` |
| `state[7:29]` | hand broadcast UDP :50000, radians | `collect/sharpa_state.py:StateReceiver`, the same reader that wrote `hand/joint_pos` |
| `base` | Orbbec Gemini 336L, 1280x800 MJPG | `collect/camera.py` device resolution + v4l2 control string |
| `wrist` | RealSense D435i, 960x540 YUYV | ditto — and this is the policy's primary view |

The rig modules are imported, not reimplemented, precisely so the joint order, the units
and the zero-order-hold semantics cannot drift from what produced the training rows. Any
source staler than `--max-age` (0.25 s) stops the rollout rather than feeding the policy a
frozen frame.

Images are downscaled to short side 256 in the GStreamer pipeline, which is what the
LeRobot converter did; the server then does its own `resize_with_pad` to 224x224. One
difference is left in on purpose: training frames made an extra h264 generation through
the converter. At crf23 that is far below the resize, and emulating it would be adding an
artefact rather than removing a difference.

## Files

| | |
|---|---|
| `RUNBOOK.md` | the operating procedure — what to type, in what order, and the pass criterion for each step |
| `main.py` | the 30 Hz loop, chunk scheduling, keys, flight recorder |
| `actions.py` | 28-d action → commanded wrist pose. Pure numpy, no hardware |
| `arm.py` | UDP wire, engage handshake, per-step and cumulative wrist envelopes |
| `hand.py` | SDK command sink (units, slew, per-joint clip) and the broadcast readback |
| `cameras.py` | live RGB, configured from the rig's own `Camera` |
| `rig.py` | locates the rig checkout, openpi-client and the Sharpa SDK; checks the client deps |
| `ping_policy.py` | server smoke test and latency measurement, no robot needed |
| `probe_hand_units.py` | the one hardware measurement this client cannot do without |
| `actions_test.py`, `schedule_test.py`, `hand_test.py`, `loop_test.py` | no hardware at all — `loop_test.py` runs the real control loop against fakes |

Run the tests after touching anything here. They need numpy, scipy and pytest and nothing
else — no JAX, no GPU, no rig — so any interpreter that has those will do, including the
rig venv:

```bash
cd examples/sharpa/rollout && python -m pytest . -q      # ~3 s, 45 tests
```

They are deliberately **not** wired into `scripts/sharpa_tests.sbatch`. That job runs on
the FASRC cluster against a different checkout, and its whole reason to exist is that
importing JAX takes a minute or two — neither of which applies to these. Adding a path
there that the cluster checkout does not have would also make `pytest` collect nothing and
silently stop running the training-side tests that job does cover.

## Two safety envelopes, and where their numbers come from

Both are derived from the checkpoint's own `norm_stats.json`, not guessed:

| | training max | default limit | catches |
|---|---|---|---|
| `--max-step-lin` | 4.6 mm per action | 20 mm | one wild action, on the step that produced it |
| `--max-step-ang` | 1.3° per action | 5.7° | ditto, rotation |
| `--max-lin` | — | 0.45 m | slow drift that is in-distribution every single step |
| `--max-ang` | — | 1.8 rad | ditto, rotation |

Both are needed. The per-step guard cannot see a 4 mm/step drift — every action looks
normal — and the cumulative envelope would take a hundred of them, three seconds, to fire.
A rejected step raises **before** it is integrated and before it is published, so the node
keeps holding the last target that passed.

At the training maximum the wrist moves 13.7 cm/s and rotates 39°/s. If a rollout looks
much faster than that, the policy is out of distribution.

## Not using `ActionChunkBroker`

`openpi_client.action_chunk_broker.ActionChunkBroker` looks like the right tool and is not:
its `infer` calls the inner policy **synchronously** when the chunk runs out, so the control
loop stalls for a whole forward pass at every boundary. It also always resumes a fresh
chunk at index 0, which is right only because it does not prefetch.

`main.py` runs the forward pass on its own thread, fired `--infer-lead` steps early, and
consumes the returned chunk from index `lead` to compensate. That is the part
`ActionChunkBroker` has no equivalent for, and the part that is wrong if it is left out
(see the wrist-increment note above). It also drops a `dm-tree` dependency from the robot
side.

## Known gaps

* **Tactile is not used.** The converter writes `observation.tactile_force` and the five
  deform streams into a separate dataset, and the pi0.5 configs here do not consume them.
  When the tactile-steering phase lands it will need its own observation path.
* **No success detection and no auto-reset.** A rollout runs until `q`. Resetting the
  scene and re-homing the arm between takes is manual, as it is during collection.
* **Checkpoint choice is untested.** There is no validation set — all 38 episodes went
  into training — so `8640` vs `15120` is a question only the robot can answer. The A/B
  protocol, and why the model card's numbers cannot settle it, are at the end of
  `RUNBOOK.md`.
