"""Pins the chunk scheduler's index sequence.

    /path/to/python -m pytest examples/sharpa/rollout/schedule_test.py -q

This needs no hardware and no ROS: `main.py` imports only stdlib, numpy and `rig` at
module level, and `rig` does nothing at import time.

What is being protected: the wrist third of the action is an INCREMENT. Executing the
same chunk index twice, or restarting a new chunk at 0 when its observation is `lead`
steps old, integrates real motion into the command chain twice. It would not look like a
crash; it would look like the arm slowly running away from where the policy wants it.
"""

from main import ChunkSchedule
import numpy as np
import pytest

HORIZON = 30


def run(chunk_steps: int, lead: int, cycles: int):
    """Drive the scheduler for `cycles` chunks; return (indices executed, submit steps)."""
    sched = ChunkSchedule(chunk_steps=chunk_steps, lead=lead)
    executed, submits, taken = [], [], 0
    step = 0
    while taken <= cycles:
        if sched.at_boundary():
            taken += 1
            if taken > cycles:
                break
            sched.on_new_chunk()
        if sched.submit_now():
            submits.append(step)
        executed.append(sched.idx)
        sched.advance()
        step += 1
    return executed, submits


@pytest.mark.parametrize(("chunk_steps", "lead"), [(8, 4), (8, 1), (15, 5), (30, 0), (2, 1)])
def test_exactly_chunk_steps_actions_per_inference(chunk_steps, lead):
    executed, _ = run(chunk_steps, lead, cycles=6)
    assert len(executed) == 6 * chunk_steps


@pytest.mark.parametrize(("chunk_steps", "lead"), [(8, 4), (8, 1), (15, 5), (30, 0), (2, 1)])
def test_indices_stay_inside_the_action_horizon(chunk_steps, lead):
    executed, _ = run(chunk_steps, lead, cycles=6)
    assert max(executed) < HORIZON, "the offset chunk runs past the model's 30 actions"


def test_first_chunk_starts_at_zero_and_later_ones_at_lead():
    executed, _ = run(chunk_steps=8, lead=4, cycles=3)
    assert executed[:8] == list(range(8))
    assert executed[8:16] == list(range(4, 12))
    assert executed[16:24] == list(range(4, 12))


def test_inference_fires_exactly_lead_steps_before_the_boundary():
    """The lead is the whole latency budget: if it fires late the loop stalls, if it
    fires early the observation the chunk is conditioned on is needlessly old."""
    chunk_steps, lead = 8, 4
    executed, submits = run(chunk_steps, lead, cycles=4)
    # One submit per cycle. The last one would have served a fifth chunk that the run
    # never asks for -- in the real loop that is a prefetch discarded on quit.
    assert submits == [4, 12, 20, 28]
    for s in submits:
        cycle_end = (s // chunk_steps + 1) * chunk_steps
        # Exactly `lead` actions are still executed after the submit, and those are the
        # forward pass's entire latency budget.
        assert cycle_end - s == lead
        assert len(executed[s:cycle_end]) == lead


def test_zero_lead_is_fully_synchronous():
    _, submits = run(chunk_steps=30, lead=0, cycles=3)
    assert submits == [], "with no lead there is nothing to prefetch"


def test_no_index_is_executed_twice_within_a_cycle():
    executed, _ = run(chunk_steps=8, lead=4, cycles=4)
    for c in range(4):
        cycle = executed[c * 8:(c + 1) * 8]
        assert len(set(cycle)) == len(cycle)
        assert cycle == sorted(cycle)


def test_wrist_increments_are_integrated_once_per_control_step():
    """End-to-end intent check: N control steps must integrate N deltas, no more.

    Uses the real PalmChain so the assertion is about the thing that actually moves.
    """
    from actions import PalmChain

    chunk = np.zeros((HORIZON, 28))
    chunk[:, 22] = 0.001  # 1 mm of +x per action, in the palm frame

    executed, _ = run(chunk_steps=8, lead=4, cycles=5)
    chain = PalmChain.from_pose7(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    for idx in executed:
        chain.step(chunk[idx, 22:25], chunk[idx, 25:28])
    assert chain.pos[0] == pytest.approx(0.001 * len(executed))
