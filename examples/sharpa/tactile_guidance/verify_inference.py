# ruff: noqa: B023, SLF001
# Verification closures are evaluated before advancing each loop; the zero reference uses an internal test path.
"""Actual checkpoint inference checks on recorded inputs; no task-success claim."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.guidance.tactile import GuidedPolicy
from openpi.guidance.tactile import score
from openpi.policies import policy_config
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--guide-dir", type=Path, required=True, help="Directory with full.npz and no_touch.npz")
    parser.add_argument("--inputs", type=Path, required=True, help="Recorded inference_inputs.npz fixture")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.inputs, allow_pickle=False) as archive:
        fixture = dict(archive)
    jax.config.update("jax_default_matmul_precision", "highest")
    checkpoint = args.checkpoint
    cfg = config.get_config("sharpa_egg")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert dc.use_quantile_norm, "Training assumed quantile action normalization; mismatch requires retraining"
    base = policy_config.create_trained_policy(cfg, checkpoint, default_prompt="pick up the egg")
    zero = GuidedPolicy(base, args.guide_dir / "full.npz", scale=0.0)
    guided = GuidedPolicy(base, args.guide_dir / "full.npz", scale=1.0)
    no_touch = GuidedPolicy(base, args.guide_dir / "no_touch.npz", scale=1.0)
    params = guided.sampler.guide_weights.value
    rows = []
    arrays = {}
    for i in range(len(fixture["state"])):
        obs = {
            "base": fixture["base"][i],
            "wrist": fixture["wrist"][i],
            "state": fixture["state"][i],
            "prompt": "pick up the egg",
            "guidance/f6_history": fixture["f6"][i],
        }
        raw = {k: v for k, v in obs.items() if not k.startswith("guidance/")}
        noise = np.random.default_rng(100 + i).normal(size=(30, 32)).astype("f4")
        start = time.monotonic()
        baseline = base.infer(raw, noise=noise)
        off = zero.infer(obs, noise=noise)
        np.testing.assert_array_equal(off["actions"], baseline["actions"])
        reference = guided._infer_guided(obs, noise=noise, scale=0.0)
        on = guided.infer(obs, noise=noise)
        no = no_touch.infer(obs, noise=noise)
        for result in (baseline, off, on, no):
            assert result["actions"].shape == (30, 28)
            assert np.isfinite(result["actions"]).all()
        diff = on["actions"] - reference["actions"]
        assert np.max(abs(diff)) > 1e-8, "Guidance produced no action change"
        context = {"f6": jnp.asarray(obs["guidance/f6_history"])[None], "state": jnp.asarray(obs["state"])[None]}

        def clean_score(actions):
            norm = (jnp.asarray(actions) - params["policy_offset"]) / params["policy_scale"]
            padded = jnp.pad(norm, ((0, 0), (0, 4)))[None]
            return float(score(params, padded, 0.0, context, no_touch=False)[0])

        rows.append(
            {
                "fixture": i,
                "episode": int(fixture["episode"][i]),
                "t": int(fixture["t"][i]),
                "guidance_off_max_abs": float(np.max(abs(off["actions"] - baseline["actions"]))),
                "compiled_zero_vs_original_max_abs": float(np.max(abs(reference["actions"] - baseline["actions"]))),
                "guided_action_rms": float(np.sqrt((diff**2).mean())),
                "guided_normalized_action_rms": float(
                    np.sqrt(((diff / np.asarray(params["policy_scale"])) ** 2).mean())
                ),
                "guide_score_base": clean_score(baseline["actions"]),
                "guide_score_compiled_zero": clean_score(reference["actions"]),
                "guide_score_guided": clean_score(on["actions"]),
                "elapsed_including_jit_s": time.monotonic() - start,
            }
        )
        for name, r in [
            ("base", baseline),
            ("off", off),
            ("compiled_zero", reference),
            ("guided", on),
            ("no_touch", no),
        ]:
            arrays[f"{name}_{i}"] = r["actions"]
        print("INFERENCE_INPUT_OK", rows[-1], flush=True)
    np.savez_compressed(out / "inference_actions.npz", **arrays)
    (out / "inference_results.json").write_text(
        json.dumps(
            {
                "scope": "Recorded-input inference and sampler equivalence only; not physical execution or task success. Inputs held out from guide training but included in base-policy training.",
                "checkpoint": str(checkpoint),
                "numerical_control": "Public guidance-off calls original sampler exactly. Steering changes are measured against zero guidance on the same compiled guided graph; bfloat16 compilation may differ from original graph.",
                "jax_devices": [str(x) for x in jax.devices()],
                "results": rows,
            },
            indent=2,
        )
    )
    print("INFERENCE_STEERING_RESULT: OK", flush=True)


if __name__ == "__main__":
    main()
