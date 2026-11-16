# ruff: noqa: B023, SLF001
# Verification closures are evaluated before advancing each loop; the zero reference uses an internal test path.
"""Trained-weight parity and guidance mechanism tests, without a robot."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.guidance.compatibility import Guide
from openpi.guidance.tactile import guidance_step
from openpi.guidance.tactile import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.guide_dir
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    jax.config.update("jax_default_matmul_precision", "highest")
    d = dict(np.load(out / "validation_windows.npz"))
    results = {}
    for name in ("full", "no_touch"):
        z = dict(np.load(out / f"{name}.npz"))
        no = bool(z.pop("no_touch"))
        params = {k: jnp.asarray(v) for k, v in z.items()}
        model = Guide(no_touch=no)
        model.load_state_dict(torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True))
        model.eval()
        o = torch.tensor(d["obs"][:3])
        clean = torch.tensor(d["action"][:3])
        torch.manual_seed(42)
        noisy = (0.8 * clean + 0.2 * torch.randn_like(clean)).requires_grad_()
        t = torch.full((3,), 0.2)
        st = model.score(o, noisy, t)
        gt = torch.autograd.grad(st.sum(), noisy)[0].numpy()
        context = {
            "f6": jnp.asarray(
                (d["obs"][:3, :480] * np.tile(z["f6_std"], 16) + np.tile(z["f6_mean"], 16)).reshape(3, 16, 5, 6)
            ),
            "state": jnp.asarray(d["obs"][:3, 480:] * z["state_std"] + z["state_mean"]),
        }
        x = jnp.pad(jnp.asarray(noisy.detach().numpy()), ((0, 0), (0, 0), (0, 4)))
        sj = jax.jit(lambda a: score(params, a, 0.2, context, no_touch=no))(x)
        gj = jax.jit(jax.grad(lambda a: score(params, a, 0.2, context, no_touch=no).sum()))(x)
        np.testing.assert_allclose(sj, st.detach().numpy(), atol=8e-5, rtol=8e-5)
        np.testing.assert_allclose(np.asarray(gj)[..., :28], gt, atol=8e-5, rtol=5e-4)
        np.testing.assert_array_equal(np.asarray(gj)[..., 28:], np.zeros((3, 30, 4)))

        def fn(scale, t):
            return guidance_step(
                params, x, t, x, context, scale=scale, max_time=0.4, max_step_rms=0.001, num_steps=10, no_touch=no
            )

        zero = fn(0.0, 0.2)
        early = fn(1.0, 0.8)
        nxt = fn(1.0, 0.2)
        np.testing.assert_array_equal(zero, x)
        np.testing.assert_array_equal(early, x)
        np.testing.assert_array_equal(np.asarray(nxt)[..., 28:], np.asarray(x)[..., 28:])
        rms = np.sqrt(((np.asarray(nxt - x)[..., :28]) ** 2).mean((1, 2)))
        assert rms.max() <= 0.001001
        gain = np.asarray(score(params, nxt, 0.2, context, no_touch=no) - sj)
        assert gain.min() >= -1e-6
        assert gain.max() > 0
        tg = jax.grad(lambda f: score(params, x, 0.2, {"f6": f, "state": context["state"]}, no_touch=no).sum())(
            context["f6"]
        )
        if no:
            np.testing.assert_array_equal(np.asarray(tg), np.zeros_like(np.asarray(tg)))
        else:
            assert float(jnp.linalg.norm(tg)) > 0
        results[name] = {
            "forward_max_abs": float(np.max(abs(np.asarray(sj) - st.detach().numpy()))),
            "gradient_max_abs": float(np.max(abs(np.asarray(gj)[..., :28] - gt))),
            "positive_model_score_gain": gain.tolist(),
            "scale_zero_exact": True,
            "early_steps_unchanged": True,
            "padding_unchanged": True,
            "gradient_bound_checked": True,
        }
    (args.out / "runtime_tests.json").write_text(json.dumps(results, indent=2))
    print("RUNTIME_TEST_RESULT: OK", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
