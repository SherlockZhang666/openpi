"""F6/proprioception contact-compatibility adaptation for egg, not full TouchGuide.
Demo-only training; episode-held-out validation and separate policy-rollout probe.
Negatives mean mismatched demonstrated actions, not physical failure labels.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from torch.nn import functional

from openpi.guidance.compatibility import Guide

W, H = 16, 30


def episode(e, cache):
    with np.load(cache / "episodes" / f"ep_{e['episode_index']:06d}.npz") as z:
        return {k: z[k].copy() for k in ("f6", "proprio", "action28")}


def windows(entries, st, cache, stride=3):
    arrays = {k: [] for k in ("obs", "action", "episode", "t")}
    for e in entries:
        a = episode(e, cache)
        raw = a["f6"].reshape(-1, 30).astype("f4")
        state = a["proprio"].astype("f4")
        actions = a["action28"].astype("f4")
        ts = np.arange(W - 1, len(raw) - H, stride)
        if not len(ts):
            continue
        force = (raw - st["f6_mean"]) / st["f6_std"]
        prop = (state - st["state_mean"]) / st["state_std"]
        ac = (actions - st["policy_offset"]) / st["policy_scale"]
        hist = force[ts[:, None] - np.arange(W - 1, -1, -1)[None]].reshape(len(ts), -1)
        arrays["obs"].append(np.concatenate([hist, prop[ts]], 1).astype("f4"))
        arrays["action"].append(ac[ts[:, None] + np.arange(H)[None]].astype("f4"))
        arrays["episode"].append(np.full(len(ts), e["episode_index"], int))
        arrays["t"].append(ts)
    return {k: np.concatenate(v) for k, v in arrays.items()}


@torch.no_grad()
def probe(model, d, seed=19):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(d["obs"]), min(768, len(d["obs"])), replace=False)
    o = torch.from_numpy(d["obs"][ids])
    a = torch.from_numpy(d["action"][ids])
    t = torch.full((len(ids),), 0.2)
    noise = torch.from_numpy(rng.normal(size=tuple(a.shape)).astype("f4"))
    noisy = 0.8 * a + 0.2 * noise
    pos = model.score(o, noisy, t).numpy()
    # Observation-only nearest state across a DIFFERENT episode, then evaluate its action.
    state = d["obs"][:, 480:]
    others = []
    for i in ids:
        pool = np.flatnonzero(d["episode"] != d["episode"][i])
        dist = ((state[pool] - state[i]) ** 2).mean(1)
        others.append(pool[np.argmin(dist)])
    alt = torch.from_numpy(d["action"][others])
    alt = 0.8 * alt + 0.2 * noise
    neg = model.score(o, alt, t).numpy()
    shuffled = o.clone()
    shuffled[:, :480] = shuffled[torch.from_numpy(rng.permutation(len(ids))), :480]
    sh = model.score(shuffled, noisy, t).numpy()
    return {
        "n_windows": len(ids),
        "n_episodes": len(np.unique(d["episode"][ids])),
        "matched_mean": float(pos.mean()),
        "near_state_mismatch_mean": float(neg.mean()),
        "score_gap": float((pos - neg).mean()),
        "matched_above_near_state_fraction": float((pos > neg).mean()),
        "shuffled_touch_mean": float(sh.mean()),
        "note": "Retrieval diagnostic, not success probability or physical feasibility accuracy.",
    }


def export(model, st, path):
    params = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    params.update(st)
    params["no_touch"] = np.array(model.no_touch)
    np.savez(path, **params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, required=True, help="rig cache with manifest.json and episodes/*.npz")
    ap.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Base policy step directory with assets/local_repo/norm_stats.json",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.epochs < 1 or a.threads < 1:
        ap.error("epochs and threads must be positive")
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.cache / "manifest.json").read_text())
    entries = manifest["entries"]
    tr = [e for e in entries if e["source"] == "demo" and e["split"] == "train"]
    va = [e for e in entries if e["source"] == "demo" and e["split"] == "val"]
    te = [e for e in entries if e["split"] == "pi05_heldout"]
    if len(tr) < 2 or len(va) < 2:
        raise ValueError("Need at least two independent demo train and validation episodes for cross-episode negatives")
    raw = [episode(e, a.cache) for e in tr]
    f6 = np.concatenate([d["f6"].reshape(-1, 30) for d in raw])
    state = np.concatenate([d["proprio"] for d in raw])
    floor = np.tile([0.05, 0.05, 0.05, 0.001, 0.001, 0.001], 5)
    n = json.loads((a.checkpoint / "assets/local_repo/norm_stats.json").read_text())["norm_stats"]["actions"]
    q01 = np.array(n["q01"][:28], "f4")
    q99 = np.array(n["q99"][:28], "f4")
    st = {
        "f6_mean": f6.mean(0).astype("f4"),
        "f6_std": np.maximum(f6.std(0), floor).astype("f4"),
        "state_mean": state.mean(0).astype("f4"),
        "state_std": np.maximum(state.std(0), 0.01).astype("f4"),
        "policy_offset": (q01 + (q99 - q01 + 1e-6) / 2).astype("f4"),
        "policy_scale": ((q99 - q01 + 1e-6) / 2).astype("f4"),
    }
    train, val = [windows(es, st, a.cache) for es in (tr, va)]
    test = windows(te, st, a.cache) if te else None
    meta = {
        "scope": "Egg F6+proprioception compatibility adaptation; no visual guidance encoder; visual policy unchanged.",
        "train_episode_ids": [e["episode_index"] for e in tr],
        "validation_episode_ids": [e["episode_index"] for e in va],
        "test_policy_episode_ids": [e["episode_index"] for e in te],
        "cache_attributes": manifest["attrs"],
        "policy_checkpoint": str(a.checkpoint),
        "noise_time_train": [0.0, 0.4],
        "noise_protocol": "50% clean, otherwise uniform flow interpolation in [0,.4] with unit Gaussian.",
        "model_selection": "best validation matched-minus-nearest-cross-episode-state score gap",
        "validation_not_independent_test": True,
        "W": W,
        "H": H,
        "guide_context_keys": ["guidance/f6_history", "state"],
        "f6_contract": "16x5x6 bias-subtracted N/Nm in rig_egg finger/frame convention; caller must calibrate before inference",
        "config": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(a).items()},
        "results": {},
    }
    np.savez_compressed(a.out / "validation_windows.npz", **val)
    if test is not None:
        np.savez_compressed(a.out / "test_policy_windows.npz", **test)
    o = torch.from_numpy(train["obs"])
    ac = torch.from_numpy(train["action"])
    for tag, no_touch in [("full", False), ("no_touch", True)]:
        torch.manual_seed(2026)
        model = Guide(no_touch=no_touch)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best = -float("inf")
        history = []
        for epoch in range(a.epochs):
            model.train()
            order = torch.randperm(len(o))
            losses = []
            for start in range(0, len(o), 256):
                ids = order[start : start + 256]
                if len(ids) < 2:
                    continue
                t = torch.rand(len(ids)) * 0.4
                t = torch.where(torch.rand(len(ids)) < 0.5, 0.0, t)
                noise = torch.randn_like(ac[ids])
                mixed = (1 - t[:, None, None]) * ac[ids] + t[:, None, None] * noise
                oo = model.observation(o[ids])
                aa = model.action(mixed, t)
                logits = oo @ aa.T / 0.1
                labels = torch.arange(len(ids))
                loss = (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 2
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            model.eval()
            metrics = probe(model, val)
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics})
            if metrics["score_gap"] > best:
                best = metrics["score_gap"]
                torch.save(model.state_dict(), a.out / f"{tag}.pt")
                export(model, st, a.out / f"{tag}.npz")
                best_epoch = epoch + 1
            print(
                tag,
                "epoch",
                epoch + 1,
                "loss",
                round(np.mean(losses), 4),
                "val gap",
                round(metrics["score_gap"], 4),
                flush=True,
            )
        model.load_state_dict(torch.load(a.out / f"{tag}.pt", weights_only=True))
        model.eval()
        meta["results"][tag] = {
            "best_epoch": best_epoch,
            "validation": probe(model, val),
            "test_policy": probe(model, test) if test is not None else None,
            "history": history,
        }
        (a.out / "training_results.json").write_text(json.dumps(meta, indent=2))
    print("TRAIN_GUIDE_RESULT: OK", a.out, flush=True)


if __name__ == "__main__":
    main()
