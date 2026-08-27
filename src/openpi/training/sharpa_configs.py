"""pi0.5 training configs for the OpenArm + Sharpa Wave left-hand rig.

This module reads NO files at import time. The dataset root comes from
SHARPA_DATASET_ROOT, the pretrained checkpoint from SHARPA_PRETRAINED_CKPT, and the
checkpoint root from SHARPA_CKPT_ROOT (defaults below). dexjoco's dexjoco_configs.py
does `open("config.yaml")` at module level, which makes importing it depend on the
current working directory; that is deliberately not done here.

Nor does it read the clock. dexjoco stamps exp_name with today's date, which makes the
same config name resolve to a different checkpoint directory on every import. Pass
--exp-name explicitly for real runs instead.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

_DEFAULT_DATASET_ROOT = "/n/netscratch/ydu_lab/Lab/hangxing/data/sharpa_lerobot"
_DEFAULT_PRETRAINED = "gs://openpi-assets/checkpoints/pi05_base/params"
_DEFAULT_CKPT_ROOT = "/n/netscratch/ydu_lab/Lab/hangxing/ckpt/sharpa_pi05"

# Matches dexjoco. discrete_state_input is left at the pi05 default (True), so the 29-d
# state is discretised into language tokens rather than fed in as a continuous input.
# Upstream's own pi05_libero sets it to False; we follow dexjoco because that matches the
# form pi05_base was pretrained in. This is a decision, not a default that drifted --
# see sharpa_configs_test.test_state_input_is_discretised.
_MODEL_KWARGS = {
    "pi05": True,
    "action_horizon": 30,
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
    "max_token_len": 250,
}


def dataset_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SHARPA_DATASET_ROOT", _DEFAULT_DATASET_ROOT))


def pretrained_ckpt() -> str:
    return os.environ.get("SHARPA_PRETRAINED_CKPT", _DEFAULT_PRETRAINED)


def ckpt_root() -> str:
    return os.environ.get("SHARPA_CKPT_ROOT", _DEFAULT_CKPT_ROOT)


@dataclasses.dataclass(frozen=True)
class SharpaTaskConfig:
    name: str
    data_subdir: str
    num_train_steps: int = 30_000
    head_img_name: str = "observation.images.head"
    wrist_img_name: str = "observation.images.wrist"


# The four tasks from HANDOFF.md section 3, all left-hand single-hand.
SHARPA_TASKS: list[SharpaTaskConfig] = [
    SharpaTaskConfig(name="sharpa_egg", data_subdir="pick_up_the_egg"),
    SharpaTaskConfig(name="sharpa_tissue", data_subdir="pull_tissue"),
    SharpaTaskConfig(name="sharpa_card", data_subdir="extract_card"),
    SharpaTaskConfig(name="sharpa_tablet", data_subdir="push_tablet"),
]


def make_sharpa_config(task: SharpaTaskConfig):
    # Imported here to avoid a circular import with config.py, matching how upstream's
    # roboarena_config / polaris_config do it.
    from openpi.training.config import DataConfig
    from openpi.training.config import SharpaDataConfig
    from openpi.training.config import TrainConfig

    model = pi0_config.Pi0Config(**_MODEL_KWARGS)

    return TrainConfig(
        name=task.name,
        # A fixed exp_name means <ckpt_root>/<name>/<exp_name> collides across reruns.
        # Pass --exp-name explicitly for real runs; the smoke script already passes
        # --exp-name=smoke --overwrite. Stamping the date here instead (as dexjoco does)
        # would make the config unreproducible.
        exp_name=task.name,
        model=model,
        data=SharpaDataConfig(
            root=dataset_root() / task.data_subdir,
            repo_id="local_repo",
            head_img_name=task.head_img_name,
            wrist_img_name=task.wrist_img_name,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=task.num_train_steps,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(pretrained_ckpt()),
        num_train_steps=task.num_train_steps,
        save_interval=5_000,
        wandb_enabled=True,
        checkpoint_base_dir=ckpt_root(),
    )


def get_sharpa_configs():
    return [make_sharpa_config(t) for t in SHARPA_TASKS]
