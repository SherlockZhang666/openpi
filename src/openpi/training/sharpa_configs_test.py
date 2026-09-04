"""Tests for the OpenArm(7) + Sharpa Wave left-hand(22) single-hand rig integration.

See tactile_steering/docs/sharpa-pi05-plan.md.
"""

import dataclasses
import inspect
import pathlib

from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
from openpi.policies import sharpa_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharpa_configs


def test_data_config_has_root_field():
    """Upstream DataConfig has no `root`. Without it a local LeRobot dataset can only
    be reached by guessing a path through HF_LEROBOT_HOME."""
    fields = {f.name for f in dataclasses.fields(_config.DataConfig)}
    assert "root" in fields
    assert _config.DataConfig().root is None


class _FakeMeta:
    """Stands in for LeRobotDatasetMetadata; records the kwargs it was constructed with."""

    def __init__(self, repo_id, root=None, **kwargs):
        self.repo_id = repo_id
        self.root = root
        self.fps = 10
        self.tasks = {}


class _FakeDataset:
    def __init__(self, repo_id, root=None, delta_timestamps=None, video_backend=None, **kwargs):
        self.repo_id = repo_id
        self.root = root
        self.delta_timestamps = delta_timestamps
        self.video_backend = video_backend

    def __len__(self):
        return 0


def _capture_lerobot_call(monkeypatch, data_config):
    """Run create_torch_dataset against stubbed LeRobot classes and return what they got."""
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMeta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeDataset)
    return _data_loader.create_torch_dataset(
        data_config, action_horizon=4, model_config=pi0_config.Pi0Config(pi05=True)
    )


def test_create_torch_dataset_forwards_root_to_lerobot(monkeypatch):
    """Adding the field is not enough -- data_loader has to actually pass it on.

    Both LeRobotDatasetMetadata and LeRobotDataset need it: the metadata object is what
    resolves fps and the task list, so pointing only the dataset at the local root would
    still send the metadata lookup to the hub.
    """
    dc = _config.DataConfig(
        repo_id="local_repo", root=pathlib.Path("/data/egg"), action_sequence_keys=("action",)
    )
    ds = _capture_lerobot_call(monkeypatch, dc)
    assert ds.root == pathlib.Path("/data/egg")
    assert ds.repo_id == "local_repo"
    # delta_timestamps has to be keyed on the *raw dataset* action key.
    assert list(ds.delta_timestamps) == ["action"]


def test_create_torch_dataset_keeps_upstream_behaviour_when_root_is_none(monkeypatch):
    """Regression guard on the upstream path: root=None must reach LeRobot as None so it
    resolves through the hub / HF_LEROBOT_HOME exactly as before this patch."""
    dc = _config.DataConfig(repo_id="physical-intelligence/libero")
    ds = _capture_lerobot_call(monkeypatch, dc)
    assert ds.root is None
    assert list(ds.delta_timestamps) == ["actions"]


def test_data_config_has_video_backend_field():
    """LeRobot's get_safe_default_codec() picks torchcodec on find_spec alone, without ever
    checking that it loads. It does not load on this cluster (no ffmpeg shared libs), so the
    default blows up on the first decoded frame -- during training, not at construction."""
    fields = {f.name for f in dataclasses.fields(_config.DataConfig)}
    assert "video_backend" in fields
    assert _config.DataConfig().video_backend is None


def test_create_torch_dataset_forwards_video_backend_to_lerobot(monkeypatch):
    dc = _config.DataConfig(repo_id="local_repo", video_backend="pyav")
    assert _capture_lerobot_call(monkeypatch, dc).video_backend == "pyav"


def test_create_torch_dataset_keeps_upstream_behaviour_when_video_backend_is_none(monkeypatch):
    """None must reach LeRobot as None so it keeps choosing its own default."""
    dc = _config.DataConfig(repo_id="physical-intelligence/libero")
    assert _capture_lerobot_call(monkeypatch, dc).video_backend is None


def test_sharpa_data_config_pins_pyav():
    """The whole point of the field: SharpaDataConfig must not leave the choice to LeRobot."""
    dc = _config.SharpaDataConfig(root=pathlib.Path("/x"), repo_id="local_repo")
    created = dc.create(pathlib.Path("/tmp/assets"), pi0_config.Pi0Config(pi05=True))
    assert created.video_backend == "pyav"


def test_data_loader_patch_is_still_in_place():
    """Structural backstop for the upstream-file edits, which are the ones most
    likely to be silently dropped on a rebase onto upstream."""
    src = inspect.getsource(_data_loader.create_torch_dataset)
    assert "data_config.root" in src
    assert src.count("root=root") >= 2, "both LeRobotDatasetMetadata and LeRobotDataset need root="
    assert "video_backend=data_config.video_backend" in src


# ---------------------------------------------------------------------------
# Task 3: SharpaDataConfig -- LeRobot key remapping
# ---------------------------------------------------------------------------


def _model_cfg():
    return pi0_config.Pi0Config(**sharpa_configs._MODEL_KWARGS)  # noqa: SLF001


def test_repack_maps_dataset_keys_to_policy_keys():
    dc = _config.SharpaDataConfig(root=pathlib.Path("/nonexistent"), repo_id="local_repo")
    created = dc.create(pathlib.Path("/nonexistent/assets"), _model_cfg())
    structure = created.repack_transforms.inputs[0].structure
    assert structure == {
        "base": "observation.images.head",
        "wrist": "observation.images.wrist",
        "state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    }


def test_head_and_wrist_img_names_are_overridable():
    dc = _config.SharpaDataConfig(
        root=pathlib.Path("/nonexistent"),
        repo_id="local_repo",
        head_img_name="observation.images.head_left",
        wrist_img_name="observation.images.wrist_left",
    )
    structure = dc.create(pathlib.Path("/x"), _model_cfg()).repack_transforms.inputs[0].structure
    assert structure["base"] == "observation.images.head_left"
    assert structure["wrist"] == "observation.images.wrist_left"


def test_create_propagates_action_sequence_key_and_root():
    """Both fields have to be forwarded into the returned DataConfig explicitly.

    action_sequence_keys is used by create_torch_dataset to build delta_timestamps over
    the *raw dataset* keys, before the repack transform runs. Its default is ("actions",)
    while the dataset's key is "action", so not forwarding it is a KeyError at dataset
    construction. root is what actually locates a dataset that only exists on scratch.
    """
    dc = _config.SharpaDataConfig(root=pathlib.Path("/data/egg"), repo_id="local_repo")
    created = dc.create(pathlib.Path("/x"), _model_cfg())
    assert created.action_sequence_keys == ("action",)
    assert created.root == pathlib.Path("/data/egg")


def test_data_transforms_use_sharpa_policy():
    dc = _config.SharpaDataConfig(root=pathlib.Path("/x"), repo_id="local_repo")
    created = dc.create(pathlib.Path("/x"), _model_cfg())
    assert isinstance(created.data_transforms.inputs[0], sharpa_policy.SharpaInputs)
    assert isinstance(created.data_transforms.outputs[0], sharpa_policy.SharpaOutputs)
    assert created.data_transforms.inputs[0].model_type == _model.ModelType.PI05


def test_no_delta_action_transform_is_applied():
    """The 28 dims do not share a meaning (fingers absolute, wrist already delta), so a
    blanket DeltaActions would contradict palm_centric. Pin its absence."""
    dc = _config.SharpaDataConfig(root=pathlib.Path("/x"), repo_id="local_repo")
    created = dc.create(pathlib.Path("/x"), _model_cfg())
    names = [type(t).__name__ for t in created.data_transforms.inputs]
    assert "DeltaActions" not in names


# ---------------------------------------------------------------------------
# Task 4: task table and TrainConfig registration
# ---------------------------------------------------------------------------


def test_four_tasks_are_defined():
    names = [t.name for t in sharpa_configs.SHARPA_TASKS]
    assert names == ["sharpa_egg", "sharpa_tissue", "sharpa_card", "sharpa_tablet"]


def test_model_keeps_native_action_dim_32():
    cfg = sharpa_configs.make_sharpa_config(sharpa_configs.SHARPA_TASKS[0])
    assert cfg.model.action_dim == 32, "28 <= 32; do NOT run any action-dim conversion script"
    assert cfg.model.action_horizon == 30
    # Pi0Config.model_type returns PI05 whenever pi05=True -- not PI0.
    assert cfg.model.model_type == _model.ModelType.PI05


def test_state_input_is_discretised():
    """A deliberate choice, not a drifting default.

    pi05=True makes __post_init__ set discrete_state_input=True; upstream's own
    pi05_libero sets it to False explicitly. We follow dexjoco and take True. This
    assertion exists so that changing it has to be deliberate.
    """
    cfg = sharpa_configs.make_sharpa_config(sharpa_configs.SHARPA_TASKS[0])
    assert cfg.model.discrete_state_input is True


def test_lora_variants_and_freeze_filter_are_consistent():
    """The freeze filter has to be derived from the *same* model config as the model
    itself; deriving it from a differently-parameterised Pi0Config would freeze the
    wrong parameters silently."""
    cfg = sharpa_configs.make_sharpa_config(sharpa_configs.SHARPA_TASKS[0])
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    assert cfg.model.action_expert_variant == "gemma_300m_lora"
    assert cfg.model.max_token_len == 250
    assert cfg.freeze_filter == cfg.model.get_freeze_filter()
    assert cfg.ema_decay is None


def test_dataset_root_comes_from_the_env_and_task_subdir(monkeypatch):
    monkeypatch.setenv("SHARPA_DATASET_ROOT", "/tmp/sharpa_data")
    cfg = sharpa_configs.make_sharpa_config(sharpa_configs.SHARPA_TASKS[0])
    assert cfg.data.root == pathlib.Path("/tmp/sharpa_data/pick_up_the_egg")


def test_configs_are_registered_and_retrievable():
    cfg = _config.get_config("sharpa_egg")
    assert isinstance(cfg.data, _config.SharpaDataConfig)


def test_config_names_are_unique_across_all_openpi_configs():
    names = [c.name for c in _config._CONFIGS]  # noqa: SLF001
    assert len(names) == len(set(names))


def test_module_import_does_not_read_files_from_cwd(tmp_path, monkeypatch):
    """dexjoco's dexjoco_configs.py does open('config.yaml') at module level, which blows
    up depending on the cwd. Do not repeat that."""
    import importlib

    monkeypatch.chdir(tmp_path)
    importlib.reload(sharpa_configs)
    assert sharpa_configs.SHARPA_TASKS
