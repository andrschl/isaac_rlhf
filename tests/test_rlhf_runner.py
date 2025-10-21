import types

import pytest

from isaac_rlhf.config import RlhfCfg
from isaac_rlhf.runners import rlhf_runner


class DummyTaskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reward_params = []

    def close(self):
        pass


class DummyWandb:
    def __init__(self):
        self.run = None
        self.logged = []
        self.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)
        self.init_kwargs = None

    def login(self, key=None):
        return None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = types.SimpleNamespace()
        return self.run

    def log(self, data, step=None):
        self.logged.append((data, step))

    def finish(self):
        self.run = None


@pytest.fixture(autouse=True)
def stub_wandb(monkeypatch):
    dummy = DummyWandb()
    monkeypatch.setattr(rlhf_runner, "wandb", dummy)
    yield dummy


@pytest.fixture(autouse=True)
def stub_task_manager(monkeypatch):
    monkeypatch.setattr(rlhf_runner, "RlhfTaskManager", DummyTaskManager)


def test_log_dir_for_ts_double(monkeypatch, stub_wandb):
    created_paths = []
    monkeypatch.setattr(rlhf_runner.os, "makedirs", lambda path, exist_ok=True: created_paths.append(path))
    monkeypatch.setenv("ISAAC_RLHF_SWEEP_GROUP", "unit_sweep")

    cfg = RlhfCfg(
        task="gridworld",
        device="cpu",
        rlhf_algorithm="ts_double",
        beta1=0.1,
        beta2=0.2,
        base_seed=99,
    )

    runner = rlhf_runner.RlhfRunner(cfg)

    assert runner.log_dir.endswith("ts_double/beta1_0.1/beta2_0.2/seed_99")
    assert created_paths and created_paths[0] == runner.log_dir
    assert stub_wandb.init_kwargs is not None
    assert stub_wandb.init_kwargs["dir"] == runner.log_dir
    assert stub_wandb.init_kwargs["group"] == "unit_sweep"
    run_name = stub_wandb.init_kwargs["name"]
    assert "ts_double" in run_name
    assert "beta1_0.1" in run_name
    assert "beta2_0.2" in run_name
    assert "seed_99" in run_name
    assert "tuning" in stub_wandb.init_kwargs["tags"]


def test_runner_rejects_unknown_algorithm(monkeypatch):
    monkeypatch.setattr(rlhf_runner.os, "makedirs", lambda *_, **__: None)
    monkeypatch.delenv("ISAAC_RLHF_SWEEP_GROUP", raising=False)

    cfg = RlhfCfg(task="gridworld", device="cpu", rlhf_algorithm="unsupported")

    with pytest.raises(ValueError):
        rlhf_runner.RlhfRunner(cfg)
