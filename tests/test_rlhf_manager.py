import multiprocessing

from isaac_rlhf.config import RlhfCfg
from isaac_rlhf.algorithms.rlhf import WorkerTask, RlhfTaskManager


def _grid_cfg(**overrides):
    cfg = RlhfCfg()
    cfg.task = "gridworld"
    cfg.device = "cpu"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_worker_initializes_shared_constants():
    cfg = _grid_cfg()
    rewards_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    termination_event = multiprocessing.Event()
    shared_data = {}
    constants_event = multiprocessing.Event()
    constants_lock = multiprocessing.Lock()

    worker = WorkerTask(
        idx=0,
        rewards_queue=rewards_queue,
        results_queue=results_queue,
        termination_event=termination_event,
        cfg=cfg,
        shared_data=shared_data,
        constants_event=constants_event,
        constants_lock=constants_lock,
    )

    assert constants_event.is_set()
    assert "gt_params" in shared_data and "dt" in shared_data
    assert shared_data["dt"] == 1.0
    assert len(shared_data["gt_params"]) > 0

    # cleanup queues used for construction
    rewards_queue.close()
    results_queue.close()


def test_task_manager_loads_shared_constants():
    cfg = _grid_cfg(num_rl_runs=3)

    manager = RlhfTaskManager.__new__(RlhfTaskManager)
    manager.cfg = cfg
    manager.constants_event = multiprocessing.Event()
    manager.constants_event.set()
    manager.constants_loaded = False
    manager.shared_data = {"gt_params": {"a": 1.0, "b": 2.0}, "dt": 0.2}
    manager.reward_params = []

    # bind instance methods
    manager.init_from_shared_data = RlhfTaskManager.init_from_shared_data.__get__(manager)
    manager.ensure_constants_loaded = RlhfTaskManager.ensure_constants_loaded.__get__(manager)

    manager.ensure_constants_loaded()

    assert cfg.gt_params == {"a": 1.0, "b": 2.0}
    assert cfg.num_features == 2
    assert len(manager.reward_params) == cfg.num_rl_runs
    for param in manager.reward_params:
        assert param.shape[0] == cfg.num_features
