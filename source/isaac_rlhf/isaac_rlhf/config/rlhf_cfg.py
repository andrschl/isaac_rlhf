from dataclasses import dataclass, asdict, replace as dc_replace, field
from typing import Optional, Literal
import torch

ENV_ID_TO_RL_TASK = {
    "Isaac-Humanoid-v0": "humanoid",
    "Isaac-Cartpole-v0": "cartpole",
    "Isaac-Cartpole-RGB-v0": "cartpole",
    "Isaac-Reach-Franka-v0": "reach",
    "Isaac-Lift-Cube-Franka-v0": "lift",
    "Isaac-Open-Drawer-Franka-v0": "cabinet",
    "Isaac-Velocity-Flat-Anymal-B-v0": "velocity",
}


@dataclass
class RlhfCfg:
    """Configuration for RLHF training."""

    # Environment arguments
    task: str = "Isaac-Cartpole-v0"
    #  "Isaac-Velocity-Flat-H1-v0"  # "Isaac-Reach-Franka-v0"  # "Isaac-Cartpole-v0" "Isaac-Humanoid-v0"
    # )
    num_envs: Optional[int] = None
    num_features: Optional[int] = None
    gt_params: Optional[torch.Tensor] = None
    dt: Optional[float] = None
    rl_task_type = ENV_ID_TO_RL_TASK.get(task, "cartpole")
    # RLHF arguments
    num_rlhf_iterations: int = 30
    rlhf_algorithm: Literal["vanilla", "ts_double", "ts_last", "ts_nC2", "rl"] = "vanilla"
    num_rl_runs: int = 1
    num_trajectories_per_run: int = 10
    trajectory_length: int = 150
    beta1: float = 1.0
    beta2: float = 1.0
    lambda_: float = 1.0
    lazy: bool = False
    lazy_constant: float = 2.0
    opt_design: bool = False
    ignored_reward_terms: list[str] = field(default_factory=lambda: [])
    pure_exploration: bool = False
    # ignored_reward_terms: list[str] = field(default_factory=lambda: ["terminating"])   # never observed in cartpole

    # MLE arguments
    num_mle_iterations: int = 100
    mle_lr: float = 1e-3
    mle_l2_reg: float = 1e-6
    mle_epochs: int = 500
    mle_batch_size: int = 64

    # RL arguments
    num_rl_iterations: int = 30
    rl_library: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl"
    resume: bool = False

    # System arguments
    base_seed: int = 42
    num_processes: int = 1
    device: str = "cuda"
    preference_mode: str = "llm"

    def to_dict(self):
        return asdict(self)

    def replace(self, **kwargs):
        return dc_replace(self, **kwargs)
