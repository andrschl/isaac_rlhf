from dataclasses import dataclass, asdict, replace as dc_replace, field
from typing import Optional, Literal
import torch


@dataclass
class RlhfCfg:
    """Configuration for RLHF training."""

    # Environment arguments
    task: str = "Isaac-Cartpole-v0"
    num_envs: Optional[int] = None
    num_features: Optional[int] = None
    gt_params: Optional[torch.Tensor] = None
    dt: Optional[float] = None

    # RLHF arguments
    num_rlhf_iterations: int = 10
    rlhf_algorithm: Literal["vanilla", "ts_double", "ts_last"] = "vanilla"
    num_rl_runs: int = 2
    num_trajectories_per_run: int = 100
    trajectory_length: int = 150

    # MLE arguments
    num_mle_iterations: int = 100
    mle_lr: float = 1e-3
    mle_l2_reg: float = 1e-6
    mle_epochs: int = 500
    mle_batch_size: int = 64

    # RL arguments
    num_rl_iterations: int = 100
    rl_library: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl"

    # System arguments
    base_seed: int = 42
    num_processes: int = 1
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)

    def replace(self, **kwargs):
        return dc_replace(self, **kwargs)
