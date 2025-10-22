from dataclasses import dataclass, asdict, replace as dc_replace, field
from typing import Optional, Literal
import torch


@dataclass
class RlhfCfg:
    """Configuration for RLHF training."""

    # Environment arguments (any manager-based env or gridworld e.g {"Isaac-Velocity-Flat-H1-v0", "Isaac-Reach-Franka-v0", "Isaac-Cartpole-v0", "gridworld"})
    task: str = (
        "gridworld"
    )
    num_envs: Optional[int] = None
    num_features: Optional[int] = None
    gt_params: Optional[torch.Tensor] = None
    dt: Optional[float] = None

    # RLHF arguments
    num_rlhf_iterations: int = 30
    rlhf_algorithm: Literal["vanilla", "ts_double", "ts_last", "rl"] = "ts_last"
    num_rl_runs: int = 1
    num_trajectories_per_run: int = 100
    trajectory_length: int = 150
    beta1: float = 1
    beta2: float = 1
    lambda_: float = 1.0
    lazy: bool = False
    lazy_constant: float = 2.0
    opt_design: bool = False
    ignored_reward_terms: list[str] = field(default_factory=lambda: [])
    pure_exploration: bool = False
    # ignored_reward_terms: list[str] = field(default_factory=lambda: ["terminating"])   # never observed in cartpole

    # MLE arguments
    #num_mle_iterations: int = 100
    mle_lr: float = 1e-2
    mle_l2_reg: float = 1e-6
    mle_epochs: int = 50
    mle_batch_size: int = 256

    # RL arguments
    num_rl_iterations: int = 100
    rl_library: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl"
    resume: bool = False

    # Tabular specific RL arguments
    entropy_coeff: float = 1e-3
    tabular_alg: str = "svi"    # only for tabular envs: "svi" (soft_policy_iteration), or "npg" (natural policy gradient)

    # System arguments
    base_seed: int = 42
    num_processes: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return asdict(self)

    def replace(self, **kwargs):
        return dc_replace(self, **kwargs)
