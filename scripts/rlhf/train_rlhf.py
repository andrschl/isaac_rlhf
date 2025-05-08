# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train an RL agent with RLHF."""

import argparse
import os
import multiprocessing as mp

from isaac_rlhf.runners import RlhfRunner
from isaac_rlhf.config import RlhfCfg


def main(args_cli):
    kwargs = {k: v for k, v in vars(args_cli).items() if v is not None}
    cfg = RlhfCfg(**kwargs)
    rlhf = RlhfRunner(cfg)

    rlhf.run()

    import torch

    torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Train an RL agent with RLHF.")

    # Environment arguments
    parser.add_argument("--task", type=str, help="Name of the task.")
    parser.add_argument(
        "--num_envs",
        type=int,
        help="Number of environments to use for training. If None, it will use the default value for the task.",
    )
    # RLHF arguments
    parser.add_argument(
        "--num_rlhf_iterations",
        type=int,
        help="The number of RLHF iterations to run.",
    )
    parser.add_argument(
        "--rlhf_algorithm",
        type=str,
        choices=["vanilla", "ts_double", "ts_last", "rl"],
        help="The RLHF algorithm to use.",
    )
    parser.add_argument(
        "--beta1", type=float, help="The beta parameter for Thompson sampling."
    )
    parser.add_argument(
        "--beta2", type=float, help="The beta parameter for Thompson sampling."
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        help="The lambda parameter for Thompson sampling.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Use lazy Thompson sampling.",
    )
    parser.add_argument(
        "--lazy_constant",
        type=float,
        help="The lazy constant for Thompson sampling.",
    )
    parser.add_argument(
        "--opt_design",
        action="store_true",
        help="Use optimal design for Thompson sampling.",
    )
    parser.add_argument(
        "--mle_l2_reg",
        type=float,
        help="The L2 regularization parameter for MLE.",
    )

    # RL arguments
    parser.add_argument(
        "--num_rl_runs",
        type=int,
        help="Number of RL runs per reward iteration. Should be multiple of two.",
    )
    parser.add_argument(
        "--num_trajectories_per_run",
        type=int,
        help="The number of RL steps to run per RL run.",
    )
    parser.add_argument(
        "--num_rl_iterations",
        type=int,
        help="The number of RL training iterations per RL run.",
    )
    parser.add_argument(
        "--rl_library",
        type=str,
        choices=["rsl_rl", "rl_games", "skrl"],
        help="The RL training library to use.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )

    # System arguments
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to run training on."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="The number of processes to use for training.",
    )
    parser.add_argument(
        "--base_seed", type=int, help="The random seed to use for the environment."
    )

    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)
