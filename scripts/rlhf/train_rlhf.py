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

    cfg = RlhfCfg(
        # Environment arguments
        task=args_cli.task,
        num_envs=args_cli.num_envs,
        # RLHF arguments
        num_rlhf_iterations=args_cli.num_rlhf_iterations,
        rlhf_algorithm=args_cli.rlhf_algorithm,
        # MLE arguments
        # RL arguments
        num_rl_runs=args_cli.num_rl_runs,
        num_trajectories_per_run=args_cli.num_trajectories_per_run,
        num_rl_iterations=args_cli.num_rl_iterations,
        rl_library=args_cli.rl_library,
        # System arguments
        base_seed=args_cli.base_seed,
        num_processes=1,
        device=args_cli.device,
    )
    rlhf = RlhfRunner(rlhf_config)

    rlhf.run()


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
    parser.add_argument("--num_rlhf_iterations", type=int, default=10, help="The number of RLHF iterations to run.")
    parser.add_argument(
        "--rlhf_algorithm",
        type=str,
        choices=["vanilla", "ts_double", "ts_last"],
        help="The RLHF algorithm to use.",
    )
    
    # RL arguments
    parser.add_argument(
        "--num_rl_runs", type=int, default=2, help="Number of RL runs per reward iteration. Should be multiple of two."
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

    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument(
        "--num_processes",
        type=int,
        help="The number of processes to use for training.",
    )
    parser.add_argument("--base_seed", type=int, help="The random seed to use for the environment.")
    
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)