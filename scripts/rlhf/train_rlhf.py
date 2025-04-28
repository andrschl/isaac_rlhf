# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train an RL agent with RLHF."""

import argparse
import os
import multiprocessing as mp

from isaac_rlhf.runners import RlhfRunner

def main(args_cli):
    rlhf = RlhfRunner(
        task=args_cli.task,
        rl_library=args_cli.rl_library,
        num_rl_runs=args_cli.num_rl_runs,
        num_rl_iterations=args_cli.num_rl_iterations,
        rlhf_algorithm=args_cli.rlhf_algorithm,
        num_rlhf_iterations=args_cli.num_rlhf_iterations,
        base_seed=args_cli.base_seed,
        device=args_cli.device,
    )

    rlhf.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Train an RL agent with RLHF.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Name of the task.")
    parser.add_argument(
        "--num_rl_runs", type=int, default=2, help="Number of RL runs per reward iteration."
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--base_seed", type=int, default=42, help="The random seed to use for the environment.")
    parser.add_argument("--num_rlhf_iterations", type=int, default=10, help="The number of RLHF iterations to run.")
    parser.add_argument(
        "--num_rl_iterations",
        type=int,
        default=10,
        help="The number of RL training iterations per RL run.",
    )
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games", "skrl"],
        help="The RL training library to use.",
    )
    parser.add_argument(
        "--rlhf_algorithm",
        type=str,
        default="vanilla",
        choices=["vanilla", "ts"],
        help="The RLHF algorithm to use.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of environments to use for training. If None, it will use the default value for the task.",
    )
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)