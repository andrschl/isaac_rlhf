# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import numpy as np
import os
import wandb
from typing import Literal, Optional

from isaac_rlhf.algorithms.rlhf import RlhfTaskManager
from isaac_rlhf.utils.rlhf_utils import load_tensorboard_logs

class RlhfRunner:
    """Runs Rlhf training for a given task."""

    def __init__(
        self,
        task: str,
        rl_library: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl",
        rlhf_algorithm: Literal["vanilla", "ts"] = "vanilla",
        num_rl_runs: int = 1,
        num_rl_iterations: int = 10,
        num_rlhf_iterations: int = 10,
        num_processes: int = 2,
        base_seed: int = 42,
        num_envs: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Initialize the RlhfRunner.

        Args:  
            task: The name of the task to run.
            rl_library: The RL library to use for training.
            rlhf_algorithm: The RLHF algorithm to use.
            num_rl_runs: The number of parallel runs to execute.
            base_seed: The random seed to use for the environment.
            num_rl_iterations: The maximum number of training iterations to run.
            device: The device to run the training on.
        """

        self.num_processes = num_processes
        self.num_rlhf_iterations = num_rlhf_iterations
        self.num_features = None
        self.gt_params = None

        print("[INFO]: Setting up the RLHF Task Manager...")
        self.task_manager = RlhfTaskManager(
            task=task,
            device=device,
            base_seed=base_seed,
            rl_library=rl_library,
            rlhf_algorithm=rlhf_algorithm,
            num_processes=num_processes,
            num_rl_runs=num_rl_runs,
            num_rl_iterations=num_rl_iterations,
            num_envs=num_envs,
        )

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("logs", "rlhf", task, timestamp)
        os.makedirs(self.log_dir)
        # init wandb
        wandb.init(
            project="isaac_rlhf",
            dir=self.log_dir,
            config={  # log CLI args or configs if you like
                "task": task,
                "rl_library": rl_library,
                "rlhf_algorithm": rlhf_algorithm,
                "num_rl_runs": num_rl_runs,
            },
        )
        self.writer = wandb
        print("[INFO]: RLHF Task Manager setup complete.")

    def run(self):
        """Run the RLHF training loop.

        Args:
            num_rlhf_iterations: The maximum number of RLHF iterations to run.
        """

        for iter in range(self.num_rlhf_iterations):
            print(f"\n{'#' * 20} Running RLHF Iteration {iter} {'#' * 20} \n")

            # Update reward function
            print("[INFO]: Updating the reward function...")
            reward_params = self.task_manager.get_reward_params()

            # Train the RL agent
            print("[INFO]: Training RL agent with the following reward parameters:", reward_params)
            results = self.task_manager.distribute_rewards(reward_params)

            # Observe feedback
            print("[INFO]: Observing preference feedback...")
            self.task_manager.get_preferences()

            # Logging
            print("[INFO]: Logging...")
            logdict = {
                    "gt_reward": self.task_manager.get_gt_reward(),
                    "pred_reward": self.task_manager.get_pred_reward(),
                    "reward_error": self.task_manager.get_reward_error(),
                    "reward_params": reward_params,
                    "V_inv eigenvalues": self.task_manager.get_V_inv_eigenvalues()
                }
            self.logging_step(logdict, iter)

        self.save_final_results()

        print("[INFO]: RLHF training completed.")
        self.task_manager.close()

    def logging_step(self, logdict, step):
        """Log and print the results."""
        print(f"{'#' * 20} RLHF step {step} {'#' * 20}")
        for key, value in logdict.items():
            print(f"{key}: {value}")
        self.writer.log(logdict, step=step)

    def save_final_results(self):
        """Save the final results."""
        self.task_manager.save_results(self.log_dir)
        print(f"[INFO]: Final results saved to {self.log_dir}")