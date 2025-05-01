# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import numpy as np
import os
import wandb
from typing import Literal, Optional

from isaac_rlhf.algorithms.rlhf import RlhfTaskManager
from isaac_rlhf.config import RlhfCfg


class RlhfRunner:
    """Runs Rlhf training for a given task."""

    def __init__(
        self,
        cfg: RlhfCfg
    ):
        """
        Initialize the RlhfRunner.
        """
        
        self.num_rlhf_iterations = cfg.num_rlhf_iterations

        print("[INFO]: Setting up the RLHF Task Manager...")
        self.task_manager = RlhfTaskManager(cfg)

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("logs", "rlhf", cfg.task, timestamp)
        os.makedirs(self.log_dir)
        # init wandb
        wandb.init(
            project="isaac_rlhf",
            dir=self.log_dir,
            config=cfg.to_dict(),
            name=f"{cfg.task}_{cfg.rlhf_algorithm}_{timestamp}",
        )
        self.writer = wandb
        print("[INFO]: RLHF Task Manager setup complete.")

    def run(self):
        """
        Run the RLHF training loop.
        """

        for iter in range(self.num_rlhf_iterations):
            print(f"\n{'#' * 20} Running RLHF Iteration {iter} {'#' * 20} \n")

            # Update reward function
            print("[INFO]: Updating the reward function...")
            reward_params = self.task_manager.sample_reward_params()

            # Train the RL agent
            print("[INFO]: Training RL agent with the following reward parameters:", reward_params)
            results = self.task_manager.distribute_rewards()

            # Observe feedback
            print("[INFO]: Observing preference feedback...")
            self.task_manager.get_preferences()

            # Logging
            print("[INFO]: Logging...")
            logdict_wandb = {
                "gt_reward": self.task_manager.get_gt_reward(results),
                "pred_reward": self.task_manager.get_pred_reward(results), 
                "pred_reward_debug": sum([result["mean_episode_reward"] for result in results]) / len(results),
                "reward_error": self.task_manager.get_reward_error(),
                "lambda_max(V_inv)": self.task_manager.get_V_inv_eigenvalues().max(),
                "lambda_min(V_inv)": self.task_manager.get_V_inv_eigenvalues().min(),
            }
            logdict_console = logdict_wandb.copy()
            logdict_console["reward_params"] = reward_params,
            logdict_console["reward_params_gt"] = self.task_manager.gt_params_as_tensor().tolist()
            self.logging_step(logdict_wandb, logdict_console, iter)

        self.save_final_results()

        print("[INFO]: RLHF training completed.")
        self.task_manager.close()

    def logging_step(self, logdict_wandb, logdict_console, step):
        """Log and print the results."""
        print(f"{'#' * 20} RLHF step {step} {'#' * 20}")
        for key, value in logdict_console.items():
            print(f"{key}: {value}")
        self.writer.log(logdict_wandb, step=step)

    def save_final_results(self):
        """Save the final results."""
        self.task_manager.save_results(self.log_dir)
        print(f"[INFO]: Final results saved to {self.log_dir}")