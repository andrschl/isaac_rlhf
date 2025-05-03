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

    def __init__(self, cfg: RlhfCfg):
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
        if wandb.run is None:
            wandb.init(
                project=f"isaac_rlhf",
                dir=self.log_dir,
                config=cfg.to_dict(),
                name=f"{cfg.rlhf_algorithm}{'_lazy' if cfg.lazy else ''}_{timestamp}",
                group=f"{cfg.task}",
            )
        else:
            # running under a sweep agent, just update the config from sweep
            wandb.config.update(cfg.to_dict(), allow_val_change=True)
        self.writer = wandb
        print("[INFO]: RLHF Task Manager setup complete.")

    def run(self):
        """
        Run the RLHF training loop.
        """

        for iter in range(self.num_rlhf_iterations):
            print(f"\n{'#' * 20} Running RLHF Iteration {iter} {'#' * 20} \n")
            wandb.log({"Step": iter})
            # Train the RL agent
            print(
                "[INFO]: Training RL agent with the following reward parameters:",
                self.task_manager.reward_params,
            )
            results = self.task_manager.distribute_rewards()

            # Observe feedback and update reward
            print("[INFO]: Observing preference feedback and update reward...")
            if self.task_manager.query_now():
                self.task_manager.get_preferences()
                self.task_manager.mle_update()
            self.task_manager.sample_reward_params()

            # Logging
            print("[INFO]: Logging...")
            logdict_wandb = {
                "rlhf/gt_reward": self.task_manager.get_gt_reward(results),
                "rlhf/pred_reward": self.task_manager.get_pred_reward(results),
                "rlhf/pred_reward_debug": sum(
                    [result["mean_episode_reward"] for result in results]
                )
                / len(results),
                "rlhf/reward_error": self.task_manager.get_reward_error(),
                "rlhf/lambda_max(V_inv)": self.task_manager.get_V_inv_eigenvalues()
                .max()
                .item(),
                "rlhf/lambda_min(V_inv)": self.task_manager.get_V_inv_eigenvalues()
                .min()
                .item(),
            }
            logdict_console = logdict_wandb.copy()
            logdict_console["reward_params"] = self.task_manager.reward_params
            logdict_console["reward_params_gt"] = (
                self.task_manager.gt_params_as_tensor()
            )
            self.logging_step(logdict_wandb, logdict_console, iter)

        self.save_final_results()

        print("[INFO]: RLHF training completed.")
        self.task_manager.close()
        wandb.finish()

    def logging_step(self, logdict_wandb, logdict_console, step):
        """Log and print the results."""
        print(f"{'#' * 20} RLHF step {step} {'#' * 20}")
        for key, value in logdict_console.items():
            print(f"{key}: {value}")
        print(
            "[DEBUG] "
            + ", ".join([f"{key}: {value}" for key, value in logdict_wandb.items()])
        )
        wandb.log(logdict_wandb)

    def save_final_results(self):
        """Save the final results."""
        self.task_manager.save_results(self.log_dir)
        print(f"[INFO]: Final results saved to {self.log_dir}")
