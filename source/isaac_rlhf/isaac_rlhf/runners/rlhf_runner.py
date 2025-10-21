# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
from dotenv import load_dotenv

# load .env into os.environ
load_dotenv()

import datetime
import pandas as pd
import wandb

from isaac_rlhf.algorithms.rlhf import RlhfTaskManager
from isaac_rlhf.config import RlhfCfg


DEBUG_ENABLED = os.environ.get("ISAAC_RLHF_DEBUG", "0").lower() not in {
    "0",
    "false",
    "no",
    "",
}


def debug_print(*args, **kwargs):
    if DEBUG_ENABLED:
        print(*args, **kwargs)


class RlhfRunner:
    """Runs Rlhf training for a given task."""

    def __init__(self, cfg: RlhfCfg):
        """
        Initialize the RlhfRunner.
        """

        self.num_rlhf_iterations = cfg.num_rlhf_iterations

        print("[INFO]: Setting up the RLHF Task Manager...")
        self.task_manager = RlhfTaskManager(cfg)

        # Setup logging directory and wandb
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if cfg.resume:
            base_dir = os.path.join("logs", "rlhf", cfg.task, "resume")
        else:
            base_dir = os.path.join("logs", "rlhf", cfg.task)
        self.log_dir = self._build_log_dir(cfg, base_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[INFO]: Logging directory: {self.log_dir}")

        # Login to wandb using the key from .env (not hardcoded)
        try:
            api_key = os.getenv("WANDB_API_KEY_ANDREAS")
            wandb.login(key=api_key)
        except Exception as e:
            print(f"[WARNING]: Could not login to wandb: {e}")
            print("[INFO]: Continuing without wandb logging...")
        
        # init wandb
        run_name = self._build_run_name(cfg, timestamp)
        sweep_group = os.environ.get("ISAAC_RLHF_SWEEP_GROUP")
        group_name = sweep_group or f"{cfg.task}_{cfg.rlhf_algorithm}"

        # Add tags for easier filtering during hyperparameter tuning
        tags = [cfg.task, cfg.rlhf_algorithm]
        if cfg.lazy:
            tags.append("lazy")
        tags.append(f"beta1_{cfg.beta1}")
        tags.append(f"beta2_{cfg.beta2}")
        if sweep_group:
            tags.append("tuning")

        if wandb.run is None:
            wandb.init(
                project="isaac_rlhf",
                dir=self.log_dir,
                config=cfg.to_dict(),
                name=run_name,
                group=group_name,
                tags=tags,
            )
        else:
            # running under a sweep agent, just update the config from sweep
            wandb.config.update(cfg.to_dict(), allow_val_change=True)
        self.writer = wandb
        self.log_history: dict[str, list] = {}
        print("[INFO]: RLHF Task Manager setup complete.")

    def run(self):
        """
        Run the RLHF training loop.
        """
        lazy_update_count = 0
        query_count = 0
        cumulative_gt_reward = 0.0
        for iter in range(self.num_rlhf_iterations + 1):

            self.task_manager.rlhf_iter = iter
            print(f"\n{'#' * 20} Running RLHF Iteration {iter} {'#' * 20} \n")

            # Train the RL agent
            print(
                "[INFO]: Training RL agent with the following reward parameters:",
                self.task_manager.reward_params,
            )
            results = self.task_manager.distribute_rewards()
            self.task_manager.check_results(results)

            # Observe feedback and update reward
            print("[INFO]: Observing preference feedback and update reward...")
            if self.task_manager.query_now():
                _, y_new = self.task_manager.get_preferences()
                query_count += len(y_new)
                self.task_manager.mle_update()
                lazy_update_count += 1

            # Sample new reward parameters
            self.task_manager.sample_reward_params()

            # Logging
            print("[INFO]: Logging...")
            gt_reward = self.task_manager.get_gt_reward(results)
            cumulative_gt_reward += gt_reward
            pred_reward = self.task_manager.get_pred_reward(results)
            mean_policy_reward = sum(
                [result["mean_episode_reward"] for result in results]
            ) / len(results)
            logdict_wandb = {
                "rlhf/gt_reward": gt_reward,
                "rlhf/cumulative_gt_reward": cumulative_gt_reward,
                "rlhf/pred_reward": pred_reward,
                "rlhf/pred_reward_debug": mean_policy_reward,
                "rlhf/reward_error": self.task_manager.get_reward_error(),
                "rlhf/lambda_max(V_inv)": self.task_manager.get_V_inv_eigenvalues()
                .max()
                .item(),
                "rlhf/lambda_min(V_inv)": self.task_manager.get_V_inv_eigenvalues()
                .min()
                .item(),
                "rlhf/lazy_update_count": lazy_update_count,
                "rlhf/num_queries": query_count,
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
        debug_print(
            "[DEBUG] "
            + ", ".join([f"{key}: {value}" for key, value in logdict_wandb.items()])
        )
        # append into your buffers
        for k, v in logdict_wandb.items():
            self.log_history.setdefault(k, []).append(v)
        self.log_history.setdefault("step", []).append(step)
        wandb.log(logdict_wandb, step=step)

    def save_final_results(self):
        """Save the final results."""
        df = pd.DataFrame(self.log_history)
        csv_path = os.path.join(self.log_dir, "logs.csv")
        df.to_csv(csv_path, index=False)
        self.task_manager.save_results(self.log_dir)
        print(f"[INFO]: Final results saved to {self.log_dir}")

    @staticmethod
    def _format_value(value):
        if isinstance(value, float):
            return format(value, "g")
        return str(value)

    def _build_log_dir(self, cfg: RlhfCfg, base_dir: str) -> str:
        formatted_beta1 = self._format_value(cfg.beta1)
        formatted_beta2 = self._format_value(cfg.beta2)

        if cfg.rlhf_algorithm == "ts_last":
            suffix = cfg.rlhf_algorithm
            if cfg.lazy and cfg.opt_design:
                suffix += "_lazy_opt_design"
            elif cfg.lazy:
                suffix += "_lazy"
            parts = [
                base_dir,
                suffix,
                f"beta1_{formatted_beta1}",
                f"beta2_{formatted_beta2}",
                f"seed_{cfg.base_seed}",
            ]
        elif cfg.rlhf_algorithm == "ts_double":
            parts = [
                base_dir,
                cfg.rlhf_algorithm,
                f"beta1_{formatted_beta1}",
                f"beta2_{formatted_beta2}",
                f"seed_{cfg.base_seed}",
            ]
        elif cfg.rlhf_algorithm in {"vanilla", "rl"}:
            parts = [base_dir, cfg.rlhf_algorithm, f"seed_{cfg.base_seed}"]
        else:
            raise ValueError(
                f"Unsupported rlhf_algorithm '{cfg.rlhf_algorithm}'."
            )

        return os.path.join(*parts)

    def _build_run_name(self, cfg: RlhfCfg, timestamp: str) -> str:
        components = [cfg.rlhf_algorithm]
        if cfg.lazy:
            components.append("lazy")
        if cfg.opt_design:
            components.append("opt_design")
        if cfg.pure_exploration:
            components.append("pure_exploration")
        if cfg.beta1 is not None:
            components.append(f"beta1_{self._format_value(cfg.beta1)}")
        if cfg.beta2 is not None:
            components.append(f"beta2_{self._format_value(cfg.beta2)}")
        components.append(f"seed_{cfg.base_seed}")
        components.append(timestamp)
        return "-".join(components)
