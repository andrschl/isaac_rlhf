import multiprocessing
import os
import traceback
import torch
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Literal
from einops import einsum
import math

from isaac_rlhf.config import RlhfCfg
from isaac_rlhf.utils.rlhf_utils import MuteOutput, get_freest_gpu

# Helpers


def set_seed(cfg):
    import torch
    import random
    import numpy as np

    torch.manual_seed(cfg.base_seed)
    np.random.seed(cfg.base_seed)
    random.seed(cfg.base_seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.base_seed)


def create_environment(cfg: RlhfCfg, init: bool = False):
    from isaaclab.app import AppLauncher

    if cfg.device.startswith("cuda"):
        cfg.device = f"cuda:{get_freest_gpu()}"
    launcher = AppLauncher(headless=True, device=cfg.device)
    simulation_app = launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    num_envs = cfg.num_envs if not init else 1  # use 1 env for initialization
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        cfg.task, num_envs=num_envs, device=cfg.device
    )
    set_seed(cfg)
    env_cfg.seed = cfg.base_seed
    env = gym.make(cfg.task, cfg=env_cfg)
    return env, simulation_app


# Worker class
class WorkerTask:
    def __init__(
        self, idx, rewards_queue, results_queue, termination_event, cfg: RlhfCfg
    ):
        self.idx = idx
        self.rewards_queue = rewards_queue
        self.results_queue = results_queue
        self.termination_event = termination_event
        self.cfg = cfg
        self.device = cfg.device

        print(f"[DEBUG] Worker {self.idx} start creating environment.")
        self.env, self.simulation_app = create_environment(cfg)
        print(f"[DEBUG] Worker {self.idx} created environment: {self.env}")

    def prepare_rlhf_environment(self, reward_param: torch.Tensor):
        """Prepare environment for RLHF using reward_param."""
        from isaaclab.envs import ManagerBasedRLEnv

        # Adjust reward parameters in the environment:
        unwrapped = self.env.unwrapped
        if isinstance(unwrapped, ManagerBasedRLEnv):
            idx = 0
            for name, term_cfg in zip(
                unwrapped.reward_manager._term_names,
                unwrapped.reward_manager._term_cfgs,
            ):
                if term_cfg.weight != 0.0 and name not in self.cfg.ignored_reward_terms:
                    term_cfg.weight = float(reward_param[idx].item())
                    idx += 1
        else:
            raise Exception("Environment must be of type ManagerBasedRLEnv.")

    def rl_training(self):
        """Run training for the environment. Return features and a log directory."""
        from isaaclab_tasks.utils.parse_cfg import (
            load_cfg_from_registry,
            get_checkpoint_path,
        )

        if self.cfg.rl_library == "rsl_rl":
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
                self.cfg.task, "rsl_rl_cfg_entry_point"
            )
            agent_cfg.device = self.device
            agent_cfg.seed = self.cfg.base_seed
            agent_cfg.max_iterations = self.cfg.num_rl_iterations

            log_root_path = os.path.join(
                "logs", "rl_runs", "rsl_rl_rlhf", agent_cfg.experiment_name
            )
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs: {time-stamp}_{run_name}
            run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{self.idx}"
            if agent_cfg.run_name:
                run_dir += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, run_dir)

            env = RslRlVecEnvWrapper(self.env)
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
            )

            # Load from previous checkpoint if available and not the first iteration
            prev_checkpoint_path = getattr(self, "prev_checkpoint_path", None)
            print(
                f"[DEBUG] Resume is {self.cfg.resume}, prev_checkpoint_path: {prev_checkpoint_path}, prev_checkpoint_path exists: {os.path.exists(prev_checkpoint_path) if prev_checkpoint_path else 'N/A'}"
            )
            # Check if the previous checkpoint exists
            if (
                self.cfg.resume
                and prev_checkpoint_path
                and os.path.exists(prev_checkpoint_path)
            ):
                print(f"[DEBUG] Loading previous checkpoint.")
                print(
                    "[DEBUG] Previous checkpoint path: ",
                    f"{prev_checkpoint_path}, exists: {os.path.exists(prev_checkpoint_path)}",
                    f"[INFO] Loading policy from previous checkpoint: {prev_checkpoint_path}",
                )
                runner.load(prev_checkpoint_path)

            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )

            # Save the checkpoint for the next iteration
            self.prev_checkpoint_path = get_checkpoint_path(
                log_root_path, run_dir, "model_*"
            )

            traj_features, mean_episode_reward = self.record_features(runner, env)
            return traj_features, mean_episode_reward, log_dir

        else:
            raise Exception(f"framework {self.cfg.rl_library} is not supported yet.")

    def record_features(self, runner, env):
        if env.num_envs < self.cfg.num_trajectories_per_run:
            raise ValueError(
                f"Number of trajectories ({self.cfg.num_trajectories_per_run}) is greater than number of environments ({env.num_envs})."
            )

        if self.cfg.rl_library == "rsl_rl":
            # reset the env first

            # from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            # env = RslRlVecEnvWrapper(env.unwrapped.copy())
            with torch.inference_mode():
                env.reset()
            obs, _ = env.get_observations()

            runner.eval_mode()
            gamma = runner.alg.gamma

            print(self.cfg.to_dict())

            traj_features = torch.zeros(
                env.num_envs,
                self.cfg.num_features,
                device=self.device,
            )
            terminated = torch.zeros(env.num_envs, device=self.device)
            episode_rewards = torch.zeros(env.num_envs, device=self.device)
            for t in range(self.cfg.trajectory_length):
                with torch.inference_mode():
                    actions = runner.alg.policy.act(obs).detach()
                obs, rewards, dones, _ = env.step(actions)
                obs = runner.obs_normalizer(obs)
                terminated = terminated.int() | dones.int()
                step_features = einsum(
                    self.get_feature_values(),
                    (1 - terminated),
                    "i j, i -> i j",
                )
                traj_features += gamma**t * step_features * self.cfg.dt
                episode_rewards += gamma**t * rewards

            return traj_features, episode_rewards.mean().item()

        else:
            raise Exception(f"framework {self.cfg.rl_library} is not supported yet.")

    def get_feature_values(self):
        reward_features = []
        for name, term_cfg in zip(
            self.env.unwrapped.reward_manager._term_names,
            self.env.unwrapped.reward_manager._term_cfgs,
        ):
            if term_cfg.weight != 0.0 and name not in self.cfg.ignored_reward_terms:
                reward_features.append(
                    term_cfg.func(self.env.unwrapped, **term_cfg.params)
                )

        return torch.stack(reward_features, dim=1).to(self.device)

    def run(self):
        """Main loop for the worker task."""

        print(f"[INFO]: Worker {self.idx} started.")
        while not self.termination_event.is_set():
            reward_param = self.rewards_queue.get()
            if reward_param == "Stop":
                break

            try:
                self.prepare_rlhf_environment(reward_param)
                # Only display output for worker 0; others can be muted
                context = nullcontext() if self.idx == 0 else MuteOutput()
                with context:
                    features, mean_episode_reward, log_dir = self.rl_training()
                result = {
                    "success": True,
                    "log_dir": log_dir,
                    "features": features.detach().cpu().clone(),
                    "mean_episode_reward": mean_episode_reward,
                }
            except Exception as e:
                result = {"success": False, "exception": str(e)}
                print(traceback.format_exc())

            self.results_queue.put((self.idx, result))

        # Cleanup when finished.
        print(f"[INFO]: Worker {self.idx} terminated.")
        self.env.close()
        self.simulation_app.close()


# Define main worker function
def worker_main(idx, rewards_queue, results_queue, termination_event, worker_cfg):
    task = WorkerTask(idx, rewards_queue, results_queue, termination_event, worker_cfg)
    task.run()


# Task manager
class RlhfTaskManager:
    def __init__(
        self,
        cfg: RlhfCfg,
    ):
        """
        Initialize the RLHF Task Manager.
        """
        # unpack the configuration
        self.cfg = cfg
        self.device = cfg.device

        # Initialize multiprocessing data structures
        self.shared_data = (
            multiprocessing.Manager().dict()
        )  # if you need to share constants
        self.rewards_queues = [
            multiprocessing.Queue() for _ in range(cfg.num_processes)
        ]
        self.results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()
        self.processes = {}

        # Initialize constants
        set_seed(cfg)
        self.init_constants()
        self.init_feature_storage()
        self.init_reward_model()

        print(
            f"[INFO] Running Rlhf with the following configuration: {self.cfg.to_dict()}"
        )

        # Create worker processes using the top-level worker function
        for idx in range(self.cfg.num_processes):
            worker_cfg = self.cfg.replace(base_seed=self.cfg.base_seed + idx + 1)
            p = multiprocessing.Process(
                target=worker_main,
                args=(
                    idx,
                    self.rewards_queues[idx],
                    self.results_queue,
                    self.termination_event,
                    worker_cfg,
                ),
            )
            self.processes[idx] = p
            p.start()

        print(f"[INFO] Created {self.cfg.num_processes} worker processes for Rlhf.")

    # Helpers for initialization
    def init_constants(self):
        p = multiprocessing.Process(target=self.init_process)
        p.start()
        p.join()
        self.init_from_shared_data()

    def init_process(self):
        env, simulation_app = create_environment(self.cfg, init=True)
        self.shared_data["gt_params"] = self.fetch_gt_params(env)
        self.shared_data["dt"] = env.unwrapped.step_dt

        env.close()
        time.sleep(15)  # Give some time for the process to close properly
        simulation_app.close()

    def init_from_shared_data(self):
        print(
            f"[INFO] Initializing RLHF with {len(self.shared_data['gt_params'])} reward parameters."
        )
        self.cfg.gt_params = self.shared_data["gt_params"]
        self.cfg.num_features = len(self.cfg.gt_params)
        self.cfg.dt = self.shared_data["dt"]
        self.reward_params = [
            torch.randn(self.cfg.num_features, device="cpu")
            for _ in range(self.cfg.num_rl_runs)
        ]

    def fetch_gt_params(self, env):
        gt_params = {}
        term_cfgs = env.unwrapped.reward_manager._term_cfgs
        term_names = env.unwrapped.reward_manager._term_names
        for term_cfg, name in zip(term_cfgs, term_names):
            if term_cfg.weight != 0.0 and name not in self.cfg.ignored_reward_terms:
                gt_params[name] = term_cfg.weight
        return gt_params

    def init_feature_storage(self):
        from isaac_rlhf.storage.feature_storage_rlhf import FeatureStorageRlhf

        self.feature_storage = FeatureStorageRlhf(cfg=self.cfg)

    def init_reward_model(self):
        from isaac_rlhf.modules import LinearReward

        self.reward_model = LinearReward(
            num_features=self.cfg.num_features,
            lambda_=1.0,
            gt_params=self.gt_params_as_tensor(),
        )

    # Helpers for logging
    def get_gt_reward(self, results):
        """Compute approx. ground truth reward."""
        traj_features = sum([result["features"] for result in results]) / len(results)
        traj_features = traj_features.to(self.device)
        gt_reward = self.reward_model.get_gt_reward(traj_features).mean().item()
        return gt_reward

    def gt_params_as_tensor(self):
        return torch.Tensor(list(self.cfg.gt_params.values()))

    def get_pred_reward(self, results):
        """Compute approx. predicted reward."""
        print(f"[DEBUG] Predicted reward params: {results[0]['features'].shape}")
        traj_features = sum([result["features"] for result in results]) / len(results)
        pred_reward = self.reward_model.get_reward(traj_features).mean().item()
        return pred_reward

    def get_reward_error(self):
        """Compute the difference between ground truth and predicted reward."""
        return torch.norm(
            self.gt_params_as_tensor() - self.reward_model.get_reward_params(), p=2
        ).item()

    def get_V_inv_eigenvalues(self):
        """Compute the eigenvalues of the covariance matrix."""
        eigvals, _ = torch.linalg.eig(self.feature_storage.V_inv)
        return eigvals.cpu().real

    # Core methods
    def distribute_rewards(self) -> list[dict]:
        """Distribute reward parameters to workers and collect results."""
        all_results = []
        total = len(self.reward_params)
        for i in range(0, total, self.cfg.num_processes):
            batch = self.reward_params[i : i + self.cfg.num_processes]
            for idx in range(len(batch)):
                self.rewards_queues[idx].put(batch[idx])
            batch_results = [None] * len(batch)
            for _ in range(len(batch)):
                idx, result = self.results_queue.get()
                batch_results[idx] = result
            all_results.extend(batch_results)
        self.feature_storage.fill_storage(all_results)
        print(f"[INFO] Collected results from {len(all_results)} policies.")
        return all_results

    def get_preferences(self):
        """Get synthetic preferences."""
        return self.feature_storage.get_preferences(self.reward_model)

    def mle_update(self, iter=None):
        from isaac_rlhf.algorithms import train_reward_model

        return train_reward_model(
            self.reward_model,
            self.feature_storage,
            lr=self.cfg.mle_lr,
            l2_reg=self.cfg.mle_l2_reg,
            epochs=self.cfg.mle_epochs,
            batch_size=self.cfg.mle_batch_size,
            iter=iter,
        )

    def sample_reward_params(self, iter=0) -> list[torch.Tensor]:
        """Return the reward parameters as CPU tensors."""

        # Return updated reward params
        thetahat = self.reward_model.get_reward_params()
        if self.feature_storage.update_params:
            if self.cfg.rlhf_algorithm == "vanilla":
                self.reward_params = [thetahat] * self.cfg.num_rl_runs
                return self.reward_params
            if self.cfg.rlhf_algorithm in ["ts_double", "ts_last"]:
                eps = 1e-6
                beta = self.cfg.beta1 + self.cfg.beta2 * min(math.log(iter + 1), 1)
                cov = beta**2 * self.feature_storage.V_inv
                cov = cov + eps * torch.eye(
                    cov.shape[0]
                )  # Add small noise to covariance for numerical stability
                print(
                    f"[DEBUG] Is symmetric: {torch.allclose(cov, cov.T)}, is pos def: {torch.all(torch.linalg.eigvals(cov).real > 1e-8)}"
                )
                distribution = torch.distributions.MultivariateNormal(
                    thetahat, covariance_matrix=cov
                )
                self.reward_params = [
                    distribution.sample().cpu() for _ in range(self.cfg.num_rl_runs)
                ]
                return self.reward_params
            if self.cfg.rlhf_algorithm == "rl":
                self.reward_params = [self.gt_params_as_tensor()] * self.cfg.num_rl_runs
                return self.reward_params
            else:
                raise Exception(
                    f"RLHF algorithm {self.cfg.rlhf_algorithm} is not supported yet."
                )

    def query_now(self):
        """Check if the lazy update condition is met."""
        if self.cfg.lazy:
            det_curr_V = torch.det(self.feature_storage.curr_V)
            det_V = torch.det(self.feature_storage.V)
            return det_curr_V >= self.cfg.lazy_constant * det_V
        else:
            return True

    # Save results
    def save_results(self, log_dir: str):
        """Save results to the specified directory."""
        os.makedirs(log_dir, exist_ok=True)
        self.feature_storage.save(log_dir)
        self.reward_model.save(log_dir)

    # Close the task manager
    def close(self):
        self.termination_event.set()
        # Signal workers to stop:
        for q in self.rewards_queues:
            q.put("Stop")
        for p in self.processes.values():
            p.join()
