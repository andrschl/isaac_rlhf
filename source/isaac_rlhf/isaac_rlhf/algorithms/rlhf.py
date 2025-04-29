import multiprocessing
import os
import traceback
import torch
from contextlib import nullcontext
from datetime import datetime
from typing import Literal
from einops import einsum

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
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(cfg.task, num_envs=num_envs, device=cfg.device)
    env_cfg.seed = cfg.base_seed
    env = gym.make(cfg.task, cfg=env_cfg)
    return env, simulation_app

# Worker class
class WorkerTask:
    def __init__(
            self, 
            idx, 
            rewards_queue, 
            results_queue, 
            termination_event, 
            cfg: RlhfCfg
            ):
                    
        self.idx = idx
        self.rewards_queue = rewards_queue
        self.results_queue = results_queue
        self.termination_event = termination_event
        self.device = cfg.device

        set_seed(cfg)
        self.env, self.simulation_app = create_environment(cfg)
        
    def prepare_rlhf_environment(self, reward_param: torch.Tensor):
        """Prepare environment for RLHF using reward_param."""
        from isaaclab.envs import ManagerBasedRLEnv
        # Adjust reward parameters in the environment:
        unwrapped = self.env.unwrapped
        if isinstance(unwrapped, ManagerBasedRLEnv):
            for idx, term_cfg in enumerate(unwrapped.reward_manager._term_cfgs):
                if term_cfg.weight != 0.0:
                    term_cfg.weight = float(reward_param[idx].item())
        else:
            raise Exception("Environment must be of type ManagerBasedRLEnv.")
    
    def rl_training(self):
        """Run training for the environment. Return features and a log directory."""
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        if self.rl_cfg["rl_library"] == "rsl_rl":
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(self.cfg.task, "rsl_rl_cfg_entry_point")
            agent_cfg.device = self.device
            agent_cfg.max_iterations = self.cfg.num_rl_iterations

            log_root_path = os.path.join("logs", "rl_runs", "rsl_rl_rlhf", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs: {time-stamp}_{run_name}
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{self.idx}"
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, log_dir)

            env = RslRlVecEnvWrapper(self.env)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

            return self.record_features(runner, env), log_dir

        else:
            raise Exception(f"framework {self.rl_cfg['rl_library']} is not supported yet.")

        
    def record_features(self, runner, env, max_traj_len=128):

        if env.num_envs < self.cfg.num_trajectories_per_run:
            raise ValueError(
                f"Number of trajectories ({self.cfg.num_trajectories_per_run}) is greater than number of environments ({env.num_envs})."
            )
        
        if self.rl_cfg["rl_library"] == "rsl_rl":
            # reset the env first

            # from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            # env = RslRlVecEnvWrapper(env.unwrapped.copy())
            with torch.inference_mode():
                env.reset()
            obs, _ = env.get_observations()

            # policy = runner.get_inference_policy(device=self.device)
            runner.eval_mode()
            gamma  = runner.alg.gamma

            traj_features = torch.zeros(self.rlhf_cfg["num_trajectories"], self.env_cfg["num_features"], device=self.device)
            terminated = torch.zeros(self.rlhf_cfg["num_trajectories"], device=self.device)
            for t in range(max_traj_len):
                with torch.inference_mode():
                    actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                obs = runner.obs_normalizer(obs)
                terminated = terminated.int() | dones[:self.rlhf_cfg["num_trajectories"]].int()
                step_features = einsum(self.get_reward_features()[:self.rlhf_cfg["num_trajectories"]], (1-terminated), 'i j, i -> i j')
                traj_features += gamma ** t * step_features * self.env_cfg["dt"]

            return traj_features
        
        else:
            raise Exception(f"framework {self.rl_cfg['rl_library']} is not supported yet.")
    
    def get_reward_features(self):

        reward_features = []
        for term_cfg in self.env.unwrapped.reward_manager._term_cfgs:
            if term_cfg.weight != 0.0:
                reward_features.append(term_cfg.func(self.env.unwrapped, **term_cfg.params))

        return torch.stack(reward_features, dim=1).to(self.device)

    def run(self):
        """Main loop for the worker task."""

        self.set_seed()
        while not self.termination_event.is_set():
            reward_param = self.rewards_queue.get()
            if reward_param == "Stop":
                break

            try:
                self.prepare_rlhf_environment(reward_param)
                # Only display output for worker 0; others can be muted
                context = nullcontext() if self.idx == 0 else MuteOutput()
                with context:
                    features, log_dir = self.rl_training()
                result = {"success": True, "log_dir": log_dir, "features": features.detach().cpu().clone()}
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
        self.shared_data = multiprocessing.Manager().dict()  # if you need to share constants
        self.rewards_queues = [multiprocessing.Queue() for _ in range(cfg.num_processes)]
        self.results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()
        self.processes = {}

        # Initialize constants
        self.set_seed()
        self.init_constants()
        self.init_feature_storage()
        self.init_reward_model()

        # Create worker processes using the top-level worker function
        for idx in range(self.cfg.num_processes):
            worker_cfg = cfg.replace(base_seed=self.cfg.base_seed + idx)
            p = multiprocessing.Process(
                target=worker_main,
                args=(idx, self.rewards_queues[idx], self.results_queue, self.termination_event, worker_cfg)
            )
            self.processes[idx] = p
            p.start()

    # Helpers for initialization
    def init_constants(self):

        p = multiprocessing.Process(target=self.init_process)
        p.start()
        p.join()
        self.init_from_shared_data()

    def init_process(self):
            env, simulation_app = self.create_environment()
            self.shared_data["num_features"] = self.get_num_features(env) 
            self.shared_data["gt_params"] = self.get_reward_weights(env)
            self.shared_data["dt"] = env.unwrapped.step_dt
            env.close()
            simulation_app.close()

    def init_from_shared_data(self):
        if "num_features" in self.shared_data:
            print(f"[INFO] Using {self.shared_data['num_features']} features.")
            self.env_cfg["num_features"] = self.shared_data["num_features"]
        if "gt_params" in self.shared_data:
            print(f"[INFO] Using {len(self.shared_data['gt_params'])} reward parameters.")
            self.env_cfg["gt_params"] = self.shared_data["gt_params"]
        if "dt" in self.shared_data:
            print(f"[INFO] Using dt = {self.shared_data['dt']}.")
            self.env_cfg["dt"] = self.shared_data["dt"]
    
    def create_environment(self):
        from isaaclab.app import AppLauncher
        if self.device.startswith("cuda"):
            self.device = f"cuda:{get_freest_gpu()}"
        launcher = AppLauncher(headless=True, device=self.device)
        simulation_app = launcher.app

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(self.env_cfg["task"])
        env_cfg.num_envs = 1
        env = gym.make(self.env_cfg["task"], cfg=env_cfg)
        return env, simulation_app
    
    def get_num_features(self, env):
        return len([0 for term_cfg in env.unwrapped.reward_manager._term_cfgs if term_cfg.weight != 0.0])
    
    def get_reward_weights(self, env):
        weights = []
        for term_cfg in env.unwrapped.reward_manager._term_cfgs:
            if term_cfg.weight != 0.0:
                weights.append(term_cfg.weight)
        return weights
    
    def init_feature_storage(self):
        from isaac_rlhf.storage.feature_storage_rlhf import FeatureStorageRlhf
        self.feature_storage = FeatureStorageRlhf(
            num_features=self.env_cfg["num_features"],
            dt=self.env_cfg["dt"],
            device="cpu"    # use CPU for feature storage
        )

    def init_reward_model(self):
        from isaac_rlhf.modules import LinearReward
        self.reward_model = LinearReward(
            num_features=self.env_cfg["num_features"],
            lambda_=1.0,
            gt_params=torch.Tensor(self.env_cfg["gt_params"]),
            device=self.device
        )

    # Helpers for logging
    def get_gt_reward(self):
        """Compute approx. ground truth reward."""
        traj_features = self.feature_storage.get_traj_features()
        gt_reward = self.reward_model.get_gt_reward(traj_features).mean().item()
        return gt_reward
    
    def get_pred_reward(self):
        """Compute approx. predicted reward."""
        traj_features = self.feature_storage.get_traj_features()
        pred_reward = self.reward_model.get_reward(traj_features).mean().item()
        return pred_reward
    
    def get_reward_error(self):
        """Compute the difference between ground truth and predicted reward."""
        return torch.norm(self.reward_model.gt_params - self.reward_model.get_reward_params(), p=2).item()
    
    def get_V_inv_eigenvalues(self):
        """Compute the eigenvalues of the covariance matrix."""
        eigvals, _ = torch.linalg.eig(self.reward_model.V_inv)
        return eigvals.cpu().real
    
    # Update and distribute rewards
    def get_reward_params(self, device: str = "cpu") -> list[torch.Tensor]:
        """Return the reward parameters as CPU tensors."""

        # Run MLE
        thetahat = self.reward_model.update_reward_and_confidence_set(
            self.feature_storage, 
            lr=self.rlhf_cfg["lr"],
            l2_reg=self.rlhf_cfg["l2_reg"],
            epochs=self.rlhf_cfg["epochs"],
            batch_size=self.rlhf_cfg["batch_size"],
            device=device
        )

        # Return updated reward params
        if self.rlhf_cfg["rlhf_algorithm"] == "vanilla":
            return [thetahat] * self.num_rl_runs
        if self.rlhf_cfg["rlhf_algorithm"] == "ts":
            covariance = self.reward_model.V_inv
            distribution = torch.distributions.MultivariateNormal(thetahat, covariance_matrix=covariance)
            return [distribution.sample().cpu() for _ in range(self.num_rl_runs)]
        else:
            raise Exception(f"RLHF algorithm {self.rlhf_cfg['rlhf_algorithm']} is not supported yet.")
    

        # print(f"[INFO] Using {self.env_cfg['num_features']} features and {len(self.env_cfg['gt_params'])} reward parameters.")
        # return [torch.tensor(self.env_cfg["gt_params"], device="cpu").detach().clone()] * self.cfg.num_processes

    def distribute_rewards(self, reward_params: list[torch.Tensor]) -> list[dict]:
        """Distribute reward parameters to workers and collect results."""
        all_results = []
        total = len(reward_params)
        for i in range(0, total, self.cfg.num_processes):
            batch = reward_params[i: i + self.cfg.num_processes]
            for idx in range(len(batch)):
                self.rewards_queues[idx].put(batch[idx])
            batch_results = [None] * len(batch)
            for _ in range(len(batch)):
                idx, result = self.results_queue.get()
                batch_results[idx] = result
                print("[INFO] Design points: ", result["features"])
            all_results.extend(batch_results)
        self.feature_storage.fill_storage(all_results)
        print(f"[INFO] Collected results from {len(all_results)} policies.")
        return all_results
    
    def get_preferences(self):
        """Get synthetic preferences."""
        return self.feature_storage.get_preferences(self.reward_model)
    

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