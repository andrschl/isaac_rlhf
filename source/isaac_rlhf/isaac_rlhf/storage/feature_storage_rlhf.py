"""Under development. Do not use yet."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, List
import warnings
from einops import einsum

if TYPE_CHECKING:
    from isaac_rlhf.modules.linear_reward import LinearReward
    from isaac_rlhf.config import RlhfCfg


# This storage is for linear reward classes. It stores the reward terms of for a manager-based environment.


class FeatureStorageRlhf:
    def __init__(
        self,
        cfg: RlhfCfg,
        max_ep_buffers_size=int(1e6),
        max_dataset_size=int(1e8),
        device="cpu",
    ):
        self.device = device
        self.cfg = cfg
        self.num_features = cfg.num_features
        self.reward_param_names = list(cfg.gt_params.keys())

        self.max_ep_buffers_size = max_ep_buffers_size
        self.max_dataset_size = max_dataset_size
        self.update_params = True
        self.lazy_update = False
        self.step = 0
        self.policy_step = 0

        # circular rlhf episode buffers
        self.traj_features = torch.zeros(
            (max_ep_buffers_size, 2, self.num_features), device=device
        )
        if cfg.rlhf_algorithm == "ts_last":
            # ts_last: compare to the last policy
            self.traj_features_prev = torch.zeros(
                (max_ep_buffers_size, self.num_features), device=device
            )
        # self.mean_episode_reward = 0.0
        # self.mean_gt_episode_reward = 0.0
        self.policy_ids = -torch.ones(
            (max_ep_buffers_size, 2), dtype=torch.long, device=device
        )
        self.mask = torch.zeros(
            (self.max_ep_buffers_size,), dtype=torch.bool, device=device
        )

        # preference dataset
        self.X_hist = torch.zeros(
            (self.max_dataset_size, self.num_features), device=device
        )
        self.y_hist = torch.zeros(
            (self.max_dataset_size,), dtype=torch.long, device=device
        )
        self.hist_mask = torch.zeros(
            (self.max_dataset_size,), dtype=torch.bool, device=device
        )
        self.hist_step = 0

        # design matrices
        self.V = self.cfg.lambda_ * torch.eye(self.cfg.num_features).to("cpu")
        self.V_inv = (1 / self.cfg.lambda_) * torch.eye(self.cfg.num_features).to("cpu")
        self.curr_V = self.V.clone().to("cpu")

    def fill_storage(self, results: List[Dict[str, torch.Tensor]]):
        """Fill the storage with features from the results.
        The results are expected to be a list of dictionaries, where each dictionary contains all
        the features for a given policy in result["features"], a tensor of shape
        [num_trajs_per_run, num_features].
        """

        if self.cfg.rlhf_algorithm in ["vanilla", "rl"]:
            # vanilla: rely only on entropy exploration

            for idx in range(0, len(results)):
                features = results[idx]["features"].to(self.device)
                num_comparisons = self.cfg.num_trajectories_per_run // 2
                for j in range(0, num_comparisons):
                    buf_idx = self.step % self.max_ep_buffers_size
                    self.traj_features[buf_idx, 0] = features[2 * j]
                    self.traj_features[buf_idx, 1] = features[2 * j + 1]
                    self.curr_V += torch.outer(
                        self.traj_features[buf_idx, 0] - self.traj_features[buf_idx, 1],
                        self.traj_features[buf_idx, 0] - self.traj_features[buf_idx, 1],
                    )
                    self.policy_ids[buf_idx, 0] = self.policy_step
                    self.policy_ids[buf_idx, 1] = self.policy_step
                    self.mask[buf_idx] = True
                    self.step += 1
                self.policy_step += 1

        elif self.cfg.rlhf_algorithm == "ts_last":
            # ts_last: compare to the last policy

            for idx in range(0, len(results)):
                features = results[idx]["features"].to(self.device)
                num_comparisons = (
                    self.cfg.num_trajectories_per_run // 2
                )  # use only half of samples at time t and half at time t+1
                for j in range(0, num_comparisons):
                    buf_idx = self.step % self.max_ep_buffers_size
                    self.traj_features[buf_idx, 0] = features[2 * j]
                    self.policy_ids[buf_idx, 0] = self.policy_step
                    if self.step == 0 and self.policy_step == 0:
                        # Compare to trajectories from the same policy
                        self.traj_features[buf_idx, 1] = features[2 * j + 1]
                        self.policy_ids[buf_idx, 1] = self.policy_step
                    else:
                        # Compare to trajectories from the previous policy
                        self.traj_features[buf_idx, 1] = self.traj_features_prev[j]
                        self.policy_ids[buf_idx, 1] = self.policy_step - 1
                    self.traj_features_prev[j] = features[2 * j + 1]
                    self.curr_V += torch.outer(
                        self.traj_features[buf_idx, 0] - self.traj_features[buf_idx, 1],
                        self.traj_features[buf_idx, 0] - self.traj_features[buf_idx, 1],
                    )
                    self.mask[buf_idx] = True
                    self.step += 1
                self.policy_step += 1

        elif self.cfg.rlhf_algorithm == "ts_double":
            # ts_double: compare the two current policies, expect len(results) to be even and >=2.

            for idx in range(0, len(results), 2):
                f0 = results[idx]["features"].to(self.device)
                f1 = results[idx + 1]["features"].to(self.device)
                num_comparisons = self.cfg.num_trajectories_per_run // 2

                for j in range(0, num_comparisons):
                    buf_idx = self.step % self.max_ep_buffers_size
                    self.traj_features[buf_idx, 0] = f0[j]
                    self.traj_features[buf_idx, 1] = f1[j]
                    self.curr_V += torch.outer(
                        self.traj_features[buf_idx, 0] - self.traj_features[buf_idx, 1],
                        self.traj_features[buf_idx, 0] - self.traj_features[buf_idx, 1],
                    )
                    self.policy_ids[buf_idx, 0] = 2 * self.policy_step
                    self.policy_ids[buf_idx, 1] = 2 * self.policy_step + 1
                    self.mask[buf_idx] = True
                    self.step += 1
                self.policy_step += 1
        else:
            raise ValueError(
                f"Unknown rlhf_algorithm: {self.cfg.rlhf_algorithm}. "
                "Expected one of ['vanilla', 'ts_double', 'ts_last']."
            )

        self.curr_V_inv = torch.linalg.inv(self.curr_V)

        if self.step > self.max_ep_buffers_size:
            warnings.warn(
                f"Write pointer step={self.step} exceeded max_ep_buffers_size={self.max_ep_buffers_size}. "
                "Old entries will be overwritten."
            )

    def get_preferences(self, reward_model: "LinearReward"):
        # collect only the valid design‐points  TODO: Add optimal design here.
        X_new = self.get_new_design_points()  # [N, num_features]
        print(
            f"[DEBUG]: Check curr_V {torch.allclose(self.curr_V, self.V + X_new.T @ X_new)}"
        )
        utilities = reward_model.get_gt_reward(X_new)  # [N]
        probs = torch.sigmoid(utilities)  # P(prefer first over second)
        y_new = torch.bernoulli(probs).long()  # [N]
        print(f"[DEBUG]: X_new={X_new}, y_new={y_new}")
        self.update_V()

        # store and return new data points
        for i in range(X_new.size(0)):
            hidx = self.hist_step % self.max_dataset_size
            self.X_hist[hidx] = X_new[i]
            self.y_hist[hidx] = y_new[i]
            self.hist_mask[hidx] = True
            self.hist_step += 1

        self.clear()

        return X_new, y_new

    def update_V(self, X_new=None):
        # design_points: [num_samples, num_features]    TODO: Add functionality for different settings here.
        self.V = self.curr_V.clone()
        self.V = (self.V + self.V.T) / 2  # symmetrize for numerical stability
        self.V_inv = torch.linalg.inv(self.V)
        self.V_inv = (
            self.V_inv + self.V_inv.T
        ) / 2  # symmetrize for numerical stability

    def get_new_design_points(self):
        X = self.traj_features[self.mask, 0] - self.traj_features[self.mask, 1]
        if self.cfg.opt_design:
            num_new_samples = len(X)
            det_curr_V = torch.linalg.det(self.curr_V)
            W = self.V
            W_inv = self.V_inv

            i = 0
            design_points = []
            while torch.linalg.det(W) < det_curr_V and i <= num_new_samples:
                Z = einsum(W_inv, X, "i j, k j -> k i")
                max_idx = torch.argmax(einsum(X, Z, "k i, k i -> k"))
                design_points.append(X[max_idx])
                W = W + torch.outer(X[max_idx], X[max_idx])
                print(
                    f"[DEBUG]: i={i}, max_idx={max_idx}, det(W)={torch.linalg.det(W)}, det_curr_V={det_curr_V}"
                )
                if i % 10 == 0:
                    W_inv = torch.linalg.inv(W)
                else:
                    z = Z[max_idx]
                    x = X[max_idx]
                    W_inv = W_inv - torch.outer(z, z) / (1 + torch.dot(x, z))

                i += 1

            if i > num_new_samples:
                return X
            else:
                self.curr_V = W
                return torch.stack(design_points, dim=0)
        else:
            return X

    def get_traj_features(self, policy_id=None):
        """Returns traj_features corresponding to the given policy_id.
        If policy_id is None, return features from latest policy"""
        policy_id = policy_id if policy_id is not None else self.policy_step - 1
        valid = (self.policy_ids == policy_id) & self.mask.unsqueeze(1)
        return self.traj_features[valid]

    def get_Xy(self):
        return self.X_hist[self.hist_mask], self.y_hist[self.hist_mask]

    def save(self, logdir: str):
        """
        Save the (X, y) history data as a CSV file.
        The CSV file will contain the features (columns feature_0, feature_1, …) and a column for y.
        """
        import os
        import numpy as np
        import pandas as pd

        # Get the X and y history data from get_Xy
        X, y = self.get_Xy()
        if X.numel() == 0 or y.numel() == 0:
            print("No historical (X, y) data to save.")
            return

        # Convert torch tensors to numpy arrays
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        # Build column names based on num_features
        columns = self.reward_param_names + ["y"]

        # Concatenate X and y (y is reshaped to a column vector)
        data = np.hstack([X_np, y_np.reshape(-1, 1)])

        # Create a DataFrame and save as CSV
        df = pd.DataFrame(data, columns=columns)
        csv_path = os.path.join(logdir, "preference_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved preference data to {csv_path}")

    def clear(self):
        self.traj_features[:] = 0.0
        self.policy_ids[:] = -1
        self.mask[:] = False
        self.step = 0


# """Under development. Do not use yet."""

# from __future__ import annotations

# import torch
# from typing import TYPE_CHECKING, Dict, List
# import warnings

# if TYPE_CHECKING:
#     from isaac_rlhf.modules.linear_reward import LinearReward
#     from isaac_rlhf.config import RlhfCfg


# # This storage is for linear reward classes. It stores the reward terms of for a manager-based environment.


# class FeatureStorageRlhf:
#     def __init__(
#         self,
#         cfg: RlhfCfg,
#         max_ep_buffers_size=int(1e6),
#         max_dataset_size=int(1e8),
#         device="cpu",
#     ):
#         self.device = device
#         self.cfg = cfg
#         self.num_features = cfg.num_features

#         self.max_ep_buffers_size = max_ep_buffers_size
#         self.max_dataset_size = max_dataset_size
#         self.update_params = True
#         self.lazy_update = False
#         self.step = 0
#         self.policy_step = 0

#         # pre‐allocate circular buffers
#         self.traj_features = torch.zeros(
#             (max_ep_buffers_size, 2, self.num_features), device=device
#         )
#         self.policy_ids = torch.zeros(
#             (max_ep_buffers_size, 2), dtype=torch.long, device=device
#         )
#         self.mask = torch.zeros(
#             (self.max_ep_buffers_size,), dtype=torch.bool, device=device
#         )

#         # data
#         self.X_hist = torch.zeros(
#             (self.max_dataset_size, self.num_features), device=device
#         )
#         self.y_hist = torch.zeros(
#             (self.max_dataset_size,), dtype=torch.long, device=device
#         )
#         self.hist_mask = torch.zeros(
#             (self.max_dataset_size,), dtype=torch.bool, device=device
#         )
#         self.hist_step = 0

#     def fill_storage(self, results: List[Dict[str, torch.Tensor]]):
#         # expect results: List[Tensor] of length num_rl_runs, each [num_trajs_per_run, num_features]
#         # if self.cfg.rlhf_algorithm == "vanilla":
#         #     # vanilla: rely only on entropy exploration

#         for idx in range(0, len(results), 2):
#             f0 = results[idx]["features"].to(self.device)
#             f1 = results[idx + 1]["features"].to(self.device)
#             for j in range(f0.size(0)):
#                 buf_idx = self.step % self.max_ep_buffers_size
#                 self.traj_features[buf_idx, 0] = f0[j]
#                 self.traj_features[buf_idx, 1] = f1[j]
#                 self.policy_ids[buf_idx, 0] = 2 * self.policy_step
#                 self.policy_ids[buf_idx, 1] = 2 * self.policy_step + 1
#                 self.mask[buf_idx] = True
#                 self.step += 1
#             self.policy_step += 1

#         if self.step > self.max_ep_buffers_size:
#             warnings.warn(
#                 f"Write pointer step={self.step} exceeded max_ep_buffers_size={self.max_ep_buffers_size}. "
#                 "Old entries will be overwritten."
#             )

#     def get_traj_features(self):
#         # returns traj_features
#         valid_ids = self.mask.nonzero(as_tuple=True)[0]
#         return self.traj_features[valid_ids]

#     def get_design_points(self):
#         # returns design_points
#         valid_ids = self.mask.nonzero(as_tuple=True)[0]
#         return self.traj_features[valid_ids, 0] - self.traj_features[valid_ids, 1]

#     def get_Xy(self):
#         ids = self.hist_mask.nonzero(as_tuple=True)[0]
#         return self.X_hist[ids], self.y_hist[ids]

#     def get_preferences(self, reward_model: "LinearReward"):
#         # collect only the valid design‐points
#         X = self.get_design_points()  # [N, num_features]
#         utilities = reward_model.get_gt_reward(X)  # [N]
#         probs = torch.sigmoid(utilities)  # P(prefer first over second)
#         y = torch.bernoulli(probs).long()  # [N]

#         # store and return new data points
#         for i in range(X.size(0)):
#             hidx = self.hist_step % self.max_dataset_size
#             self.X_hist[hidx] = X[i]
#             self.y_hist[hidx] = y[i]
#             self.hist_mask[hidx] = True
#             self.hist_step += 1
#         return X, y

#     def save(self, logdir: str):
#         """
#         Save the (X, y) history data as a CSV file.
#         The CSV file will contain the features (columns feature_0, feature_1, …) and a column for y.
#         """
#         import os
#         import numpy as np
#         import pandas as pd

#         # Get the X and y history data from get_Xy
#         X, y = self.get_Xy()
#         if X.numel() == 0 or y.numel() == 0:
#             print("No historical (X, y) data to save.")
#             return

#         # Convert torch tensors to numpy arrays
#         X_np = X.cpu().numpy()
#         y_np = y.cpu().numpy()

#         # Build column names based on num_features
#         num_features = X_np.shape[1]
#         columns = [f"feature_{i}" for i in range(num_features)] + ["y"]

#         # Concatenate X and y (y is reshaped to a column vector)
#         data = np.hstack([X_np, y_np.reshape(-1, 1)])

#         # Create a DataFrame and save as CSV
#         df = pd.DataFrame(data, columns=columns)
#         csv_path = os.path.join(logdir, "preference_data.csv")
#         df.to_csv(csv_path, index=False)
#         print(f"Saved preference data to {csv_path}")

#     def clear(self):
#         self.traj_features = torch.zeros(
#             (self.max_ep_buffers_size, self.num_features), device=self.device
#         )
#         self.policy_ids = torch.zeros(
#             (self.max_ep_buffers_size,), dtype=torch.long, device=self.device
#         )
#         self.mask = torch.zeros(
#             (self.max_ep_buffers_size,), dtype=torch.bool, device=self.device
#         )
#         self.step = 0
#         self.policy_step = 0
