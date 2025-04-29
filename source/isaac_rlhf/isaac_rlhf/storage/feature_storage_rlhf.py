"""Under development. Do not use yet."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, List
import warnings

if TYPE_CHECKING:
    from isaac_rlhf.modules.linear_reward import LinearReward


# This storage is for linear reward classes. It stores the reward terms of for a manager-based environment.


class FeatureStorageRlhf:
    def __init__(
        self,
        num_features,
        max_num_samples_per_ep=1000,
        max_num_samples_tot=100000,
        dt=1.0,
        device="cpu",
    ):
        self.device = device
        self.num_features = num_features
        self.max_num_samples_per_ep = max_num_samples_per_ep
        self.max_num_samples_tot = max_num_samples_tot
        self.dt = dt
        self.step = 0
        self.policy_step = 0

        # pre‐allocate circular buffers
        self.traj_features = torch.zeros(
            (max_num_samples_per_ep, 2, num_features), device=device
        )
        self.policy_ids = torch.zeros(
            (max_num_samples_per_ep, 2), dtype=torch.long, device=device
        )
        self.mask = torch.zeros(
            (self.max_num_samples_per_ep,), dtype=torch.bool, device=device
        )

        # data
        self.X_hist = torch.zeros(
            (self.max_num_samples_tot, self.num_features), device=device
        )
        self.y_hist = torch.zeros(
            (self.max_num_samples_tot,), dtype=torch.long, device=device
        )
        self.hist_mask = torch.zeros(
            (self.max_num_samples_tot,), dtype=torch.bool, device=device
        )
        self.hist_step = 0

    def fill_storage(self, results: List[Dict[str, torch.Tensor]]):
        # expect results: List[Tensor] of length 2*k, each [num_trajs, num_features]
        for idx in range(0, len(results), 2):
            f0 = results[idx]["features"].to(self.device)
            f1 = results[idx + 1]["features"].to(self.device)
            for j in range(f0.size(0)):
                buf_idx = self.step % self.max_num_samples_per_ep
                self.traj_features[buf_idx, 0] = f0[j]
                self.traj_features[buf_idx, 1] = f1[j]
                self.policy_ids[buf_idx, 0] = 2 * self.policy_step
                self.policy_ids[buf_idx, 1] = 2 * self.policy_step + 1
                self.mask[buf_idx] = True
                self.step += 1
            self.policy_step += 1

        if self.step > self.max_num_samples_per_ep:
            warnings.warn(
                f"Write pointer step={self.step} exceeded max_num_samples_per_ep={self.max_num_samples_per_ep}. "
                "Old entries will be overwritten."
            )

    def get_traj_features(self):
        # returns traj_features
        valid_ids = self.mask.nonzero(as_tuple=True)[0]
        return self.traj_features[valid_ids]

    def get_design_points(self):
        # returns design_points
        valid_ids = self.mask.nonzero(as_tuple=True)[0]
        return self.traj_features[valid_ids, 0] - self.traj_features[valid_ids, 1]

    def get_Xy(self):
        ids = self.hist_mask.nonzero(as_tuple=True)[0]
        return self.X_hist[ids], self.y_hist[ids]

    def get_preferences(self, reward_model: "LinearReward"):
        # collect only the valid design‐points
        X = self.get_design_points()  # [N, num_features]
        utilities = reward_model.get_gt_reward(X)  # [N]
        probs = torch.sigmoid(utilities)  # P(prefer first over second)
        y = torch.bernoulli(probs).long()  # [N]

        # store and return new data points
        for i in range(X.size(0)):
            hidx = self.hist_step % self.max_num_samples_tot
            self.X_hist[hidx] = X[i]
            self.y_hist[hidx] = y[i]
            self.hist_mask[hidx] = True
            self.hist_step += 1
        return X, y

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
        num_features = X_np.shape[1]
        columns = [f"feature_{i}" for i in range(num_features)] + ["y"]

        # Concatenate X and y (y is reshaped to a column vector)
        data = np.hstack([X_np, y_np.reshape(-1, 1)])

        # Create a DataFrame and save as CSV
        df = pd.DataFrame(data, columns=columns)
        csv_path = os.path.join(logdir, "preference_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved preference data to {csv_path}")

    def clear(self):
        self.traj_features = torch.zeros(
            (self.max_num_samples_per_ep, self.num_features), device=self.device
        )
        self.policy_ids = torch.zeros(
            (self.max_num_samples_per_ep,), dtype=torch.long, device=self.device
        )
        self.mask = torch.zeros(
            (self.max_num_samples_per_ep,), dtype=torch.bool, device=self.device
        )
        self.step = 0
        self.policy_step = 0
