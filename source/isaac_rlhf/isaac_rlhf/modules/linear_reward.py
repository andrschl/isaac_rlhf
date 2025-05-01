from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from isaac_rlhf.storage import FeatureStorageRlhf


class LinearReward(nn.Module):
    def __init__(
        self,
        num_features,
        lambda_=1.0,
        gt_params=None,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "RewardModelc.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.reward = nn.Linear(num_features, 1, bias=False, device=device)
        self.V = lambda_ * torch.eye(num_features).to("cpu")
        self.V_inv = (1 / lambda_) * torch.eye(num_features).to("cpu")
        self.curr_V = self.V.clone().to("cpu")
        self.curr_V_inv = self.V_inv.clone().to("cpu")
        self.gt_params = gt_params.to("cpu") if gt_params is not None else None
        self.device = device
        self.step = 0
        

    def get_reward_params(self):
            return self.reward.weight.data.view(-1).clone().cpu()
    
    def get_reward(self, features):
        return self.reward(features).squeeze(1)

    def get_gt_reward(self, features):
        if self.gt_params is None:
            raise ValueError("Ground truth parameters are not set.")
        return torch.tensordot(features, self.gt_params, dims=([-1], [0])).to("cpu")

    def update_curr_V(self, feature_storage: FeatureStorageRlhf):
        # design_points: [num_samples, num_features]
        design_points = feature_storage.get_design_points()
        for feature in design_points:
            self.curr_V += torch.outer(feature, feature)
        self.curr_V_inv = torch.linalg.inv(self.curr_V)

    def update_V(self, feature_storage: FeatureStorageRlhf):
        # design_points: [num_samples, num_features]
        self.V = self.curr_V.clone()
        self.V_inv = self.curr_V_inv.clone()

    def update_reward_and_confidence_set(
        self,
        feature_storage: FeatureStorageRlhf,
        lr: float = 1e-3,
        l2_reg: float = 1e-6,
        epochs: int = 500,
        batch_size: int = 64,
        num_workers: int = 0,
        device: str = "cpu",
    ):
        from isaac_rlhf.algorithms import train_reward_model

        self.update_curr_V(feature_storage)
        self.update_V(feature_storage)
        self.step += 1
        return train_reward_model(
            self,
            feature_storage,
            lr=lr,
            l2_reg=l2_reg,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

    def save(self, logdir: str):
        torch.save(self.state_dict(), logdir + "/reward_model.pth")
