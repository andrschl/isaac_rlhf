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
        # nn.init.zeros_(self.reward.weight)  # Initialize weights to zero
        # self.V = lambda_ * torch.eye(num_features).to("cpu")
        # self.V_inv = (1 / lambda_) * torch.eye(num_features).to("cpu")
        # self.curr_V = self.V.clone().to("cpu")
        # self.curr_V_inv = self.V_inv.clone().to("cpu")
        self.gt_params = gt_params.to("cpu") if gt_params is not None else None
        
        # Ensure the linear layer weights have the same dtype as gt_params
        if gt_params is not None:
            self.reward.weight.data = self.reward.weight.data.to(gt_params.dtype)
        
        self.device = device
        self.step = 0

    def get_reward_params(self):
        return self.reward.weight.data.view(-1).clone().cpu()

    def get_reward(self, features):
        features = features.to(self.device)
        # Ensure features have the same dtype as the model weights
        features = features.to(self.reward.weight.dtype)
        return self.reward(features).squeeze(1)

    def get_gt_reward(self, features):
        features = features.to(self.device)
        if self.gt_params is None:
            raise ValueError("Ground truth parameters are not set.")
        gt_params = self.gt_params.to(self.device).to(features.dtype)
        return torch.tensordot(features, gt_params, dims=([-1], [0])).to("cpu")

    # def update_curr_V(self, feature_storage: FeatureStorageRlhf):
    #     # design_points: [num_samples, num_features]
    #     design_points = feature_storage.get_new_design_points()
    #     for feature in design_points:
    #         self.curr_V += torch.outer(feature, feature)
    #     self.curr_V_inv = torch.linalg.inv(self.curr_V)

    # def update_V(self, feature_storage: FeatureStorageRlhf):
    #     # design_points: [num_samples, num_features]
    #     self.V = self.curr_V.clone()
    #     self.V = (self.V + self.V.T) / 2  # symmetrize for numerical stability
    #     self.V_inv = self.curr_V_inv.clone()
    #     self.V_inv = (self.V_inv + self.V_inv) / 2  # symmetrize for numerical stability

    def save(self, logdir: str):
        torch.save(self.state_dict(), logdir + "/reward_model.pth")
