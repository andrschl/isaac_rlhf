import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import warnings
from isaac_rlhf.modules import LinearReward
from isaac_rlhf.storage import FeatureStorageRlhf

class PreferenceDataset(Dataset):
    def __init__(self, feature_storage: FeatureStorageRlhf):
        X, y = feature_storage.get_Xy()
        if X is None or y is None:
            self.X = torch.empty(0)
            self.y = torch.empty(0, dtype=torch.long)
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_reward_model(
    reward_model: LinearReward,
    feature_storage: FeatureStorageRlhf,
    lr: float = 1e-3,
    l2_reg: float = 1e-2,
    epochs: int = 500,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    reward_model.to(device).train()

    # build DataLoader
    dataset = PreferenceDataset(feature_storage)
    if len(dataset) == 0:
        warnings.warn("No preference data; returning initial reward parameters.")
        return reward_model.reward.weight.data.view(-1).clone()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    optimizer = Adam(reward_model.reward.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = reward_model.get_reward(Xb)
            bce    = F.binary_cross_entropy_with_logits(logits, yb)
            w      = reward_model.reward.weight.view(-1)
            l2     = 0.5 * l2_reg * torch.dot(w, w)
            (bce + l2).backward()
            optimizer.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f"[MLE epoch {epoch}/{epochs}] loss={bce.item():.4e} + L2={l2.item():.4e}")

    return reward_model.reward.weight.data.view(-1).clone()