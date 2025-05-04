import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
import wandb
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


# Use Adam with L2 regularization to fit a logistic regression model


def train_reward_model(
    reward_model: LinearReward,
    feature_storage: FeatureStorageRlhf,
    lr: float = 1e-3,
    l2_reg: float = 1e-6,
    epochs: int = 500,
    batch_size: int = 64,
    test_split: float = 0.2,
    iter=None,
    num_workers: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Train the reward_model with an L2‑regularized logistic regression,
    perform a train/test split, and log train/test accuracy each epoch.
    """
    reward_model.to(device).train()

    # load full dataset
    dataset = PreferenceDataset(feature_storage)
    N = len(dataset)
    print(f"[DEBUG] Dataset size: {N}")
    if N == 0:
        warnings.warn("No preference data; returning initial reward parameters.")
        return reward_model.reward.weight.data.view(-1).clone()

    # split into train / test
    n_test = int(test_split * N)
    n_train = N - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    optimizer = Adam(reward_model.reward.parameters(), lr=lr)

    def evaluate(loader):
        reward_model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = reward_model.get_reward(Xb)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                total += yb.size(0)
                correct += (preds == yb).sum().item()
        reward_model.train()
        return correct / total if total > 0 else 0.0

    for epoch in range(1, epochs + 1):
        bce_tot = 0.0
        gt_bce_tot = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = reward_model.get_reward(Xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            w = reward_model.reward.weight.view(-1)
            l2 = 0.5 * l2_reg * torch.dot(w, w)
            (bce + l2).backward()
            optimizer.step()

            # compute losses for logging
            if epoch == epochs or epoch % 100 == 0:
                with torch.inference_mode():
                    bce_tot += bce.item() / len(train_loader)
                    gt_logits = reward_model.get_gt_reward(Xb)
                    gt_bce_tot += F.binary_cross_entropy_with_logits(
                        gt_logits, yb
                    ).item() / len(train_loader)

        # log to console
        if epoch == epochs or epoch % 100 == 0:
            train_acc = evaluate(train_loader)
            test_acc = evaluate(test_loader)
            print(
                f"[MLE epoch {epoch}/{epochs}] bce={bce_tot:.4e}, gt_bce={gt_bce_tot:.4e}, L2={l2.item():.4e}"
                f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
            )

    wandb.log(
        {
            "mle/epoch": epoch,
            "mle/bce": bce_tot,
            "mle/gt_bce": gt_bce_tot,
            "mle/l2": l2.item(),
            "mle/train_accuracy": train_acc,
            "mle/test_accuracy": test_acc,
        },
        step=iter,
    )

    return reward_model.reward.weight.data.view(-1).clone()


# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.utils.data import Dataset, DataLoader
# import warnings
# from isaac_rlhf.modules import LinearReward
# from isaac_rlhf.storage import FeatureStorageRlhf

# class PreferenceDataset(Dataset):
#     def __init__(self, feature_storage: FeatureStorageRlhf):
#         X, y = feature_storage.get_Xy()
#         if X is None or y is None:
#             self.X = torch.empty(0)
#             self.y = torch.empty(0, dtype=torch.long)
#         else:
#             self.X = X
#             self.y = y

#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# def train_reward_model(
#     reward_model: LinearReward,
#     feature_storage: FeatureStorageRlhf,
#     lr: float = 1e-3,
#     l2_reg: float = 1e-6,
#     epochs: int = 500,
#     batch_size: int = 64,
#     test_split: float = 0.2,
#     num_workers: int = 0,
#     device: str = "cpu",
# ) -> torch.Tensor:

#     reward_model.to(device).train()

#     # build DataLoader
#     dataset = PreferenceDataset(feature_storage)

#     if len(dataset) == 0:
#         warnings.warn("No preference data; returning initial reward parameters.")
#         return reward_model.reward.weight.data.view(-1).clone()

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=(device != "cpu"),
#     )

#     optimizer = Adam(reward_model.reward.parameters(), lr=lr)
#     for epoch in range(1, epochs + 1):
#         for Xb, yb in loader:
#             Xb, yb = Xb.to(device), yb.to(device).float()
#             optimizer.zero_grad()
#             logits = reward_model.get_reward(Xb)
#             bce    = F.binary_cross_entropy_with_logits(logits, yb)
#             w      = reward_model.reward.weight.view(-1)
#             l2     = 0.5 * l2_reg * torch.dot(w, w)
#             (bce + l2).backward()
#             optimizer.step()
#         if epoch % 100 == 0 or epoch == 1:
#             print(f"[MLE epoch {epoch}/{epochs}] loss={bce.item():.4e} + L2={l2.item():.4e}")

#     return reward_model.reward.weight.data.view(-1).clone()
