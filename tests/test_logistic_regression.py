import torch
import pytest

from isaac_rlhf.config import RlhfCfg
from isaac_rlhf.modules import LinearReward
from isaac_rlhf.storage import FeatureStorageRlhf
from isaac_rlhf.algorithms import logistic_regression


@pytest.fixture(autouse=True)
def stub_wandb(monkeypatch):
    class DummyWandb:
        def __init__(self):
            self.logged = []

        def log(self, data, step=None):
            self.logged.append((data, step))

    dummy = DummyWandb()
    monkeypatch.setattr(logistic_regression, "wandb", dummy)
    yield dummy


def _make_storage(num_samples=20):
    cfg = RlhfCfg(
        task="gridworld",
        num_features=2,
        gt_params={"f1": 0.0, "f2": 0.0},
        lambda_=1.0,
        device="cpu",
    )
    storage = FeatureStorageRlhf(cfg=cfg, max_ep_buffers_size=16, max_dataset_size=num_samples)
    return storage


def test_train_reward_model_learns_separable_weights():
    storage = _make_storage(num_samples=40)

    positive = torch.tensor([2.0, 0.0])
    negative = torch.tensor([-2.0, 0.0])

    for i in range(20):
        storage.X_hist[i] = positive
        storage.y_hist[i] = 1
    for i in range(20, 40):
        storage.X_hist[i] = negative
        storage.y_hist[i] = 0
    storage.hist_mask[:40] = True
    storage.hist_step = 40

    reward = LinearReward(num_features=2, gt_params=torch.zeros(2))
    with torch.no_grad():
        reward.reward.weight.zero_()

    trained_params = logistic_regression.train_reward_model(
        reward_model=reward,
        feature_storage=storage,
        lr=0.1,
        l2_reg=0.0,
        epochs=200,
        batch_size=8,
        test_split=0.1,
        device="cpu",
    )

    assert trained_params.shape[0] == 2
    logits_pos = reward.get_reward(positive.unsqueeze(0))
    logits_neg = reward.get_reward(negative.unsqueeze(0))
    assert logits_pos.item() > logits_neg.item()


def test_train_reward_model_returns_initial_params_when_dataset_empty():
    storage = _make_storage(num_samples=4)
    reward = LinearReward(num_features=2, gt_params=torch.zeros(2))
    with torch.no_grad():
        reward.reward.weight.fill_(0.5)

    initial = reward.get_reward_params().clone()
    with pytest.warns(UserWarning):
        result = logistic_regression.train_reward_model(
            reward_model=reward,
            feature_storage=storage,
            lr=0.01,
            l2_reg=0.0,
            epochs=10,
            batch_size=2,
            test_split=0.0,
            device="cpu",
        )

    assert torch.allclose(result, initial)
