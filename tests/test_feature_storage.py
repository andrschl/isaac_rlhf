import torch
import pytest

from isaac_rlhf.config import RlhfCfg
from isaac_rlhf.storage import FeatureStorageRlhf


def _make_cfg(**overrides: object) -> RlhfCfg:
    cfg = RlhfCfg()
    num_features = overrides.get("num_features", 3)
    cfg.num_features = num_features
    cfg.gt_params = {f"f{i}": float(i) for i in range(num_features)}
    cfg.num_trajectories_per_run = overrides.get("num_trajectories_per_run", 4)
    cfg.rlhf_algorithm = overrides.get("rlhf_algorithm", "vanilla")
    cfg.lambda_ = float(overrides.get("lambda_", 1.0))
    cfg.device = "cpu"
    return cfg


def test_fill_storage_vanilla_uses_all_available_pairs():
    cfg = _make_cfg()
    storage = FeatureStorageRlhf(cfg=cfg, max_ep_buffers_size=8)

    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    storage.fill_storage([{"features": features}])

    assert storage.policy_step == 1
    assert storage.step == 2
    assert storage.policy_ids[0].tolist() == [0, 0]
    assert storage.policy_ids[1].tolist() == [0, 0]

    initial = cfg.lambda_ * torch.eye(cfg.num_features)
    diffs = [features[0] - features[1], features[2] - features[3]]
    expected = initial + sum(torch.outer(d, d) for d in diffs)
    assert torch.allclose(storage.curr_V, expected)


def test_fill_storage_ts_last_limits_to_previous_pairs():
    cfg = _make_cfg(rlhf_algorithm="ts_last", num_features=2)
    storage = FeatureStorageRlhf(cfg=cfg, max_ep_buffers_size=8)

    first_policy = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.0, 0.0],
        ]
    )

    storage.fill_storage([{"features": first_policy}])
    assert storage.policy_step == 1
    assert storage.prev_pairs == 2

    second_policy = torch.tensor(
        [
            [0.2, 0.2],
            [0.1, 0.0],
            [0.3, 0.4],
            [0.0, 0.2],
            [0.6, 0.6],
            [0.4, 0.1],
        ]
    )

    with pytest.warns(UserWarning, match="More trajectory pairs"):
        storage.fill_storage([{"features": second_policy}])

    assert storage.policy_step == 2
    assert storage.step == 4
    assert storage.policy_ids[2].tolist() == [1, 0]
    assert storage.policy_ids[3].tolist() == [1, 0]

    initial = cfg.lambda_ * torch.eye(cfg.num_features)
    diffs_first = [
        first_policy[0] - first_policy[1],
        first_policy[2] - first_policy[3],
    ]
    diffs_second = [
        second_policy[0] - first_policy[1],
        second_policy[2] - first_policy[3],
    ]
    expected = initial
    for diff in diffs_first + diffs_second:
        expected = expected + torch.outer(diff, diff)
    assert torch.allclose(storage.curr_V, expected)


def test_fill_storage_ts_double_requires_even_results():
    cfg = _make_cfg(rlhf_algorithm="ts_double")
    storage = FeatureStorageRlhf(cfg=cfg)

    with pytest.raises(ValueError, match="even number"):
        storage.fill_storage([{"features": torch.zeros((2, cfg.num_features))}])


def test_fill_storage_ts_double_uses_minimum_trajectory_count():
    cfg = _make_cfg(rlhf_algorithm="ts_double", num_features=2)
    storage = FeatureStorageRlhf(cfg=cfg, max_ep_buffers_size=8)

    policy_a = torch.tensor(
        [
            [0.2, 0.0],
            [0.3, 0.1],
            [0.4, 0.2],
        ]
    )
    policy_b = torch.tensor(
        [
            [0.1, 0.2],
            [0.2, 0.2],
        ]
    )

    with pytest.warns(UserWarning, match="Policy pair produced a different number"):
        storage.fill_storage([
            {"features": policy_a},
            {"features": policy_b},
        ])

    assert storage.policy_step == 1
    assert storage.step == 2
    assert storage.policy_ids[0].tolist() == [0, 1]
    assert storage.policy_ids[1].tolist() == [0, 1]

    initial = cfg.lambda_ * torch.eye(cfg.num_features)
    diffs = [
        policy_a[0] - policy_b[0],
        policy_a[1] - policy_b[1],
    ]
    expected = initial + sum(torch.outer(d, d) for d in diffs)
    assert torch.allclose(storage.curr_V, expected)


def test_get_preferences_updates_design_matrices_and_history():
    cfg = _make_cfg(rlhf_algorithm="vanilla", num_features=2, lambda_=1.0)
    storage = FeatureStorageRlhf(cfg=cfg, max_ep_buffers_size=8, max_dataset_size=16)

    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )

    storage.fill_storage([{"features": features}])
    curr_V_before = storage.curr_V.clone()

    class DummyReward:
        def __init__(self):
            self.params = torch.tensor([0.3, -0.1])

        def get_gt_reward(self, X):
            return X @ self.params

    torch.manual_seed(0)
    X_new, y_new = storage.get_preferences(DummyReward())

    assert X_new.shape[1] == cfg.num_features
    assert y_new.shape[0] == X_new.shape[0]
    assert torch.allclose(storage.V, curr_V_before, atol=1e-6)
    identity = torch.eye(cfg.num_features)
    assert torch.allclose(storage.V @ storage.V_inv, identity, atol=1e-5)
    assert storage.hist_step == X_new.size(0)
    assert torch.allclose(storage.X_hist[:X_new.size(0)], X_new)
    assert storage.mask.sum() == 0  # cleared after sampling comparisons
