import torch

from isaac_rlhf.modules import LinearReward


def test_linear_reward_predictions_match_weights_and_ground_truth():
    gt_params = torch.tensor([1.0, -1.0, 0.5])
    reward = LinearReward(num_features=3, gt_params=gt_params)

    with torch.no_grad():
        reward.reward.weight.copy_(torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32))

    features = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    predicted = reward.get_reward(features)
    expected_pred = torch.mv(features, torch.tensor([0.2, 0.3, 0.4]))
    assert torch.allclose(predicted, expected_pred)

    gt_reward = reward.get_gt_reward(features)
    expected_gt = torch.mv(features, gt_params)
    assert torch.allclose(gt_reward, expected_gt)


def test_get_reward_params_returns_cpu_copy():
    gt_params = torch.tensor([0.0, 0.0, 0.0])
    reward = LinearReward(num_features=3, gt_params=gt_params)

    params = reward.get_reward_params()
    assert params.device.type == "cpu"
    assert params.shape == (3,)
