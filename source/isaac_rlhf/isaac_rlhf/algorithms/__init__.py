"""Implementation of different IRL algorithms."""

from .rlhf import RlhfTaskManager
from .logistic_regression import train_reward_model

__all__ = ["RlhfTaskManager", "train_reward_model"]