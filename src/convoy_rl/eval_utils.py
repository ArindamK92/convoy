"""Evaluation utility helpers."""

import torch

from torch.utils.data import DataLoader
from rl4co.envs import CVRPTWEnv
from rl4co.models import AttentionModel


def extract_reward(metrics: dict) -> float:
    """Extract a scalar reward from metric dictionaries with varying key names."""
    for key in ("test/reward", "test/reward/0"):
        if key in metrics:
            return float(metrics[key])
    reward_keys = [k for k in metrics if "reward" in k]
    if not reward_keys:
        raise RuntimeError(f"Could not find reward key in test metrics: {metrics}")
    return float(metrics[reward_keys[0]])


def evaluate_policy_on_dataset(
    model: AttentionModel,
    env: CVRPTWEnv,
    dataset,
    batch_size: int,
) -> float:
    """Compute mean reward of a model on a given dataset with greedy decoding."""
    was_training = model.training
    model.eval()
    rewards = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            td = env.reset(batch)
            out = model.policy(td, env, phase="test", decode_type="greedy")
            rewards.append(out["reward"].detach().cpu())
    if was_training:
        model.train()
    return float(torch.cat(rewards, dim=0).mean())
