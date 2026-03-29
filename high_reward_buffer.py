"""
Reward-based memory buffer for retaining genuinely good outcomes.

Unlike the salience buffer, this memory is populated purely from reward.
Transitions enter only if their reward clears a configurable threshold,
and when full the buffer keeps the highest-reward examples seen so far.
"""

import random

import numpy as np
import torch

from config import DEVICE, HIGH_REWARD_BUFFER_SIZE, HIGH_REWARD_THRESHOLD


class HighRewardBuffer:
    """Memory bank that keeps the highest-reward transitions seen so far."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = HIGH_REWARD_BUFFER_SIZE,
        reward_threshold: float = HIGH_REWARD_THRESHOLD,
    ):
        self.max_size = max_size
        self.reward_threshold = reward_threshold
        self.buffer: list[dict] = []

    def add(self, state, action, reward, cost, next_state, done):
        """
        Store a transition if it is both high-reward and safe.

        When full, evicts the current lowest-reward transition only if the new
        one is better.
        """
        reward = float(reward)
        cost = float(cost)
        if reward < self.reward_threshold or cost > 0:
            return

        transition = {
            "state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": reward,
            "cost": cost,
            "next_state": np.array(next_state, dtype=np.float32),
            "done": float(done),
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            return

        min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["reward"])
        if reward > self.buffer[min_idx]["reward"]:
            self.buffer[min_idx] = transition

    def get_transitions(self, batch_size: int):
        """
        Sample transitions as flat vectors for diffusion training:
            [s, a, r, c, s', d]
        """
        batch_size = min(batch_size, len(self.buffer))
        if batch_size == 0:
            return None
        batch = random.sample(self.buffer, batch_size)
        return np.stack([
            np.concatenate([
                t["state"], t["action"], [t["reward"]], [t["cost"]], t["next_state"], [t["done"]]
            ])
            for t in batch
        ])

    def sample(self, batch_size: int):
        """Sample transitions as GPU tensors for direct SAC updates."""
        batch_size = min(batch_size, len(self.buffer))
        if batch_size == 0:
            return None
        batch = random.sample(self.buffer, batch_size)
        return (
            torch.FloatTensor(np.stack([t["state"] for t in batch])).to(DEVICE),
            torch.FloatTensor(np.stack([t["action"] for t in batch])).to(DEVICE),
            torch.FloatTensor(np.array([t["reward"] for t in batch], dtype=np.float32)).to(DEVICE),
            torch.FloatTensor(np.array([t["cost"] for t in batch], dtype=np.float32)).to(DEVICE),
            torch.FloatTensor(np.stack([t["next_state"] for t in batch])).to(DEVICE),
            torch.FloatTensor(np.array([t["done"] for t in batch], dtype=np.float32)).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)
