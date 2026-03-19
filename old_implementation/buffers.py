"""
Replay buffers: standard uniform replay and rare-event memory bank.

In RL, we store past experience and re-sample it for training. This file
has two buffers:
  1. ReplayBuffer     — standard ring buffer, stores everything uniformly
  2. RareEventBuffer  — small separate store for catastrophic (cost>0) transitions
"""

import random
import numpy as np
import torch

from config import BUFFER_SIZE, RARE_BUFFER_SIZE, DEVICE


class ReplayBuffer:
    """
    Standard fixed-size replay buffer using a ring/circular buffer strategy.
    Stores transitions as (state, action, reward, cost, next_state, done).

    When full, new transitions overwrite the oldest ones (FIFO).
    Pre-allocates numpy arrays for speed — avoids appending to lists.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0       # points to the next slot to write into
        self.size = 0      # how many valid transitions are stored

        # Pre-allocate arrays — much faster than growing a list
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.costs = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def add(self, state, action, reward, cost, next_state, done):
        """
        Store one transition. If buffer is full, overwrites the oldest entry.
        The pointer wraps around using modulo (ring buffer).
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size   # wrap around
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Uniformly sample a batch of transitions. Returns GPU tensors.
        This is what SAC uses every gradient step — random past experience.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.costs[idx]).to(DEVICE),
            torch.FloatTensor(self.next_states[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
        )

    def sample_with_idx(self, n: int):
        """
        Like sample(), but also returns the buffer indices of the sampled
        transitions. PGR needs these indices to:
          1. Compute curiosity scores for specific transitions
          2. Later retrieve those same transitions as flat vectors for
             diffusion training via get_transitions()

        Uses replace=False so each transition appears at most once.
        """
        n = min(n, self.size)
        idx = np.random.choice(self.size, n, replace=False)
        return (
            torch.FloatTensor(self.states[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.next_states[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.costs[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
            idx,
        )

    def get_transitions(self, idx):
        """
        Return transitions as a single flat numpy array per transition:
            [s, a, r, c, s', d]

        This flat format is what the diffusion model trains on — it learns
        to generate entire transition tuples as one vector.
        The done flag is included so the diffusion model learns when episodes
        end (critical for correct Bellman targets on synthetic data).
        """
        return np.concatenate([
            self.states[idx],
            self.actions[idx],
            self.rewards[idx].reshape(-1, 1),
            self.costs[idx].reshape(-1, 1),
            self.next_states[idx],
            self.dones[idx].reshape(-1, 1),
        ], axis=-1)

    def __len__(self):
        return self.size


class RareEventBuffer:
    """
    Small memory bank specifically for catastrophic/hazardous transitions.

    This is our key contribution: PGR's curiosity-based relevance function
    will eventually stop finding hazard transitions "novel" (because the ICM
    learns to predict them). When that happens, the diffusion model stops
    generating hazard-adjacent synthetic data, and the policy "forgets" how
    to avoid hazards.

    By keeping a dedicated buffer of hazardous transitions and injecting
    them into diffusion training, we prevent this forgetting.

    Currently uses simple FIFO eviction (oldest out first).
    TODO: Could be improved with severity-based priority.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = RARE_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: list[dict] = []    # list of transition dicts

    def add(self, state, action, reward, cost, next_state, done):
        """
        Store a hazardous transition. Called by the agent whenever cost > 0.
        If buffer is full, drops the oldest entry (FIFO).
        """
        self.buffer.append({
            "state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": float(reward),
            "cost": float(cost),
            "next_state": np.array(next_state, dtype=np.float32),
            "done": float(done),
        })
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)    # drop oldest

    def get_transitions(self, batch_size: int):
        """
        Sample batch_size transitions and return them as flat vectors,
        matching the format expected by the diffusion model:
            [s, a, r, c, s', d]

        Returns None if the buffer is empty.
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
        """
        Sample batch_size transitions as GPU tensors for direct SAC updates.

        This is Fix 3: instead of only injecting rare events into diffusion
        training, we also inject them directly into the SAC policy batch.
        This bypasses the diffusion quality bottleneck — the policy gets
        REAL hazard data every gradient step, guaranteed.

        Returns: (states, actions, rewards, costs, next_states, dones) as GPU tensors,
                 or None if buffer is empty.
        """
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
