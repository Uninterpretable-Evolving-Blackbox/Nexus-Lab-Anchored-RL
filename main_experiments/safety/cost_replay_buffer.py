"""
Extended replay buffer that stores cost alongside standard (s, a, r, s', d) transitions.

Also includes the RareEventBuffer for storing hazardous transitions separately.
"""

import numpy as np
import random


class CostReplayBuffer:
    """
    FIFO replay buffer extended with a cost dimension.
    Drop-in replacement for REDQ's ReplayBuffer with an extra cost_buf.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost=0.0):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            cost=self.cost_buf[idxs],
            done=self.done_buf[idxs],
            idxs=idxs,
        )


class RareEventBuffer:
    """
    Small FIFO buffer for hazardous transitions (cost > 0).

    Stores transitions and provides:
    - get_flat_transitions(): for mixing into diffusion training batches
    - sample_batch(): for mixing into SAC training batches
    """

    def __init__(self, obs_dim, act_dim, max_size=500):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer = []

    def add(self, obs, act, rew, next_obs, done, cost):
        self.buffer.append({
            'obs': np.array(obs, dtype=np.float32),
            'act': np.array(act, dtype=np.float32),
            'rew': float(rew),
            'cost': float(cost),
            'next_obs': np.array(next_obs, dtype=np.float32),
            'done': float(done),
        })
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_flat_transitions(self, batch_size, include_cost=True):
        """
        Return flat transition vectors for diffusion training:
            [obs, act, rew, cost, next_obs]  (if include_cost)
            [obs, act, rew, next_obs]         (if not include_cost)
        """
        if len(self.buffer) == 0:
            return None
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        rows = []
        for t in batch:
            parts = [t['obs'], t['act'], [t['rew']]]
            if include_cost:
                parts.append([t['cost']])
            parts.append(t['next_obs'])
            rows.append(np.concatenate(parts))
        return np.stack(rows)

    def sample_batch(self, batch_size):
        """Sample as a dict matching CostReplayBuffer format."""
        if len(self.buffer) == 0:
            return None
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        return dict(
            obs1=np.stack([t['obs'] for t in batch]),
            obs2=np.stack([t['next_obs'] for t in batch]),
            acts=np.stack([t['act'] for t in batch]),
            rews=np.array([t['rew'] for t in batch], dtype=np.float32),
            cost=np.array([t['cost'] for t in batch], dtype=np.float32),
            done=np.array([t['done'] for t in batch], dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
