"""
Extended PGR agent that handles cost in transitions and supports rare-event buffer.

Extends REDQRLPDCondAgent to:
1. Store cost in replay buffers
2. Include cost dimension in diffusion training data
3. Mix rare-event transitions into training batches
4. Apply Lagrangian cost constraint to rewards
"""

import numpy as np
import torch
from torch import Tensor

from redq.algos.core import soft_update_model1_with_model2
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent
from safety.cost_replay_buffer import CostReplayBuffer, RareEventBuffer


def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


class CostREDQRLPDCondAgent(REDQRLPDCondAgent):
    """PGR agent extended with cost handling and optional rare-event buffer."""

    def __init__(self, cond_hidden_size, diffusion_buffer_size=int(1e6),
                 diffusion_sample_ratio=0.5,
                 use_rare_buffer=False,
                 rare_buffer_size=500,
                 rare_batch_ratio=0.2,
                 rare_loss_weight=5.0,
                 use_lagrangian=False,
                 cost_limit=2.0,
                 lambda_lr=1e-2,
                 lambda_init=0.0,
                 *args, **kwargs):
        super().__init__(cond_hidden_size, diffusion_buffer_size,
                         diffusion_sample_ratio, *args, **kwargs)

        # Replace replay buffers with cost-aware versions
        self.replay_buffer = CostReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        self.diffusion_buffer = CostReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=diffusion_buffer_size)

        # Rare-event buffer
        self.use_rare_buffer = use_rare_buffer
        self.rare_batch_ratio = rare_batch_ratio
        self.rare_loss_weight = rare_loss_weight
        if use_rare_buffer:
            self.rare_buffer = RareEventBuffer(
                self.obs_dim, self.act_dim, max_size=rare_buffer_size)
        else:
            self.rare_buffer = None

        # Lagrangian cost constraint
        self.use_lagrangian = use_lagrangian
        if use_lagrangian:
            self.lam = lambda_init
            self.lambda_lr = lambda_lr
            self.cost_limit = cost_limit
        else:
            self.lam = 0.0

    def store_data(self, o, a, r, o2, d, cost=0.0):
        """Store transition with cost."""
        self.replay_buffer.store(o, a, r, o2, d, cost=cost)
        # Store in rare buffer if cost > 0
        if self.use_rare_buffer and cost > 0:
            self.rare_buffer.add(o, a, r, o2, d, cost)

    def record_episode_cost(self, episode_cost):
        """Update Lagrange multiplier based on episode cost."""
        if self.use_lagrangian:
            # Dual gradient descent: increase lambda if cost > limit
            grad = episode_cost - self.cost_limit
            self.lam = max(0.0, self.lam + self.lambda_lr * grad)

    def get_effective_reward(self, rews_tensor, cost_tensor):
        """Apply Lagrangian penalty: r_eff = r - lambda * c."""
        if self.use_lagrangian and self.lam > 0:
            return rews_tensor - self.lam * cost_tensor
        return rews_tensor

    def sample_data(self, batch_size):
        """Sample mixed batch from real + diffusion + rare buffers."""
        # Determine batch splits
        rare_size = 0
        if self.use_rare_buffer and self.rare_buffer is not None and len(self.rare_buffer) > 0:
            rare_size = int(batch_size * self.rare_batch_ratio)

        diffusion_batch_size = int((batch_size - rare_size) * self.diffusion_sample_ratio)
        online_batch_size = batch_size - diffusion_batch_size - rare_size

        # If diffusion buffer doesn't have enough, fall back
        if self.diffusion_buffer.size < diffusion_batch_size:
            diffusion_batch_size = 0
            online_batch_size = batch_size - rare_size

        # Sample from each buffer
        online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)

        obs_parts = [online_batch['obs1']]
        obs_next_parts = [online_batch['obs2']]
        acts_parts = [online_batch['acts']]
        rews_parts = [online_batch['rews']]
        cost_parts = [online_batch['cost']]
        done_parts = [online_batch['done']]

        if diffusion_batch_size > 0:
            diff_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
            obs_parts.append(diff_batch['obs1'])
            obs_next_parts.append(diff_batch['obs2'])
            acts_parts.append(diff_batch['acts'])
            rews_parts.append(diff_batch['rews'])
            cost_parts.append(diff_batch['cost'])
            done_parts.append(diff_batch['done'])

        if rare_size > 0:
            rare_batch = self.rare_buffer.sample_batch(rare_size)
            if rare_batch is not None:
                obs_parts.append(rare_batch['obs1'])
                obs_next_parts.append(rare_batch['obs2'])
                acts_parts.append(rare_batch['acts'])
                rews_parts.append(rare_batch['rews'])
                cost_parts.append(rare_batch['cost'])
                done_parts.append(rare_batch['done'])

        obs_tensor = Tensor(np.concatenate(obs_parts, axis=0)).to(self.device)
        obs_next_tensor = Tensor(np.concatenate(obs_next_parts, axis=0)).to(self.device)
        acts_tensor = Tensor(np.concatenate(acts_parts, axis=0)).to(self.device)
        rews_tensor = Tensor(np.concatenate(rews_parts, axis=0)).unsqueeze(1).to(self.device)
        cost_tensor = Tensor(np.concatenate(cost_parts, axis=0)).unsqueeze(1).to(self.device)
        done_tensor = Tensor(np.concatenate(done_parts, axis=0)).unsqueeze(1).to(self.device)

        # Apply Lagrangian penalty to rewards
        effective_rews = self.get_effective_reward(rews_tensor, cost_tensor)

        return obs_tensor, obs_next_tensor, acts_tensor, effective_rews, done_tensor

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = CostReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim,
            size=self.diffusion_buffer.max_size)
