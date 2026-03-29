"""
PGR variant with separate hazard memory and reward memory.

This keeps the existing rare-event mechanism for costly transitions while
replacing salience replay with a true reward-based memory buffer.
"""

import numpy as np
import torch

from agents import SACPGRAgent
from buffers import RareEventBuffer
from config import (
    BATCH_SIZE,
    DEVICE,
    HIGH_REWARD_BATCH_RATIO,
    HIGH_REWARD_THRESHOLD,
    HIGH_REWARD_WEIGHT,
    PGR_START_BUFFER,
    RARE_BATCH_RATIO,
    RARE_WEIGHT,
    REPLAY_RATIO,
    SAFE_EPISODE_COST_LIMIT,
    TOP_K_HIGH_REWARD_PER_EPISODE,
)
from high_reward_buffer import HighRewardBuffer
from networks import normalize_scores


class SACPGRMemoryRewardAgent(SACPGRAgent):
    """PGR + hazard memory + reward memory."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        self.rare_buffer = RareEventBuffer(state_dim, action_dim)
        self.high_reward_buffer = HighRewardBuffer(state_dim, action_dim)
        self.train_steps = 0
        self.log_interval = 250

    def add_transition(self, s, a, r, c, ns, d):
        """Store transitions in replay and hazard memory as appropriate."""
        self.buffer.add(s, a, r, c, ns, d)
        if c > 0:
            self.rare_buffer.add(s, a, r, c, ns, d)

    def store_good_episode_transitions(self, episode_transitions, ep_reward, ep_cost):
        """Store top high-reward safe transitions, but only from safe episodes."""
        _ = ep_reward  # kept for API symmetry and future episode-level criteria
        if ep_cost > SAFE_EPISODE_COST_LIMIT:
            return

        good_transitions = [t for t in episode_transitions if t[2] >= HIGH_REWARD_THRESHOLD and t[3] == 0]
        if not good_transitions:
            return

        good_transitions.sort(key=lambda t: t[2], reverse=True)
        top_transitions = good_transitions[:TOP_K_HIGH_REWARD_PER_EPISODE]

        for s, a, r, c, ns, d in top_transitions:
            self.high_reward_buffer.add(s, a, r, c, ns, d)

    def _maybe_log_buffers(self):
        if self.train_steps % self.log_interval != 0:
            return
        print(
            "[PGR+Memory+Reward] "
            f"replay={len(self.buffer)} "
            f"rare={len(self.rare_buffer)} "
            f"high_reward={len(self.high_reward_buffer)}"
        )

    def _refresh_diffusion_normalization(self, pool_trans):
        self.trans_mean = torch.FloatTensor(pool_trans.mean(axis=0)).to(DEVICE)
        std = np.maximum(pool_trans.std(axis=0), 1e-2)
        cost_idx = self.state_dim + self.action_dim + 1
        std[cost_idx] = 1.0
        std[-1] = 1.0
        self.trans_std = torch.FloatTensor(std).to(DEVICE)

    def _assemble_sac_batch(self, scores_np=None):
        diffusion_ready = scores_np is not None and self.diffusion_updates > 2000
        n_syn = int(BATCH_SIZE * REPLAY_RATIO) if diffusion_ready else 0
        n_rare_sac = min(int(BATCH_SIZE * RARE_BATCH_RATIO), len(self.rare_buffer))
        n_reward_sac = min(int(BATCH_SIZE * HIGH_REWARD_BATCH_RATIO), len(self.high_reward_buffer))
        n_real = max(0, BATCH_SIZE - n_syn - n_rare_sac - n_reward_sac)

        parts_s = []
        parts_a = []
        parts_r = []
        parts_c = []
        parts_ns = []
        parts_d = []

        if n_real > 0:
            real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(n_real)
            parts_s.append(real_s)
            parts_a.append(real_a)
            parts_r.append(real_r)
            parts_c.append(real_c)
            parts_ns.append(real_ns)
            parts_d.append(real_d)

        if n_syn > 0:
            syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)
            parts_s.append(syn_s)
            parts_a.append(syn_a)
            parts_r.append(syn_r)
            parts_c.append(syn_c)
            parts_ns.append(syn_ns)
            parts_d.append(syn_d)

        if n_rare_sac > 0:
            rare_sample = self.rare_buffer.sample(n_rare_sac)
            if rare_sample is not None:
                rs, ra, rr, rc, rns, rd = rare_sample
                parts_s.append(rs)
                parts_a.append(ra)
                parts_r.append(rr)
                parts_c.append(rc)
                parts_ns.append(rns)
                parts_d.append(rd)

        if n_reward_sac > 0:
            reward_sample = self.high_reward_buffer.sample(n_reward_sac)
            if reward_sample is not None:
                hs, ha, hr, hc, hns, hd = reward_sample
                parts_s.append(hs)
                parts_a.append(ha)
                parts_r.append(hr)
                parts_c.append(hc)
                parts_ns.append(hns)
                parts_d.append(hd)

        return (
            torch.cat(parts_s),
            torch.cat(parts_a),
            torch.cat(parts_r),
            torch.cat(parts_c),
            torch.cat(parts_ns),
            torch.cat(parts_d),
        )

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        self.train_steps += 1
        self._maybe_log_buffers()
        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(min(5000, len(self.buffer)))
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            pool_trans = self.buffer.get_transitions(idx)
            self._refresh_diffusion_normalization(pool_trans)
            self._train_icm()

            n_rare_diff = min(int(BATCH_SIZE * RARE_BATCH_RATIO), len(self.rare_buffer))
            n_reward_diff = min(int(BATCH_SIZE * HIGH_REWARD_BATCH_RATIO), len(self.high_reward_buffer))
            n_normal = max(0, BATCH_SIZE - n_rare_diff - n_reward_diff)

            batch_idx = np.random.choice(len(idx), n_normal, replace=True)
            normal_trans = self.buffer.get_transitions(idx[batch_idx])
            normal_scores = scores_np[batch_idx]
            normal_weights = np.ones(len(normal_trans))

            reward_prompt = np.quantile(scores_np, 0.75)
            all_trans = normal_trans
            all_scores = normal_scores
            all_weights = normal_weights

            if n_rare_diff > 0:
                rare_trans = self.rare_buffer.get_transitions(n_rare_diff)
                if rare_trans is not None:
                    rare_scores = np.ones(len(rare_trans)) * scores_np.max()
                    rare_weights = np.ones(len(rare_trans)) * RARE_WEIGHT
                    all_trans = np.vstack([all_trans, rare_trans])
                    all_scores = np.concatenate([all_scores, rare_scores])
                    all_weights = np.concatenate([all_weights, rare_weights])

            if n_reward_diff > 0:
                reward_trans = self.high_reward_buffer.get_transitions(n_reward_diff)
                if reward_trans is not None:
                    reward_scores = np.ones(len(reward_trans)) * reward_prompt
                    reward_weights = np.ones(len(reward_trans)) * HIGH_REWARD_WEIGHT
                    all_trans = np.vstack([all_trans, reward_trans])
                    all_scores = np.concatenate([all_scores, reward_scores])
                    all_weights = np.concatenate([all_weights, reward_weights])

            all_weights = all_weights / all_weights.mean()
            self._train_diffusion(all_trans, all_scores, all_weights)
            states, actions, rewards, costs, next_states, dones = self._assemble_sac_batch(scores_np)
        else:
            states, actions, rewards, costs, next_states, dones = self._assemble_sac_batch()

        self._sac_update(states, actions, rewards, costs, next_states, dones)

        if self.train_steps % 100 == 0:
            print(
                f"[PGR+Memory+Reward] "
                f"replay={len(self.buffer)} "
                f"rare={len(self.rare_buffer)} "
                f"high_reward={len(self.high_reward_buffer)} "
                f"diff_updates={self.diffusion_updates}"
            )
