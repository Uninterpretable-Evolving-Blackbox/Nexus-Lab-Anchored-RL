"""
DMC environment wrapper that adds a binary hazard cost signal.

Wraps cheetah-run (via dmcgym) so that excessive forward velocity
triggers cost=1. Tuned so ~1-5% of timesteps incur cost for a
learning agent.
"""

import gym
import numpy as np


class HazardWrapper(gym.Wrapper):
    """
    Wraps a Gym environment to add a binary cost signal.

    The cost is stored in the info dict under 'cost'.
    The reward is unchanged. Cost = 1 if |forward_velocity| > threshold.

    For cheetah-run-v0 (DMC via dmcgym):
        obs is a flat vector. The velocity components are in the latter
        portion. We use a velocity threshold that triggers ~1-5% of the
        time for a trained agent.
    """

    def __init__(self, env, velocity_threshold=7.0, velocity_idx=None):
        """
        Args:
            env: base gym environment (already wrapped by dmcgym)
            velocity_threshold: cost=1 if |velocity| exceeds this
            velocity_idx: index into the flat obs for the velocity signal.
                          If None, auto-detect based on env name.
        """
        super().__init__(env)
        self.velocity_threshold = velocity_threshold
        self.velocity_idx = velocity_idx
        self.total_cost = 0.0
        self.episode_cost = 0.0
        self.total_steps = 0
        self.cost_steps = 0

    def reset(self, **kwargs):
        self.episode_cost = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1

        # Compute cost from velocity
        cost = self._compute_cost(obs)
        info['cost'] = cost
        info['episode_cost'] = self.episode_cost

        return obs, reward, done, info

    def _compute_cost(self, obs):
        """Binary cost: 1 if velocity exceeds threshold, 0 otherwise."""
        if self.velocity_idx is not None:
            vel = abs(obs[self.velocity_idx])
        else:
            # For cheetah-run-v0: obs is flat. The first obs_dim/2 are positions,
            # second half are velocities. The forward velocity is typically the
            # first velocity component.
            # DMC cheetah-run obs: 17-dim (rootx excluded).
            # Velocities start at index 8 (or 9 depending on version).
            # We use the root velocity (index 8) as the hazard signal.
            vel = abs(obs[8]) if len(obs) > 8 else 0.0

        cost = 1.0 if vel > self.velocity_threshold else 0.0
        self.total_cost += cost
        self.episode_cost += cost
        if cost > 0:
            self.cost_steps += 1
        return cost

    @property
    def hazard_rate(self):
        if self.total_steps == 0:
            return 0.0
        return self.cost_steps / self.total_steps


class TwoPhaseHazardWrapper(gym.Wrapper):
    """
    Wraps a HazardWrapper to suppress cost during a 'safe' phase.

    Phase 1 (0 to safe_start_frac):     hazards active
    Phase 2 (safe_start_frac to safe_end_frac): hazards suppressed (cost=0)
    Phase 3 (safe_end_frac to 1.0):     hazards return
    """

    def __init__(self, env, total_steps, safe_start_frac=0.3, safe_end_frac=0.7):
        super().__init__(env)
        self.total_training_steps = total_steps
        self.safe_start_step = int(total_steps * safe_start_frac)
        self.safe_end_step = int(total_steps * safe_end_frac)
        self.current_step = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1

        # Suppress cost during safe phase
        if self.safe_start_step <= self.current_step < self.safe_end_step:
            info['cost'] = 0.0
            info['hazards_active'] = False
        else:
            info['hazards_active'] = True

        return obs, reward, done, info
