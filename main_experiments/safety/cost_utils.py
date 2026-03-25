"""
Utilities for cost-extended transitions.

Modified versions of synther/online/utils.py and synther/diffusion/utils.py
that include cost as an extra dimension in the flat transition vector.

Transition format: [obs, actions, rewards, cost, next_obs]
                    (cost is inserted between reward and next_obs)
"""

import gin
import gym
import numpy as np
import torch
from typing import List, Optional, Union

from safety.cost_replay_buffer import CostReplayBuffer


def make_cost_inputs_from_replay_buffer(
        replay_buffer: CostReplayBuffer,
        model_terminals: bool = False,
) -> np.ndarray:
    """
    Build flat transition arrays from CostReplayBuffer.
    Format: [obs, actions, reward, cost, next_obs]
    """
    ptr_location = replay_buffer.ptr if replay_buffer.ptr > 0 else replay_buffer.size
    obs = replay_buffer.obs1_buf[:ptr_location]
    actions = replay_buffer.acts_buf[:ptr_location]
    next_obs = replay_buffer.obs2_buf[:ptr_location]
    rewards = replay_buffer.rews_buf[:ptr_location]
    costs = replay_buffer.cost_buf[:ptr_location]
    inputs = [obs, actions, rewards[:, None], costs[:, None], next_obs]
    if model_terminals:
        terminals = replay_buffer.done_buf[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None])
    return np.concatenate(inputs, axis=1)


def split_cost_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    """
    Split flat diffusion samples back into (obs, actions, rewards, costs, next_obs).
    Format: [obs, actions, reward, cost, next_obs]
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    costs = samples[:, obs_dim + action_dim + 1]
    next_obs = samples[:, obs_dim + action_dim + 2: obs_dim + action_dim + 2 + obs_dim]

    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, costs, next_obs, terminals
    else:
        return obs, actions, rewards, costs, next_obs
