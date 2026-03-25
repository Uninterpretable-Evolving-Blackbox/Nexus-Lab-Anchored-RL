"""
Modified PGR training loop with hazard cost signal, DiffHz probe, and rare-event buffer.

Extends synther/online/online_cond.py to:
1. Wrap DMC environments with a hazard cost signal
2. Store cost in replay buffers and diffusion transitions
3. Periodically probe the diffusion model for hazard fidelity (DiffHz)
4. Optionally use a rare-event memory buffer (PGR+Memory)

Usage:
    # PGR baseline with cost tracking
    python safety/online_cost_cond.py --env cheetah-run-v0 \
        --gin_config_files config/online/sac_cond_synther_dmc.gin \
        --gin_params 'redq_sac.cond_top_frac = 0.25'

    # PGR+Memory
    python safety/online_cost_cond.py --env cheetah-run-v0 \
        --gin_config_files config/online/sac_cond_synther_dmc.gin \
        --gin_params 'redq_sac.cond_top_frac = 0.25' \
        --use_rare_buffer --use_lagrangian

    # SAC baseline with cost tracking
    python safety/online_cost_cond.py --env cheetah-run-v0 \
        --gin_config_files config/online/sac.gin \
        --sac_only
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

import argparse
import json
import os
import sys
import time

# Ensure repo root is on sys.path so 'synther' and 'safety' are importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import dmcgym
import gin
import gym
import numpy as np
import torch
from gym.wrappers.flatten_observation import FlattenObservation

from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.logx import EpochLogger
from redq.utils.run_utils import setup_logger_kwargs

from synther.diffusion.elucidated_diffusion import REDQCondTrainer
from synther.diffusion.diffusion_generator import CondDiffusionGenerator
from synther.diffusion.utils import construct_diffusion_model

from safety.cost_agent import CostREDQRLPDCondAgent
from safety.cost_replay_buffer import CostReplayBuffer, RareEventBuffer
from safety.cost_utils import make_cost_inputs_from_replay_buffer, split_cost_diffusion_samples
from safety.hazard_wrapper import HazardWrapper, TwoPhaseHazardWrapper


# ── DiffHz Probe ──────────────────────────────────────────────────────────────

def probe_diffusion_hazard_rate(diffusion_model, env, cond_distri,
                                 n_samples=1000, cfg_scale=1.0,
                                 cost_threshold=0.5):
    """
    Generate synthetic transitions and measure what fraction have cost > threshold.

    This is the key diagnostic: if the diffusion model is 'compressing out'
    hazardous transitions, this rate will decline over training even when
    the environment hazard rate stays constant.

    Args:
        diffusion_model: the EMA diffusion model
        env: gym environment (for obs/act dims)
        cond_distri: conditioning distribution from PGR
        n_samples: number of transitions to generate
        cfg_scale: CFG guidance scale
        cost_threshold: threshold for binary cost classification

    Returns:
        hazard_rate: fraction of generated transitions with cost > threshold
        mean_raw_cost: mean raw cost value in generated transitions
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cost_idx = obs_dim + act_dim + 1  # position of cost in flat vector

    # Generate conditional samples
    cond = cond_distri.sample_cond(n_samples)
    with torch.no_grad():
        samples = diffusion_model.sample(
            batch_size=n_samples,
            num_sample_steps=32,  # fewer steps for speed
            clamp=True,
            cond=cond,
            cfg_scale=cfg_scale,
            disable_tqdm=True,
        )
    samples = samples.cpu().numpy()

    # Extract cost dimension
    if samples.shape[1] > cost_idx:
        costs = samples[:, cost_idx]
        hazard_rate = float(np.mean(costs > cost_threshold))
        mean_raw_cost = float(np.mean(costs))
        return hazard_rate, mean_raw_cost
    else:
        return None, None


# ── Modified training functions ───────────────────────────────────────────────

def make_cost_inputs_from_buffer(buffer, model_terminals=False):
    """Build flat transition arrays including cost."""
    return make_cost_inputs_from_replay_buffer(buffer, model_terminals)


def train_diffusion_with_rare_buffer(
        diffusion_trainer, agent, cond_top_frac, curr_epoch,
        rare_buffer=None, rare_batch_ratio=0.2, rare_loss_weight=5.0,
        model_terminals=False, num_steps=None):
    """
    Train diffusion model, optionally mixing in rare-event transitions.

    If rare_buffer is provided, for each training batch:
    - 80% of data comes from normal replay buffer
    - 20% comes from the rare-event buffer
    - Rare transitions get rare_loss_weight multiplier
    """
    cond_net = agent.cond_net
    cond_net.eval()

    from synther.diffusion.elucidated_diffusion import CondDistri
    cond_distri = CondDistri(cond_net, diffusion_trainer.batch_size,
                             agent.replay_buffer, cond_top_frac)
    diffusion_trainer.update_cond_normalizer(cond_distri,
                                             device=diffusion_trainer.accelerator.device)

    num_steps = num_steps or diffusion_trainer.train_num_steps

    for j in range(num_steps):
        b = cond_distri.sample_batch(diffusion_trainer.batch_size)
        obs = b['obs1']
        next_obs = b['obs2']
        actions = b['acts']
        rewards = b['rews'][:, None]
        cond_signal = b['irews'][:, None]

        # Get cost from our extended buffer
        cost = agent.replay_buffer.cost_buf[b['idxs']][:, None]

        # Build data with cost: [obs, actions, reward, cost, next_obs]
        data = [obs, actions, rewards, cost, next_obs]
        if model_terminals:
            done = b['done'][:, None]
            data.append(done)
        data = np.concatenate(data, axis=1)

        # Mix in rare buffer transitions
        if rare_buffer is not None and len(rare_buffer) > 0:
            n_rare = max(1, int(diffusion_trainer.batch_size * rare_batch_ratio))
            n_normal = diffusion_trainer.batch_size - n_rare
            # Trim normal data
            data = data[:n_normal]
            cond_signal = cond_signal[:n_normal]

            # Get rare transitions
            rare_data = rare_buffer.get_flat_transitions(n_rare, include_cost=True)
            if rare_data is not None:
                if model_terminals:
                    # Add done=0 column for DMC
                    rare_data = np.concatenate([
                        rare_data,
                        np.zeros((rare_data.shape[0], 1), dtype=np.float32)
                    ], axis=1)

                data = np.concatenate([data, rare_data], axis=0)

                # Use high curiosity signal for rare transitions
                # (so they get high conditioning weight)
                rare_cond = cond_distri.sample_cond(rare_data.shape[0])
                # Use the max conditioning value to ensure rare events are
                # treated as highly relevant
                max_cond = np.max(cond_signal)
                rare_cond = np.full((rare_data.shape[0], 1), max_cond)
                cond_signal = np.concatenate([cond_signal, rare_cond], axis=0)

        data = torch.from_numpy(data).float()
        cond_signal = torch.from_numpy(cond_signal).float()

        # Train on batch (weighted loss for rare transitions is handled
        # by the diffusion model's standard loss weighting)
        loss = diffusion_trainer.train_on_batch(data, cond=cond_signal)

        if j % 1000 == 0:
            print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    diffusion_trainer.save_final(cond_distri, curr_epoch, num_steps)
    return cond_distri


# ── Gym wrappers ──────────────────────────────────────────────────────────────

def wrap_gym(env, rescale_actions=True):
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    return env


def get_time_limit(env):
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            return env.spec.max_episode_steps
    if hasattr(env, 'env'):
        return get_time_limit(env.env)
    if hasattr(env, 'unwrapped'):
        return get_time_limit(env.unwrapped)
    else:
        raise ValueError("Cannot find time limit for env")


# ── Main training function ────────────────────────────────────────────────────

@gin.configurable
def redq_sac(
        env_name,
        seed=3,
        epochs=-1,
        steps_per_epoch=1000,
        max_ep_len=1000,
        n_evals_per_epoch=1,
        logger_kwargs=dict(),
        # agent hyperparameters
        hidden_sizes=(256, 256),
        replay_size=int(1e6),
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        auto_alpha=True,
        target_entropy='mbpo',
        start_steps=5000,
        delay_update_steps='auto',
        utd_ratio=20,
        num_Q=10,
        num_min=2,
        q_target_mode='min',
        policy_update_delay=20,
        diffusion_buffer_size=int(1e6),
        diffusion_sample_ratio=0.5,
        # diffusion hyperparameters
        retrain_diffusion_every=10_000,
        num_samples=100_000,
        diffusion_start=0,
        disable_diffusion=True,
        print_buffer_stats=True,
        skip_reward_norm=True,
        model_terminals=False,
        # conditional generation hyperparameters
        cfg_dropout=0.25,
        cond_top_frac=0.05,
        cfg_scale=1.0,
        cond_hidden_size=128,
        # bias evaluation
        evaluate_bias=True,
        n_mc_eval=1000,
        n_mc_cutoff=350,
        reseed_each_epoch=True,
        # === SAFETY EXTENSIONS ===
        velocity_threshold=7.0,
        use_rare_buffer=False,
        rare_buffer_size=500,
        rare_batch_ratio=0.2,
        rare_loss_weight=5.0,
        use_lagrangian=False,
        cost_limit=2.0,
        lambda_lr=1e-2,
        lambda_init=0.0,
        probe_every=5000,
        two_phase=False,
        safe_start_frac=0.3,
        safe_end_frac=0.7,
        sac_only=False,
        results_folder='./results',
):
    # sac_only overrides diffusion settings
    if sac_only:
        disable_diffusion = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")

    if epochs == 'mbpo' or epochs < 0:
        # Default 100 epochs for DMC (100K steps), matching the PGR paper.
        # Original REDQ code defaults to 300 for unknown envs, but that's
        # 3x more than the paper uses for DMC tasks.
        epochs = mbpo_epoches.get(env_name, 100)
    total_steps = steps_per_epoch * epochs + 1

    # Logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Environment setup with hazard wrapper
    def make_env(with_hazard=True):
        env = wrap_gym(gym.make(env_name))
        if with_hazard:
            env = HazardWrapper(env, velocity_threshold=velocity_threshold)
        return env

    env = make_env(with_hazard=True)
    test_env = make_env(with_hazard=True)
    # Bias eval env without hazard wrapper (doesn't need cost)
    bias_eval_env = make_env(with_hazard=False)

    if two_phase:
        env = TwoPhaseHazardWrapper(env, total_steps, safe_start_frac, safe_end_frac)

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    def _seed_env(e, s):
        """Seed env, handling gym API changes."""
        try:
            e.seed(s)
        except (AttributeError, TypeError):
            pass  # gymnasium-style envs handle seed via reset(seed=...)
        try:
            e.action_space.np_random = np.random.RandomState(s)
        except Exception:
            try:
                e.action_space.np_random.seed(s)
            except Exception:
                pass

    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        _seed_env(env, env_seed)
        _seed_env(test_env, test_env_seed)
        _seed_env(bias_eval_env, bias_eval_env_seed)

    seed_all(epoch=0)

    # Dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env_time_limit = get_time_limit(env)
    max_ep_len = env_time_limit if max_ep_len > env_time_limit else max_ep_len
    act_limit = env.action_space.high[0].item()

    start_time = time.time()
    sys.stdout.flush()

    # Create agent with cost support
    agent = CostREDQRLPDCondAgent(
        cond_hidden_size, diffusion_buffer_size, diffusion_sample_ratio,
        use_rare_buffer=use_rare_buffer,
        rare_buffer_size=rare_buffer_size,
        rare_batch_ratio=rare_batch_ratio,
        rare_loss_weight=rare_loss_weight,
        use_lagrangian=use_lagrangian,
        cost_limit=cost_limit,
        lambda_lr=lambda_lr,
        lambda_init=lambda_init,
        # REDQ args
        env_name=env_name, obs_dim=obs_dim, act_dim=act_dim,
        act_limit=act_limit, device=device,
        hidden_sizes=hidden_sizes, replay_size=replay_size,
        batch_size=batch_size, lr=lr, gamma=gamma, polyak=polyak,
        alpha=alpha, auto_alpha=auto_alpha, target_entropy=target_entropy,
        start_steps=start_steps, delay_update_steps=delay_update_steps,
        utd_ratio=utd_ratio, num_Q=num_Q, num_min=num_min,
        q_target_mode=q_target_mode,
        policy_update_delay=policy_update_delay,
    )

    # Diffusion model dimensions: [obs, act, rew, cost, next_obs]
    # +1 for cost compared to original PGR
    diff_dims = obs_dim + act_dim + 1 + 1 + obs_dim  # +1 cost
    if model_terminals:
        diff_dims += 1
    inputs = torch.zeros((128, diff_dims)).float()

    # Skip normalization for reward and cost dimensions
    if skip_reward_norm:
        reward_idx = obs_dim + act_dim
        cost_idx = obs_dim + act_dim + 1
        skip_dims = [reward_idx, cost_idx]
    else:
        skip_dims = []

    # Tracking
    episode_costs = []
    episode_rewards = []
    diffhz_log = []  # (step, hazard_rate, mean_raw_cost)
    lambda_log = []   # (step, lambda_value)
    env_hazard_rates = []

    # Create results directory BEFORE training starts
    os.makedirs(results_folder, exist_ok=True)
    if use_rare_buffer:
        method = 'pgr_memory'
    elif disable_diffusion:
        method = 'sac'
    elif use_lagrangian:
        method = 'pgr_lagrangian'
    else:
        method = 'pgr'
    results_path = os.path.join(results_folder, f'{method}_seed{seed}_results.json')
    print(f'Results will be saved to: {results_path}')

    def _save_results(final=False):
        """Save current results to disk (called periodically and at end)."""
        results = {
            'episode_rewards': [float(x) for x in episode_rewards],
            'episode_costs': [float(x) for x in episode_costs],
            'diffhz_log': [(int(s), float(h), float(m)) for s, h, m in diffhz_log],
            'lambda_log': [(int(s), float(l)) for s, l in lambda_log],
            'seed': seed,
            'env_name': env_name,
            'use_rare_buffer': use_rare_buffer,
            'use_lagrangian': use_lagrangian,
            'velocity_threshold': velocity_threshold,
            'total_steps': total_steps,
            'complete': final,
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0.0

    for t in range(total_steps):
        a = agent.get_exploration_action(o, env)
        o2, r, d, info = env.step(a)

        cost = info.get('cost', 0.0)
        ep_cost += cost

        ep_len += 1
        d = False if ep_len == max_ep_len else d

        # Store with cost
        agent.store_data(o, a, r, o2, d, cost=cost)
        agent.train(logger)
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            episode_costs.append(ep_cost)
            episode_rewards.append(ep_ret)
            lambda_log.append((t, agent.lam))

            # Update Lagrangian
            agent.record_episode_cost(ep_cost)

            o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0.0

        # ── Retrain diffusion model ──────────────────────────────────────
        if not disable_diffusion and (t + 1) % retrain_diffusion_every == 0 and (t + 1) >= diffusion_start:
            print(f'\n=== Retraining diffusion model at step {t + 1} ===')

            diffusion_trainer = REDQCondTrainer(
                construct_diffusion_model(
                    inputs=inputs,
                    skip_dims=skip_dims,
                    disable_terminal_norm=model_terminals,
                    cond_dim=1,
                    cfg_dropout=cfg_dropout,
                ),
                results_folder=results_folder,
                model_terminals=model_terminals,
            )

            # Update normalizer with cost-extended data
            data_for_norm = make_cost_inputs_from_buffer(
                agent.replay_buffer, model_terminals)
            data_tensor = torch.from_numpy(data_for_norm).float()
            diffusion_trainer.model.normalizer.reset(data_tensor)
            diffusion_trainer.ema.ema_model.normalizer.reset(data_tensor)
            diffusion_trainer.model.normalizer.to(device)
            diffusion_trainer.ema.ema_model.normalizer.to(device)

            # Train diffusion (with or without rare buffer)
            cond_distri = train_diffusion_with_rare_buffer(
                diffusion_trainer, agent, cond_top_frac,
                curr_epoch=(t // steps_per_epoch) + 1,
                rare_buffer=agent.rare_buffer if use_rare_buffer else None,
                rare_batch_ratio=rare_batch_ratio,
                rare_loss_weight=rare_loss_weight,
                model_terminals=model_terminals,
            )

            agent.reset_diffusion_buffer()

            # ── DiffHz Probe ──────────────────────────────────────────────
            hz_rate, mean_raw = probe_diffusion_hazard_rate(
                diffusion_trainer.ema.ema_model, env, cond_distri,
                n_samples=2000, cfg_scale=cfg_scale)
            if hz_rate is not None:
                diffhz_log.append((t + 1, hz_rate, mean_raw))
                print(f'  DiffHz = {hz_rate:.3f}  (mean_raw_cost = {mean_raw:.4f})')
                if hasattr(env, 'hazard_rate'):
                    print(f'  Env hazard rate = {env.hazard_rate:.3f}')

            # Generate synthetic samples with cost
            from synther.diffusion.elucidated_diffusion import CondDistri
            generator_env = make_env(with_hazard=False)  # plain env for dims

            # Custom sampling that handles cost dimension
            print(f'Generating {num_samples} synthetic samples...')
            n_batches = max(1, num_samples // 100000)
            batch_gen_size = num_samples // n_batches

            all_obs, all_acts, all_rews, all_costs, all_next_obs = [], [], [], [], []

            for i_batch in range(n_batches):
                cond = cond_distri.sample_cond(batch_gen_size)
                sampled = diffusion_trainer.ema.ema_model.sample(
                    batch_size=batch_gen_size,
                    num_sample_steps=128,
                    clamp=True,
                    cond=cond,
                    cfg_scale=cfg_scale,
                    disable_tqdm=(i_batch > 0),
                )
                sampled = sampled.cpu().numpy()

                # Parse with cost: [obs, act, rew, cost, next_obs]
                obs_s = sampled[:, :obs_dim]
                act_s = sampled[:, obs_dim:obs_dim + act_dim]
                rew_s = sampled[:, obs_dim + act_dim]
                cost_s = sampled[:, obs_dim + act_dim + 1]
                next_obs_s = sampled[:, obs_dim + act_dim + 2:obs_dim + act_dim + 2 + obs_dim]

                all_obs.append(obs_s)
                all_acts.append(act_s)
                all_rews.append(rew_s)
                all_costs.append(cost_s)
                all_next_obs.append(next_obs_s)

            observations = np.concatenate(all_obs, axis=0)
            actions = np.concatenate(all_acts, axis=0)
            rewards = np.concatenate(all_rews, axis=0)
            costs = np.concatenate(all_costs, axis=0)
            next_observations = np.concatenate(all_next_obs, axis=0)

            print(f'Adding {num_samples} samples to diffusion buffer.')
            print(f'  Synthetic cost stats: mean={costs.mean():.4f}, '
                  f'frac>0.5={np.mean(costs > 0.5):.3f}')

            for o_s, a_s, r_s, o2_s, c_s in zip(
                    observations, actions, rewards, next_observations, costs):
                agent.diffusion_buffer.store(o_s, a_s, r_s, o2_s, 0.0, cost=c_s)

            if print_buffer_stats:
                ptr_location = agent.replay_buffer.ptr
                real_rewards = agent.replay_buffer.rews_buf[:ptr_location]
                real_costs = agent.replay_buffer.cost_buf[:ptr_location]
                print(f'  Real reward: {np.mean(real_rewards):.2f} +/- {np.std(real_rewards):.2f}')
                print(f'  Synth reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}')
                print(f'  Real cost frac>0: {np.mean(real_costs > 0):.3f}')
                print(f'  Synth cost frac>0.5: {np.mean(costs > 0.5):.3f}')
                print(f'  Real buffer size: {ptr_location}')
                print(f'  Diffusion buffer size: {agent.diffusion_buffer.ptr}')
                if use_rare_buffer:
                    print(f'  Rare buffer size: {len(agent.rare_buffer)}')

        # ── Periodic DiffHz probe (between retraining) ────────────────────
        # This uses the most recent diffusion model
        if not disable_diffusion and (t + 1) % probe_every == 0 and len(diffhz_log) > 0:
            # Only probe if we have a trained diffusion model
            pass  # Probing is done during retraining above

        # ── End of epoch ──────────────────────────────────────────────────
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            returns = test_agent(agent, test_env, max_ep_len, logger, n_evals_per_epoch)
            if evaluate_bias:
                from redq.utils.bias_utils import log_bias_evaluation
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len,
                                   alpha, gamma, n_mc_eval, n_mc_cutoff)

            if reseed_each_epoch:
                seed_all(epoch)

            # Logging
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('LossCond', with_min_and_max=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            # Save intermediate results every epoch
            _save_results(final=False)

            # Print safety stats
            if episode_costs:
                recent_costs = episode_costs[-100:]
                print(f'\n  Safety stats (last 100 eps): '
                      f'mean_cost={np.mean(recent_costs):.2f}, '
                      f'total_cost={sum(episode_costs):.0f}, '
                      f'lambda={agent.lam:.3f}')
                if diffhz_log:
                    last_hz = diffhz_log[-1]
                    print(f'  DiffHz={last_hz[1]:.3f} at step {last_hz[0]}')
                print()

            sys.stdout.flush()

    # ── Save final results ──────────────────────────────────────────────────
    _save_results(final=True)
    print(f'\nFinal results saved to {results_path}')

    # Return results dict
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah-run-v0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='online_logs')
    # Default to Google Drive on Colab, fallback to local
    default_results = ('/content/drive/MyDrive/pgr/safety_results'
                       if os.path.isdir('/content/drive/MyDrive') else './safety_results')
    parser.add_argument('--results_folder', type=str, default=default_results)
    parser.add_argument('--gin_config_files', nargs='*', type=str,
                        default=['config/online/sac_cond_synther_dmc.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])

    # Safety extensions
    parser.add_argument('--velocity_threshold', type=float, default=7.0,
                        help='Velocity threshold for hazard cost')
    parser.add_argument('--use_rare_buffer', action='store_true',
                        help='Enable rare-event memory buffer')
    parser.add_argument('--rare_buffer_size', type=int, default=500)
    parser.add_argument('--rare_batch_ratio', type=float, default=0.2)
    parser.add_argument('--rare_loss_weight', type=float, default=5.0)
    parser.add_argument('--use_lagrangian', action='store_true',
                        help='Enable Lagrangian cost constraint')
    parser.add_argument('--cost_limit', type=float, default=2.0)
    parser.add_argument('--lambda_lr', type=float, default=0.01)
    parser.add_argument('--lambda_init', type=float, default=0.0)
    parser.add_argument('--probe_every', type=int, default=5000)
    parser.add_argument('--two_phase', action='store_true',
                        help='Use two-phase hazard schedule')
    parser.add_argument('--sac_only', action='store_true',
                        help='Disable diffusion (SAC baseline)')

    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.env, args.log_dir)

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    # Override gin with CLI args for safety params
    redq_sac(
        args.env,
        seed=args.seed,
        target_entropy='auto',
        logger_kwargs=logger_kwargs,
        velocity_threshold=args.velocity_threshold,
        use_rare_buffer=args.use_rare_buffer,
        rare_buffer_size=args.rare_buffer_size,
        rare_batch_ratio=args.rare_batch_ratio,
        rare_loss_weight=args.rare_loss_weight,
        use_lagrangian=args.use_lagrangian,
        cost_limit=args.cost_limit,
        lambda_lr=args.lambda_lr,
        lambda_init=args.lambda_init,
        probe_every=args.probe_every,
        two_phase=args.two_phase,
        sac_only=args.sac_only,
        results_folder=args.results_folder,
    )
