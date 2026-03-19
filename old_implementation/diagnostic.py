#!/usr/bin/env python3
"""
Quick diagnostic run: single seed, fine-grained learning curves, saves weights.
Run: python diagnostic.py --env cheetah --episodes 5000
"""

import argparse
import json
import random
import time
import numpy as np
import torch
from collections import deque

from config import DEVICE, MAX_STEPS, UPDATES_PER_EPISODE
from envs import HazardHalfCheetah, HazardAnt, PointHazardEnv
from agents import SACAgent, SACPGRAgent, SACPGRMemoryAgent, SACMemoryAgent

ENV_REGISTRY = {"point": PointHazardEnv, "cheetah": HazardHalfCheetah, "ant": HazardAnt}
AGENT_REGISTRY = {"SAC": SACAgent, "SAC+Memory": SACMemoryAgent, "PGR": SACPGRAgent, "PGR+Memory": SACPGRMemoryAgent}

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def run_diagnostic(env_name, n_episodes, methods, seed):
    EnvClass = ENV_REGISTRY[env_name]

    print("=" * 80)
    print(f"DIAGNOSTIC: {env_name.upper()}  seed={seed}  episodes={n_episodes}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # Store per-episode data for every method
    all_results = {}

    for name in methods:
        seed_everything(seed)
        env = EnvClass()
        agent = AGENT_REGISTRY[name](env.state_dim, env.action_dim)

        rewards, costs, cum_costs = [], [], []
        cost_window = deque(maxlen=50)
        running_cost = 0
        t0 = time.time()

        print(f"\n{'─' * 80}")
        print(f"Training {name}")
        print(f"{'─' * 80}")

        for ep in range(n_episodes):
            state = env.reset()
            ep_reward, ep_cost = 0.0, 0.0

            for _ in range(MAX_STEPS):
                action = agent.select_action(state)
                next_state, reward, cost, done, _ = env.step(action)
                agent.add_transition(state, action, reward, cost, next_state, done)
                ep_reward += reward
                ep_cost += cost
                state = next_state
                if done:
                    break

            rewards.append(ep_reward)
            costs.append(ep_cost)
            running_cost += ep_cost
            cum_costs.append(running_cost)
            cost_window.append(ep_cost)
            agent.record_episode_cost(ep_cost)

            for _ in range(UPDATES_PER_EPISODE):
                agent.train_step()

            if (ep + 1) % 25 == 0:
                elapsed = time.time() - t0
                lam_str = f"  lam={agent.lam:.2f}" if hasattr(agent, 'lam') else ""
                print(f"[{name:15s}] Ep {ep+1:5d}/{n_episodes}  "
                      f"R={np.mean(rewards[-25:]):7.1f}  "
                      f"C(50)={sum(cost_window):6.1f}  "
                      f"CumC={running_cost:8.1f}"
                      f"{lam_str}  {elapsed:.0f}s")

        all_results[name] = {
            "rewards": rewards,
            "costs": costs,
            "cum_costs": cum_costs,
            "total_cost": sum(costs),
        }

        # Save weights for animation
        weight_path = f"weights_{env_name}_{name.replace('+', '_')}_{seed}.pt"
        torch.save({
            "actor": agent.policy.state_dict(),
            "critic": agent.q1.state_dict(),
        }, weight_path)
        print(f"Weights saved -> {weight_path}")

        env.close()

    # Save data
    json_path = f"diagnostic_{env_name}_{seed}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f)
    print(f"\nData saved -> {json_path}")

    # Plot learning curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"SAC": "#ff7f0e", "SAC+Memory": "#d62728", "PGR": "#1f77b4", "PGR+Memory": "#2ca02c"}
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        def smooth(data, w=50):
            if len(data) < w:
                return np.array(data)
            return np.convolve(data, np.ones(w)/w, mode='valid')

        for name, res in all_results.items():
            c = colors.get(name, "#333")

            # Smoothed reward
            sr = smooth(res["rewards"])
            axes[0, 0].plot(sr, label=name, color=c, linewidth=2)

            # Smoothed cost
            sc = smooth(res["costs"])
            axes[0, 1].plot(sc, label=name, color=c, linewidth=2)

            # Cumulative cost
            axes[1, 0].plot(res["cum_costs"], label=name, color=c, linewidth=2)

            # Cost rate (cumulative cost / episode number)
            eps = np.arange(1, len(res["cum_costs"]) + 1)
            axes[1, 1].plot(eps, np.array(res["cum_costs"]) / eps, label=name, color=c, linewidth=2)

        axes[0, 0].set_title("Episode Reward (smoothed)")
        axes[0, 1].set_title("Episode Cost (smoothed)")
        axes[1, 0].set_title("Cumulative Cost")
        axes[1, 1].set_title("Average Cost Rate (cum_cost / episode)")

        for ax in axes.flat:
            ax.set_xlabel("Episode")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Diagnostic: {env_name} (seed={seed}, {n_episodes} episodes)", fontsize=14)
        plt.tight_layout()
        plot_path = f"diagnostic_{env_name}_{seed}.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Plot saved -> {plot_path}")

    except ImportError:
        print("matplotlib not found, skipping plot")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Method':<15} {'Avg Reward':>12} {'Total Cost':>12} {'Late Cost':>12}")
    print("-" * 55)
    for name, res in all_results.items():
        late = sum(res["costs"][-n_episodes // 4:])
        print(f"{name:<15} {np.mean(res['rewards']):>12.1f} {res['total_cost']:>12.0f} {late:>12.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_REGISTRY.keys()), default="cheetah")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--methods", nargs="+", choices=list(AGENT_REGISTRY.keys()), default=list(AGENT_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_diagnostic(args.env, args.episodes, args.methods, args.seed)