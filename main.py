#!/usr/bin/env python3
"""
PGR Safety Experiment — Multi-Seed Runner
==========================================

Runs all methods across multiple random seeds and generates publication-ready
plots with shaded standard deviation bands.

Usage:
  python main.py --env cheetah                          # 3 seeds, all methods
  python main.py --env cheetah --seeds 42 123           # custom seeds
  python main.py --env ant --episodes 3000              # longer run
  python main.py --env cheetah --methods SAC PGR+Memory # subset of methods
"""

import argparse
import json
import random
import time

import torch

import numpy as np
import torch

from config import DEVICE, N_EPISODES, MAX_STEPS, UPDATES_PER_EPISODE
from envs import HazardHalfCheetah, HazardAnt, PointHazardEnv
from agents import SACAgent, SACPGRAgent, SACPGRMemoryAgent, SACMemoryAgent
from train import train_agent, plot_dashboard, aggregate_multiseed_for_dashboard, plot_results

ENV_REGISTRY = {
    "point": PointHazardEnv,
    "cheetah": HazardHalfCheetah,
    "ant": HazardAnt,
}

AGENT_REGISTRY = {
    "SAC": SACAgent,
    "SAC+Memory": SACMemoryAgent,       # ablation baseline
    "PGR": SACPGRAgent,
    "PGR+Memory": SACPGRMemoryAgent,    # our contribution
}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def plot_multiseed(results: dict, save_path: str):
    """Generate publication plot with mean +/- std shading across seeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    colors = {
        "SAC": "#ff7f0e",
        "SAC+Memory": "#d62728",
        "PGR": "#1f77b4",
        "PGR+Memory": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def smooth(data, window=50):
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window) / window, mode="valid")

    for name, seed_runs in results.items():
        if not seed_runs:
            continue
        color = colors.get(name, "#333333")

        # Smooth each seed's data
        sm_rewards = [smooth(run["rewards"]) for run in seed_runs.values()]
        sm_costs = [smooth(run["costs"]) for run in seed_runs.values()]
        cum_costs = [np.cumsum(run["costs"]) for run in seed_runs.values()]

        # Align lengths (seeds might differ slightly)
        min_r = min(len(s) for s in sm_rewards)
        min_c = min(len(s) for s in sm_costs)
        min_cum = min(len(s) for s in cum_costs)

        sm_rewards = np.array([s[:min_r] for s in sm_rewards])
        sm_costs = np.array([s[:min_c] for s in sm_costs])
        cum_costs = np.array([s[:min_cum] for s in cum_costs])

        # Mean +/- std
        for ax, data, title in [
            (axes[0], sm_rewards, "Episode Reward (higher = better)"),
            (axes[1], sm_costs, "Episode Cost (lower = safer)"),
            (axes[2], cum_costs, "Cumulative Cost (lower = safer)"),
        ]:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, label=name, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\nPlot saved -> {save_path}")


def print_multiseed_summary(results: dict, n_episodes: int):
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY (mean +/- std)")
    print("=" * 80)
    print(f"{'Method':<15} {'Avg Reward':<18} {'Total Cost':<18} {'Late Cost':<18}")
    print("-" * 70)

    for name, seed_runs in results.items():
        if not seed_runs:
            continue
        avg_rews = [np.mean(r["rewards"]) for r in seed_runs.values()]
        tot_costs = [r["total_cost"] for r in seed_runs.values()]
        late_costs = [sum(r["costs"][-n_episodes // 4:]) for r in seed_runs.values()]

        print(
            f"{name:<15} "
            f"{np.mean(avg_rews):>7.1f} +/- {np.std(avg_rews):<6.1f}  "
            f"{np.mean(tot_costs):>7.0f} +/- {np.std(tot_costs):<6.0f}  "
            f"{np.mean(late_costs):>7.0f} +/- {np.std(late_costs):<6.0f}"
        )


def run_experiment(env_name, n_episodes, methods, seeds):
    EnvClass = ENV_REGISTRY[env_name]

    print("=" * 80)
    print(f"EXPERIMENT: {env_name.upper()}")
    print(f"Device:   {DEVICE}")
    print(f"Episodes: {n_episodes}")
    print(f"Methods:  {methods}")
    print(f"Seeds:    {seeds}")
    print("=" * 80)

    # results[method][seed] = {rewards, costs, total_cost}
    results = {m: {} for m in methods}

    # Try to resume from saved progress
    json_path = f"results_{env_name}.json"
    try:
        with open(json_path) as f:
            saved = json.load(f)
        for m in methods:
            if m in saved:
                for seed_str, data in saved[m].items():
                    results[m][int(seed_str)] = data
        n_resumed = sum(len(v) for v in results.values())
        if n_resumed > 0:
            print(f"Resumed {n_resumed} completed runs from {json_path}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    total_runs = len(methods) * len(seeds)
    done_runs = sum(len(v) for v in results.values())

    for seed in seeds:
        for name in methods:
            # Skip if already completed
            if seed in results[name]:
                print(f"\n[SKIP] {name} seed={seed} (already done)")
                continue

            done_runs += 1
            print(f"\n{'=' * 80}")
            print(f"[{done_runs}/{total_runs}] {name}  seed={seed}")
            print(f"{'=' * 80}")

            seed_everything(seed)
            env = EnvClass()
            agent = AGENT_REGISTRY[name](env.state_dim, env.action_dim)

            rewards, costs = train_agent(
                agent, env, n_episodes, f"{name[:10]}-s{seed}",
                updates_per_episode=UPDATES_PER_EPISODE,
                max_steps=MAX_STEPS,
            )

            results[name][seed] = {
                "rewards": rewards,
                "costs": costs,
                "total_cost": sum(costs),
            }
            env.close()

            print(f"-> {name} seed={seed}: Reward={np.mean(rewards):.1f}, Cost={sum(costs):.0f}")

            # Save model weights for later visualisation with animate.py
            weight_file = f"weights_{env_name}_{name.replace('+', '_')}_{seed}.pt"
            torch.save({
                "policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
            }, weight_file)
            print(f"   Weights saved -> {weight_file}")

            # Save progress after every run (crash-safe)
            with open(json_path, "w") as f:
                json.dump(results, f)
            print(f"   Progress saved -> {json_path}")

    print_multiseed_summary(results, n_episodes)
    plot_multiseed(results, save_path=f"publication_{env_name}.png")
    dashboard_results = aggregate_multiseed_for_dashboard(results)
    plot_dashboard(dashboard_results, save_path=f"dashboard_{env_name}.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="PGR Safety Multi-Seed Experiment")
    parser.add_argument(
        "--env", choices=list(ENV_REGISTRY.keys()), default="cheetah",
    )
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument(
        "--methods", nargs="+", choices=list(AGENT_REGISTRY.keys()),
        default=list(AGENT_REGISTRY.keys()),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
    )
    args = parser.parse_args()

    run_experiment(args.env, args.episodes, args.methods, args.seeds)


if __name__ == "__main__":
    main()