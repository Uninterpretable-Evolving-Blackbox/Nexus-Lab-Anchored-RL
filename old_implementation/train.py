"""
Training loop, evaluation, and visualization.

Contains:
  - train_agent()    — runs one agent on one environment for N episodes
  - plot_results()   — generates a 2×2 summary figure comparing methods
  - print_summary()  — text table of results + relative cost comparisons
"""

import time
from collections import deque

import numpy as np


def train_agent(agent, env, n_episodes: int, label: str, updates_per_episode: int, max_steps: int):
    """
    Train a single agent on a single environment.

    Each episode:
      1. Reset the environment
      2. Roll out the policy for up to max_steps (or until done)
      3. After the episode, do `updates_per_episode` gradient steps

    This is the standard "collect then update" loop for off-policy RL.
    All agents (SAC, PGR, PGR+Memory) use the same loop — the difference
    is what happens inside agent.train_step().

    Args:
        agent:    one of SACAgent / SACPGRAgent / SACPGRMemoryAgent
        env:      environment with (obs, reward, cost, done, info) API
        n_episodes: total episodes to train
        label:    string label for logging (e.g. "SAC", "PGR")
        updates_per_episode: gradient steps per episode
        max_steps: max env steps per episode

    Returns:
        (rewards_list, costs_list) — per-episode totals
    """
    rewards = []
    costs = []
    cost_window = deque(maxlen=50)  # rolling window for recent cost tracking
    t0 = time.time()

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0

        # ── Roll out one episode ─────────────────────────────────────────
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, cost, done, _ = env.step(action)

            # Store the transition (the agent decides which buffers to use)
            agent.add_transition(state, action, reward, cost, next_state, done)

            ep_reward += reward
            ep_cost += cost
            state = next_state
            if done:
                break

        rewards.append(ep_reward)
        costs.append(ep_cost)
        cost_window.append(ep_cost)

        # ── Update Lagrange multiplier based on episode cost ───────────
        # This tells the agent how much cost it incurred, so it can
        # auto-tune λ to penalize hazards appropriately.
        agent.record_episode_cost(ep_cost)

        # ── Gradient updates (off-policy: we can update multiple times) ──
        for _ in range(updates_per_episode):
            agent.train_step()

        # ── Logging every 25 episodes ────────────────────────────────────
        if (ep + 1) % 25 == 0:
            elapsed = time.time() - t0
            # Show current λ value so we can see the constraint adapting
            lam_str = f"  λ={agent.lam:.2f}" if hasattr(agent, 'lam') else ""
            print(
                f"[{label:12s}] Ep {ep+1:4d}/{n_episodes}  "
                f"Reward: {np.mean(rewards[-25:]):7.1f}  "
                f"Cost(50): {sum(cost_window):6.1f}  "
                f"Total: {sum(costs):7.1f}{lam_str}  "
                f"Time: {elapsed:.0f}s"
            )

    return rewards, costs


def plot_results(results: dict, save_path: str = "hazard_cheetah_experiment.png"):
    """
    Generate a 2×2 figure comparing all methods:
      - Top left:     smoothed rewards over episodes
      - Top right:    smoothed costs over episodes (lower = safer)
      - Bottom left:  cumulative cost (lower = safer overall)
      - Bottom right: bar chart of total cost vs avg reward

    Args:
        results: dict mapping method_name → {"rewards": [...], "costs": [...], ...}
        save_path: where to save the PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (works on servers)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    # Color scheme for each method
    colors = {
        "SAC": "#ff7f0e", "PGR": "#1f77b4", "PGR+Memory": "#2ca02c",
    }
    # Fallback colors for any custom method names
    default_colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for i, name in enumerate(results):
        if name not in colors:
            colors[name] = default_colors[i % len(default_colors)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top left: smoothed rewards ───────────────────────────────────────
    ax = axes[0, 0]
    for name, data in results.items():
        # Moving average with window of 25 episodes
        sm = np.convolve(data["rewards"], np.ones(25) / 25, mode="valid")
        ax.plot(sm, label=name, color=colors.get(name, None), linewidth=2)
    ax.set_title("Episode Rewards (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Top right: smoothed costs ────────────────────────────────────────
    ax = axes[0, 1]
    for name, data in results.items():
        sm = np.convolve(data["costs"], np.ones(25) / 25, mode="valid")
        ax.plot(sm, label=name, color=colors.get(name, None), linewidth=2)
    ax.set_title("Episode Costs (smoothed) — LOWER IS SAFER")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost (hazard hits)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom left: cumulative cost ─────────────────────────────────────
    ax = axes[1, 0]
    for name, data in results.items():
        ax.plot(np.cumsum(data["costs"]), label=name, color=colors.get(name, None), linewidth=2)
    ax.set_title("Cumulative Cost — LOWER IS SAFER")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom right: bar chart summary ──────────────────────────────────
    ax = axes[1, 1]
    names = list(results.keys())
    x = np.arange(len(names))
    total_costs = [results[n]["total_cost"] for n in names]
    avg_rewards = [np.mean(results[n]["rewards"]) for n in names]

    # Two y-axes: cost (red, left) and reward (green, right)
    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, total_costs, 0.4, label="Total Cost", color="red", alpha=0.7)
    bars2 = ax2.bar(x + 0.2, avg_rewards, 0.4, label="Avg Reward", color="green", alpha=0.7)
    ax.set_ylabel("Total Cost (↓ safer)", color="red")
    ax2.set_ylabel("Avg Reward (↑ better)", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Safety vs Performance")

    # Label the bars with values
    for bar, val in zip(bars1, total_costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.0f}", ha="center", va="bottom", fontsize=10, color="red",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")


def print_summary(results: dict, n_episodes: int):
    """
    Print a text summary table comparing all methods.

    Shows: average reward, total cost, early cost (first 25% of episodes),
    and late cost (last 25%). Early vs late cost reveals whether the agent
    learns to avoid hazards over time.

    Also computes relative cost change between methods (the key metric
    for evaluating our contribution).
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Method':<14} {'Avg Reward':>12} {'Total Cost':>12} {'Early Cost':>12} {'Late Cost':>12}"
    print(header)
    print("-" * 80)

    for name, data in results.items():
        avg_reward = np.mean(data["rewards"])
        total_cost = data["total_cost"]
        # First and last quarter of training
        early_cost = sum(data["costs"][: n_episodes // 4])
        late_cost = sum(data["costs"][-n_episodes // 4 :])
        print(f"{name:<14} {avg_reward:>12.1f} {total_cost:>12.0f} {early_cost:>12.0f} {late_cost:>12.0f}")

    # ── Relative comparisons ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS — Lower cost = Safer")
    print("=" * 80)

    def pct(a, b):
        """Percent change from b to a."""
        return (a - b) / max(b, 1) * 100

    names = list(results.keys())
    if len(names) >= 2:
        base = names[0]  # compare everything against the first method
        base_cost = results[base]["total_cost"]
        for name in names[1:]:
            cost = results[name]["total_cost"]
            print(f"\n{name} vs {base}: {pct(cost, base_cost):+.1f}% cost change")
    if "PGR+Memory" in results and "PGR" in results:
        print(
            f"PGR+Memory vs PGR: "
            f"{pct(results['PGR+Memory']['total_cost'], results['PGR']['total_cost']):+.1f}% cost change"
        )
