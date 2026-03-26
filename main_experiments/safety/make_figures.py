"""
Generate publication figures from experiment results.

Reframed narrative: PGR optimizes reward aggressively but is safety-blind,
incurring massive constraint violations. PGR+Memory (rare-event buffer +
Lagrangian) achieves ~98% cost reduction with minimal reward trade-off.

Produces:
  Figure 1: Reward and cost learning curves (SAC, PGR, PGR+Memory)
  Figure 2: DiffHz diagnostic — synthetic hazard rate over training
  Figure 3: Cost-reward Pareto summary (bar chart)
  Table 1: Summary statistics
"""

import json
import os
import glob
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available")


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_results(results_dir, method, seeds=(42, 123, 456)):
    """Load results JSON files for a method across seeds."""
    all_results = []
    for seed in seeds:
        # Try flat structure first (new)
        flat_path = os.path.join(results_dir, f'{method}_seed{seed}_results.json')
        if os.path.exists(flat_path):
            with open(flat_path) as f:
                all_results.append(json.load(f))
            continue
        # Fallback to nested structure (old)
        pattern = os.path.join(results_dir, f'{method}_seed{seed}',
                               f'{method}_seed{seed}_results.json')
        files = glob.glob(pattern)
        if not files:
            pattern = os.path.join(results_dir, f'{method}_seed{seed}', '*_results.json')
            files = glob.glob(pattern)
        if files:
            with open(files[0]) as f:
                all_results.append(json.load(f))
    return all_results


def smooth(x, window=10):
    """Moving average smoothing."""
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode='valid')


def mean_std_across_seeds(list_of_arrays):
    """Compute mean and std across seeds, using max length with NaN padding."""
    max_len = max(len(a) for a in list_of_arrays)
    padded = np.full((len(list_of_arrays), max_len), np.nan)
    for i, a in enumerate(list_of_arrays):
        padded[i, :len(a)] = a
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    return mean, std


METHODS = [
    ('sac', 'SAC (no diffusion)', '#ff7f0e'),
    ('pgr', 'PGR', '#1f77b4'),
    ('pgr_lagrangian', 'PGR+Lagrangian', '#d62728'),
    ('pgr_memory', 'PGR+L+Buffer (ours)', '#2ca02c'),
]


# ── Figure 1: Learning Curves (main result) ─────────────────────────────────

def plot_learning_curves(results_dir, seeds=(42, 123, 456),
                         save_path='figure1_curves.pdf'):
    """Reward and cumulative cost curves — the main result figure."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for method, label, color in METHODS:
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            continue

        # Rewards (smoothed)
        reward_curves = [smooth(np.array(r['episode_rewards'])) for r in results_list]
        if reward_curves:
            mean_r, std_r = mean_std_across_seeds(reward_curves)
            x = np.arange(len(mean_r))
            axes[0].plot(x, mean_r, color=color, linewidth=2, label=label)
            axes[0].fill_between(x, mean_r - std_r, mean_r + std_r,
                                color=color, alpha=0.15)

        # Cumulative costs
        cost_curves = [np.cumsum(r['episode_costs']) for r in results_list]
        if cost_curves:
            mean_c, std_c = mean_std_across_seeds(cost_curves)
            x = np.arange(len(mean_c))
            axes[1].plot(x, mean_c, color=color, linewidth=2, label=label)
            axes[1].fill_between(x, mean_c - std_c, mean_c + std_c,
                                color=color, alpha=0.15)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('(a) Reward')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cumulative Cost')
    axes[1].set_title('(b) Safety Violations (lower = safer)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure 1 saved to {save_path}')
    plt.close()


# ── Figure 2: DiffHz Diagnostic ─────────────────────────────────────────────

def plot_diffhz(results_dir, seeds=(42, 123, 456),
                save_path='figure2_diffhz.pdf'):
    """DiffHz over training — diagnostic showing what the diffusion model captures."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for method, label, color in [METHODS[1], METHODS[2], METHODS[3]]:  # PGR, Lagrangian, L+Buffer
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            continue

        all_steps = []
        all_rates = []
        for r in results_list:
            if 'diffhz_log' in r and r['diffhz_log']:
                steps = [x[0] for x in r['diffhz_log']]
                rates = [x[1] for x in r['diffhz_log']]
                all_steps.append(steps)
                all_rates.append(rates)

        if not all_rates:
            continue

        # Individual seeds as thin lines
        for steps, rates in zip(all_steps, all_rates):
            ax.plot(steps, rates, color=color, alpha=0.2, linewidth=0.8)

        # Mean ± std (NaN-padded to max length)
        if len(all_rates) > 1:
            max_len = max(len(r) for r in all_rates)
            padded = np.full((len(all_rates), max_len), np.nan)
            for i, r in enumerate(all_rates):
                padded[i, :len(r)] = r
            mean_rates = np.nanmean(padded, axis=0)
            std_rates = np.nanstd(padded, axis=0)
            # Use longest step grid
            longest_idx = np.argmax([len(s) for s in all_steps])
            steps_common = all_steps[longest_idx][:max_len]
            ax.plot(steps_common, mean_rates, color=color, linewidth=2.5, label=label)
            ax.fill_between(steps_common,
                           np.maximum(mean_rates - std_rates, 0),
                           mean_rates + std_rates,
                           color=color, alpha=0.15)
        else:
            ax.plot(all_steps[0], all_rates[0], color=color, linewidth=2.5, label=label)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('DiffHz (fraction cost > 0.5)')
    ax.set_title('Diffusion Model Hazard Rate Over Training')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure 2 saved to {save_path}')
    plt.close()


# ── Figure 3: Cost-Reward Summary Bar Chart ─────────────────────────────────

def plot_summary_bars(results_dir, seeds=(42, 123, 456),
                      save_path='figure3_summary.pdf'):
    """Bar chart: reward vs cost trade-off across methods."""
    if not HAS_MPL:
        return

    labels = []
    mean_rewards, std_rewards = [], []
    mean_costs, std_costs = [], []

    for method, label, color in METHODS:
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            continue
        labels.append(label)
        rews = [np.mean(r['episode_rewards'][-50:]) for r in results_list]
        costs = [np.mean(r['episode_costs'][-50:]) for r in results_list]
        mean_rewards.append(np.mean(rews))
        std_rewards.append(np.std(rews))
        mean_costs.append(np.mean(costs))
        std_costs.append(np.std(costs))

    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c'][:len(labels)]
    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1, 1]})

    # Left: Reward bars
    bars1 = ax1.bar(x, mean_rewards, 0.6, yerr=std_rewards,
                    color=colors, alpha=0.85, edgecolor='black',
                    linewidth=0.5, capsize=5)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('(a) Reward (last 50 eps)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim(bottom=0)
    # Annotate values
    for bar, mr, sr in zip(bars1, mean_rewards, std_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sr + 10,
                f'{mr:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right: Cost bars
    bars2 = ax2.bar(x, mean_costs, 0.6, yerr=std_costs,
                    color=colors, alpha=0.85, edgecolor='black',
                    linewidth=0.5, capsize=5, hatch='//')
    ax2.set_ylabel('Episode Cost')
    ax2.set_title('(b) Cost (last 50 eps, lower = safer)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim(bottom=0)
    # Annotate values
    for bar, mc, sc in zip(bars2, mean_costs, std_costs):
        y_pos = bar.get_height() + sc + max(mean_costs) * 0.03
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{mc:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure 3 saved to {save_path}')
    plt.close()


# ── Table 1: Summary Statistics ──────────────────────────────────────────────

def print_summary_table(results_dir, seeds=(42, 123, 456)):
    """Print summary table with the reframed metrics."""

    print('\n' + '=' * 90)
    print('Table 1: Summary Statistics (mean +/- std across seeds, last 50 episodes)')
    print('=' * 90)
    print(f'{"Method":<22} {"Reward":>16} {"Ep Cost":>16} {"Total Cost":>14} {"DiffHz":>14}')
    print('-' * 90)

    for method, label, _ in METHODS:
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            print(f'{label:<22} {"N/A":>16} {"N/A":>16} {"N/A":>14} {"N/A":>14}')
            continue

        # Use last 50 episodes for final stats
        rewards = [np.mean(r['episode_rewards'][-50:]) for r in results_list]
        ep_costs = [np.mean(r['episode_costs'][-50:]) for r in results_list]
        total_costs = [sum(r['episode_costs']) for r in results_list]

        diffhz_vals = []
        for r in results_list:
            if 'diffhz_log' in r and r['diffhz_log']:
                diffhz_vals.append(r['diffhz_log'][-1][1])

        mean_r, std_r = np.mean(rewards), np.std(rewards)
        mean_ec, std_ec = np.mean(ep_costs), np.std(ep_costs)
        mean_tc, std_tc = np.mean(total_costs), np.std(total_costs)

        if diffhz_vals:
            mean_hz = np.mean(diffhz_vals)
            std_hz = np.std(diffhz_vals)
            hz_str = f'{mean_hz:.1%}+/-{std_hz:.1%}'
        else:
            hz_str = 'N/A'

        print(f'{label:<22} {mean_r:>7.1f}+/-{std_r:>5.1f} '
              f'{mean_ec:>7.1f}+/-{std_ec:>5.1f} '
              f'{mean_tc:>6.0f}+/-{std_tc:>5.0f} '
              f'{hz_str:>14}')

    print('=' * 90)

    # Cost reduction callout
    pgr = load_results(results_dir, 'pgr', seeds)
    mem = load_results(results_dir, 'pgr_memory', seeds)
    if pgr and mem:
        pgr_cost = np.mean([np.mean(r['episode_costs'][-50:]) for r in pgr])
        mem_cost = np.mean([np.mean(r['episode_costs'][-50:]) for r in mem])
        pgr_rew = np.mean([np.mean(r['episode_rewards'][-50:]) for r in pgr])
        mem_rew = np.mean([np.mean(r['episode_rewards'][-50:]) for r in mem])
        if pgr_cost > 0:
            reduction = (1 - mem_cost / pgr_cost) * 100
            rew_diff = (mem_rew - pgr_rew) / pgr_rew * 100
            print(f'\nPGR+Memory vs PGR: {reduction:.1f}% cost reduction, '
                  f'{rew_diff:+.1f}% reward change')
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./safety_results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--output_dir', type=str, default='./figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = tuple(args.seeds)

    print_summary_table(args.results_dir, seeds)

    if HAS_MPL:
        plot_learning_curves(args.results_dir, seeds,
                            os.path.join(args.output_dir, 'figure1_curves.pdf'))
        plot_diffhz(args.results_dir, seeds,
                    os.path.join(args.output_dir, 'figure2_diffhz.pdf'))
        plot_summary_bars(args.results_dir, seeds,
                         os.path.join(args.output_dir, 'figure3_summary.pdf'))
    else:
        print("Install matplotlib for figures: pip install matplotlib")
