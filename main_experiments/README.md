# Safety-Aware Prioritized Generative Replay (PGR)

This directory contains the code, results, and paper draft for our safety extension of [Prioritized Generative Replay](https://github.com/renwang435/pgr) (Wang et al., ICLR 2025).

## Summary

PGR uses a conditional diffusion model to generate synthetic replay transitions, dramatically accelerating RL training. We show this acceleration is a **safety liability**: on a velocity-constrained locomotion task, PGR incurs **185× more constraint violations** than standard SAC.

We introduce two mechanisms:
1. **Lagrangian reward penalty** — makes the policy cost-aware (`r_eff = r - λc`)
2. **Rare-event memory buffer** — preserves hazardous transitions for diffusion model training

The Lagrangian alone reduces violations by 98.6%, but the diffusion model loses hazard awareness (DiffHz collapses from 13.3% → 0.6%). The rare buffer maintains DiffHz at 8.6%, enables 4× lower penalty, and achieves **99.4% total cost reduction** with only 12.7% reward trade-off.

## Results (3 seeds, 100K steps, DMC Cheetah-Run)

| Method | Reward | Episode Cost | DiffHz | λ |
|--------|--------|-------------|--------|---|
| SAC | 204 ± 17 | 0.0 | N/A | - |
| PGR | 613 ± 35 | 374 ± 44 | 13.3% | - |
| PGR+Lagrangian | 548 ± 7 | 5.2 ± 1.1 | 0.6% | 3.08 |
| **PGR+Memory (ours)** | **535 ± 10** | **2.1 ± 0.0** | **8.6%** | **0.80** |

Statistical significance (Welch's t-test): PGR vs PGR+Memory cost p=0.007; DiffHz Lagrangian vs Memory p<0.001.

## Directory Structure

```
main_experiments/
├── README.md              ← you are here
├── safety/                ← our extension code (drops into the PGR repo)
│   ├── online_cost_cond.py    — main training script
│   ├── cost_agent.py          — cost-aware PGR agent with Lagrangian + rare buffer
│   ├── cost_replay_buffer.py  — CostReplayBuffer + RareEventBuffer
│   ├── cost_utils.py          — flat transition vectors with cost dimension
│   ├── hazard_wrapper.py      — velocity constraint wrapper (cost=1 when |v|>7.0)
│   ├── make_figures.py        — generates publication figures + summary table
│   ├── colab_setup.py         — installs all dependencies on Colab
│   ├── run_experiments.sh     — runs all 12 experiments
│   ├── run_pgr_safety.ipynb   — Colab notebook (recommended way to run)
│   └── README.md              — detailed code documentation
├── results/               ← experiment results (12 JSON files, 4 methods × 3 seeds)
├── figures/               ← generated figures (PDFs)
│   ├── figure1_curves.pdf     — reward + cumulative cost learning curves
│   ├── figure2_diffhz.pdf     — DiffHz trajectory over training
│   └── figure3_summary.pdf    — bar chart: reward vs cost by method
└── paper/
    └── main.tex           ← workshop paper draft
```

## How to Reproduce

### Prerequisites

This code extends the [PGR repository](https://github.com/renwang435/pgr) by Wang et al. You need to clone PGR first, then copy our `safety/` folder into it.

### Option A: Google Colab (recommended)

1. Upload `safety/` to Google Drive at `MyDrive/pgr/safety/`
2. Open `safety/run_pgr_safety.ipynb` in Colab
3. Set runtime to **GPU (A100 preferred)**
4. Run cells in order — the notebook clones PGR, installs dependencies, and runs all experiments
5. Results save to Google Drive automatically (survives runtime disconnects)

### Option B: Local / cluster

```bash
# 1. Clone PGR with submodules
git clone --recursive https://github.com/renwang435/pgr.git
cd pgr

# 2. Copy our safety extension into the PGR repo
cp -r /path/to/main_experiments/safety/ ./safety/

# 3. Install dependencies
pip install torch numpy accelerate einops ema-pytorch tqdm gin-config \
    mujoco dm-control gymnasium[mujoco] dm-env dm-tree dmcgym
pip install -e synther/REDQ

# 4. Run all experiments (4 methods × 3 seeds = 12 runs)
bash safety/run_experiments.sh ./safety_results

# 5. Generate figures
python safety/make_figures.py --results_dir ./safety_results --output_dir ./figures
```

### Running individual experiments

```bash
# SAC baseline
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac.gin \
    --gin_params "redq_sac.disable_diffusion = True" \
    --sac_only --velocity_threshold 7.0

# PGR (no safety mechanism)
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --gin_params "redq_sac.cond_top_frac = 0.25" \
    --velocity_threshold 7.0

# PGR+Lagrangian (ablation: Lagrangian only, no rare buffer)
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --gin_params "redq_sac.cond_top_frac = 0.25" \
    --velocity_threshold 7.0 --use_lagrangian

# PGR+Memory (ours: Lagrangian + rare buffer)
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --gin_params "redq_sac.cond_top_frac = 0.25" \
    --velocity_threshold 7.0 --use_rare_buffer --use_lagrangian
```

Each run: ~30–90 min on A100, 100K environment steps (~100 episodes).

## Regenerating Figures from Existing Results

```bash
python safety/make_figures.py --results_dir results/ --output_dir figures/
```

## Base Repository

PGR: [github.com/renwang435/pgr](https://github.com/renwang435/pgr) — Wang et al., "Prioritized Generative Replay", ICLR 2025 ([arXiv:2410.18082v2](https://arxiv.org/abs/2410.18082v2))
