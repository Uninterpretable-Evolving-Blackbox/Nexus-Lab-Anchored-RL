# Safety-Aware Prioritized Generative Replay

Safety extension for PGR (Wang et al., ICLR 2025). Adds constraint-aware diffusion replay to prevent PGR from amplifying unsafe behavior.

## Key Finding

PGR's sample efficiency is a safety liability: it incurs **185× more constraint violations** than SAC on velocity-constrained Cheetah-Run. Our rare-event memory buffer + Lagrangian penalty achieves **99.4% cost reduction** with only 12.7% reward trade-off.

## Methods

| Method | Description | Flags |
|--------|-------------|-------|
| SAC | REDQ-SAC baseline, no diffusion | `--sac_only` |
| PGR | Original PGR, tracks cost but ignores it | (default) |
| PGR+Lagrangian | PGR with reward penalty r - λc | `--use_lagrangian` |
| PGR+Memory (ours) | PGR + Lagrangian + rare-event buffer | `--use_rare_buffer --use_lagrangian` |

## Results (3 seeds, 100K steps, DMC Cheetah-Run)

| Method | Reward | Episode Cost | DiffHz | λ |
|--------|--------|-------------|--------|---|
| SAC | 204 ± 17 | 0.0 | N/A | - |
| PGR | 613 ± 35 | 374 ± 44 | 13.3% | - |
| PGR+Lagrangian | 548 ± 7 | 5.2 ± 1.1 | 0.6% | 3.08 |
| **PGR+Memory** | **535 ± 10** | **2.1 ± 0.0** | **8.6%** | **0.80** |

## Files

| File | Description |
|------|-------------|
| `online_cost_cond.py` | Main training script (extends `synther/online/online_cond.py`) |
| `cost_agent.py` | CostREDQRLPDCondAgent — cost-aware replay, rare buffer, Lagrangian λ |
| `cost_replay_buffer.py` | CostReplayBuffer + RareEventBuffer |
| `cost_utils.py` | Flat transition vectors with cost dimension |
| `hazard_wrapper.py` | HazardWrapper (cost=1 when velocity > threshold) |
| `make_figures.py` | Publication figures and summary table |
| `colab_setup.py` | Dependency installation for Google Colab |
| `run_experiments.sh` | Run all 12 experiments (4 methods × 3 seeds) |
| `run_pgr_safety.ipynb` | Colab notebook for running experiments |

## Quick Start (Colab)

1. Upload `safety/` to Google Drive at `MyDrive/pgr/safety/`
2. Open `run_pgr_safety.ipynb` in Colab with A100 GPU
3. Run cells in order — results save to Google Drive automatically

## Running Individually

```bash
# From the pgr/ repo root:

# SAC baseline
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac.gin \
    --gin_params "redq_sac.disable_diffusion = True" \
    --sac_only --velocity_threshold 7.0

# PGR (tracks cost, no safety fix)
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --gin_params "redq_sac.cond_top_frac = 0.25" \
    --velocity_threshold 7.0

# PGR+Lagrangian (ablation)
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --gin_params "redq_sac.cond_top_frac = 0.25" \
    --velocity_threshold 7.0 --use_lagrangian

# PGR+Memory (ours)
python safety/online_cost_cond.py --env cheetah-run-v0 --seed 42 \
    --gin_config_files config/online/sac_cond_synther_dmc.gin \
    --gin_params "redq_sac.cond_top_frac = 0.25" \
    --velocity_threshold 7.0 --use_rare_buffer --use_lagrangian
```

## Generate Figures

```bash
python safety/make_figures.py --results_dir ./safety_results --output_dir ./figures
```

## Citation

Based on PGR: Wang et al., "Prioritized Generative Replay", ICLR 2025 (arXiv:2410.18082v2).
