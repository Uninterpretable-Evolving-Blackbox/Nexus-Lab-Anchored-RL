"""
Hyperparameters and global config for PGR Safety experiments.

All tuneable values live here so you can sweep them from one place.
"""


import random, numpy as np, torch

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
# Automatically use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ── Environment ──────────────────────────────────────────────────────────────
MAX_STEPS = 200         # max steps per episode (point env is small, 200 is plenty)
N_EPISODES = 200       # total training episodes

# ── Networks ─────────────────────────────────────────────────────────────────
HIDDEN_DIM = 128        # width of hidden layers (128 is enough for 6D state)
LR = 3e-4              # learning rate for all optimizers (Adam)
GAMMA = 0.99           # discount factor — how much we value future reward
TAU = 0.005            # soft-update rate for target Q-networks
                       # (each step: target = TAU*online + (1-TAU)*target)

# ── Replay buffer ────────────────────────────────────────────────────────────
BATCH_SIZE = 64        # transitions per gradient step
BUFFER_SIZE = 30_000    # max transitions stored in the main replay buffer

# ── PGR / Diffusion ─────────────────────────────────────────────────────────
DIFFUSION_STEPS = 50    # number of denoising steps T
                        # With T=10 and max_beta=0.02, alpha_bar only reaches 0.90
                        # meaning data is still 90% clean at the "noisiest" step.
                        # But generation starts from pure noise (alpha_bar=0)!
                        # T=50 with max_beta=0.1 lets alpha_bar reach ~0.0,
                        # matching the generation starting point.
LATENT_DIM = 32         # dimensionality of the ICM latent space
REPLAY_RATIO = 0.3      # fraction of each training batch that is synthetic
                        # (0.3 = 30% generated, 70% real transitions)
UPDATES_PER_EPISODE = 8 # gradient steps per environment episode
PGR_START_BUFFER = 200  # don't start generating until we have this many
                          # real transitions (need enough data to train diffusion)

# Classifier-Free Guidance (CFG) — this is what makes conditioning work.
# During training, we randomly "drop" the condition so the model also learns
# what unconditional generation looks like. At generation time, we amplify
# the difference between conditional and unconditional predictions.
CFG_P_UNCOND = 0.25     # probability of dropping the relevance condition
                        # during diffusion training (paper uses 0.25)
CFG_GUIDANCE_SCALE = 2.0  # omega: how strongly to steer toward the condition
                          # 1.0 = no guidance, >1 = amplified conditioning
                          # paper uses values around 1.5-3.0

# ── Rare-event memory ───────────────────────────────────────────────────────
# This is OUR contribution: a small separate buffer that stores hazardous
# transitions so the diffusion model never forgets them.
RARE_BUFFER_SIZE = 500    # max catastrophic transitions to remember
RARE_BATCH_RATIO = 0.1    # fraction of each diffusion training batch drawn / how often each memory is replayed
                          # from the rare buffer (0.2 = 20% rare, 80% normal)
RARE_WEIGHT = 3.0         # loss weight multiplier — rare transitions contribute / how strongly it affects diffusion learning
SALIENCE_BUFFER_SIZE = 500
SALIENCE_BATCH_RATIO = 0.1
SALIENCE_WEIGHT = 3.0     # salience-memory transitions get a smaller boost than hazards
                          # but are still replayed often enough to shape motivation

# High-reward memory is separate from salience memory:
# it stores genuinely good outcomes by reward, not critic TD error.
HIGH_REWARD_BUFFER_SIZE = 500
HIGH_REWARD_BATCH_RATIO = 0.1
HIGH_REWARD_WEIGHT = 2.0
HIGH_REWARD_THRESHOLD = 1.0

# ── Cost constraint (Lagrangian) ────────────────────────────────────────────
# This is what makes agents actually AVOID hazards. Without this, the policy
# only maximizes reward and ignores cost entirely.
#
# How it works:
#   effective_reward = reward - λ * cost
#   λ auto-tunes: if cumulative cost > COST_LIMIT, λ increases (penalize more)
#                 if cumulative cost < COST_LIMIT, λ decreases (relax penalty)
#
# This creates a constrained optimization: maximize reward SUBJECT TO
# expected cost per episode staying below COST_LIMIT.
COST_LIMIT = 2.0           # target max cost per episode (2 hazard hits)
                            # agents must learn to hit ≤2 hazards per episode
LAMBDA_LR = 1e-2            # learning rate for the Lagrange multiplier
                            # higher = faster adaptation but can oscillate
LAMBDA_INIT = 5.0           # initial penalty strength — start aggressive so
                            # agents learn hazard avoidance from early training
