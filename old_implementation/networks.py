"""
Neural network components for the PGR Safety experiment.

This file contains all the "brains" — the neural nets that learn:

  SAC (the RL algorithm):
    - GaussianPolicy   — the actor: picks actions given states
    - QNetwork         — the critic: estimates how good (state, action) pairs are

  Diffusion (the generative model — core of PGR):
    - SinusoidalEmbedding — encodes the diffusion timestep as a vector
    - ResidualBlock       — building block with skip connections
    - NoisePredictor      — the actual denoiser network (predicts noise to remove)
    - Diffusion           — orchestrates the full DDPM process + CFG

  ICM (Intrinsic Curiosity Module — the relevance function F):
    - StateEncoder   — maps raw states into a learned latent space
    - ForwardModel   — predicts next latent state from (current latent, action)
    - Curiosity = prediction error of the ForwardModel
      High error = "I haven't seen this kind of transition much" = novel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HIDDEN_DIM, LATENT_DIM, DIFFUSION_STEPS, DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
# SAC components
# These are standard and intentionally simple — a 6D point env doesn't need
# big networks. Two hidden layers of 128 units each is plenty.
# ═══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Q-function: estimates Q(s, a) — the expected total future reward
    if we take action `a` in state `s` and then follow our policy.

    Input:  state concatenated with action  →  [s; a]
    Output: single scalar Q-value

    SAC uses TWO Q-networks (q1, q2) and takes the minimum to reduce
    overestimation bias (this is the "clipped double-Q" trick).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),  # single Q-value output
        )

    def forward(self, state, action):
        # Concatenate state and action, pass through MLP
        return self.net(torch.cat([state, action], dim=-1))


class GaussianPolicy(nn.Module):
    """
    Stochastic actor for SAC. Outputs a Gaussian distribution over actions.

    Given a state, it predicts:
      - mean:    the center of the action distribution
      - log_std: the log standard deviation (how much randomness)

    Then we sample from this Gaussian and squash through tanh to get
    actions in [-1, 1]. The log_prob is needed for SAC's entropy term.

    Why stochastic? SAC maximizes reward + entropy, encouraging the policy
    to be as random as possible while still getting high reward. This
    promotes exploration and avoids collapsing to a single deterministic action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        # Shared backbone
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Separate heads for mean and log-std
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)
        # Clamp log_std to avoid numerical issues:
        #   exp(-20) ≈ 0 (nearly deterministic), exp(2) ≈ 7.4 (very noisy)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action and compute its log probability.

        Steps:
          1. Get mean and std from the network
          2. Sample from Normal(mean, std) using reparameterization trick
             (rsample allows gradients to flow through the sampling)
          3. Squash through tanh to bound actions to [-1, 1]
          4. Compute log_prob with the tanh correction term
             (because tanh changes the probability density)

        Returns: (action, log_prob)  both as tensors
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick: sample = mean + std * epsilon
        x = normal.rsample()
        action = torch.tanh(x)

        # Log probability with tanh squashing correction:
        # log π(a|s) = log N(x; μ, σ) - Σ log(1 - tanh²(x))
        log_prob = (
            normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """Convenience method for environment interaction (no grad needed)."""
        with torch.no_grad():
            action, _ = self.sample(state)
            return action.cpu().numpy()[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion components
#
# How diffusion works (simplified):
#   TRAINING:  take real data x0 → add noise to get xt → train network to
#              predict what noise was added. Repeat for random timesteps t.
#   GENERATION: start from pure noise xT → iteratively predict and remove
#              noise → end up with a clean sample x0.
#
# The "conditional" part: we also feed a relevance score to the network,
# so it learns to generate transitions that match a given relevance level.
# CFG amplifies this conditioning at generation time.
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    """
    Encode the diffusion timestep t (an integer) as a continuous vector.

    Uses sine/cosine at different frequencies — same idea as positional
    encoding in Transformers. This lets the network smoothly distinguish
    between different noise levels (t=0 is clean, t=T is pure noise).

    Input:  t of shape (batch_size,) — integer timesteps
    Output: embedding of shape (batch_size, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        # Logarithmically spaced frequencies from 1 to 1/10000
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        # Outer product: each timestep × each frequency
        args = t[:, None].float() * freqs[None, :]
        # Concatenate sin and cos for the full embedding
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: the backbone building block for the denoiser.

    Architecture:  x → LayerNorm → Linear → SiLU → Linear → + x (skip)

    Why residual connections matter for diffusion:
      At low noise levels (small t), the denoiser needs to output something
      very close to the input (the noise is tiny). Without skip connections,
      the network has to learn an identity-like mapping from scratch through
      all layers. Residual connections give it the identity for free — it
      just needs to learn the small corrections.

    Why LayerNorm:
      The relevance scores shift in distribution as the ICM evolves during
      training. LayerNorm keeps activations stable regardless.

    Why SiLU (Swish):
      Smooth, non-monotonic activation — works better than ReLU for
      diffusion models empirically. SiLU(x) = x * sigmoid(x).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # Skip connection: output = input + learned_residual
        return x + self.net(self.norm(x))


class NoisePredictor(nn.Module):
    """
    The core neural network of the diffusion model. Given:
      - x_t:  a noised transition vector (state, action, reward, cost, next_state)
      - t:    the noise level (which diffusion timestep)
      - rel:  the relevance condition (curiosity score)

    It predicts the noise ε that was added, so we can subtract it to denoise.

    Architecture (from PGR paper — "residual MLP"):
      1. Project transition into hidden dimension
      2. Add timestep embedding + relevance embedding (additive conditioning)
      3. Pass through 4 residual blocks
      4. LayerNorm + project back to transition dimension

    The additive conditioning means: the network processes the transition
    differently depending on what timestep and relevance value it receives.
    It learns "for high-relevance, generate this kind of transition; for
    low-relevance, generate that kind."
    """

    def __init__(self, transition_dim: int, hidden: int = HIDDEN_DIM, n_blocks: int = 4):
        super().__init__()

        # Timestep embedding: integer t → learned vector
        self.time_embed = SinusoidalEmbedding(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )

        # Relevance conditioning: scalar → learned vector
        # This is where the "priority" information enters the model
        self.rel_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )

        # Project the flat transition vector up to hidden dimension
        self.project_in = nn.Linear(transition_dim, hidden)

        # Stack of residual blocks — the main processing backbone
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])

        # Project back down to transition dimension for the noise prediction
        self.final_norm = nn.LayerNorm(hidden)
        self.project_out = nn.Linear(hidden, transition_dim)

    def forward(self, x, t, rel):
        """
        Predict noise given noised input, timestep, and relevance condition.

        Args:
            x:   (batch, transition_dim) — noised transition
            t:   (batch,) — diffusion timestep
            rel: (batch, 1) — relevance score (or zeros if condition dropped)

        Returns:
            (batch, transition_dim) — predicted noise to subtract
        """
        # Compute conditioning vectors
        t_emb = self.time_mlp(self.time_embed(t))    # (B, hidden)
        r_emb = self.rel_mlp(rel)                     # (B, hidden)

        # Project input and ADD conditioning (not concatenate — more elegant,
        # and the residual blocks can then process everything uniformly)
        h = self.project_in(x) + t_emb + r_emb

        # Pass through residual backbone
        for block in self.blocks:
            h = block(h)

        # Final norm + project to output dimension
        return self.project_out(self.final_norm(h))


class Diffusion:
    """
    Full DDPM (Denoising Diffusion Probabilistic Model) with CFG.

    This orchestrates the entire diffusion process:

    TRAINING (loss method):
      1. Take a real transition x0
      2. Pick a random timestep t
      3. Add noise: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
      4. Ask the NoisePredictor to predict ε from x_t
      5. Loss = MSE between predicted and actual noise
      * CFG twist: with probability p_uncond, we zero out the relevance
        condition. This teaches the model what "unconditional" looks like.

    GENERATION (generate method):
      1. Start from pure Gaussian noise x_T
      2. For t = T-1 down to 0:
         a. Predict noise with condition (eps_cond)
         b. Predict noise without condition (eps_uncond)
         c. CFG blend: eps = ω * eps_cond + (1-ω) * eps_uncond
            This AMPLIFIES the effect of conditioning — pushes generations
            further toward the conditioned relevance level.
         d. Remove the predicted noise to get x_{t-1}
      3. Return the final clean sample x_0

    Why CFG matters:
      Without CFG, the model sees the relevance condition every time during
      training, so it learns to mildly adjust outputs. CFG creates a contrast
      between "with condition" and "without condition", then amplifies that
      difference at generation time. This is what makes conditioning actually
      steer the generations meaningfully.
    """

    def __init__(
        self,
        model: NoisePredictor,
        T: int = DIFFUSION_STEPS,
        p_uncond: float = 0.25,
        guidance_scale: float = 2.0,
    ):
        self.model = model
        self.T = T                        # number of diffusion steps
        self.p_uncond = p_uncond          # CFG: prob of dropping condition
        self.guidance_scale = guidance_scale  # CFG: ω, how hard to steer

        # Pre-compute the noise schedule (linear beta schedule)
        # betas go from small (barely any noise) to larger (more noise)
        # max_beta=0.1 ensures alpha_bar reaches ~0.0 at step T,
        # matching the pure Gaussian noise we start from during generation.
        # (Old max_beta=0.02 left alpha_bar at ~0.90 = still 90% clean = garbage generation)
        betas = torch.linspace(1e-4, 0.1, T, device=DEVICE)
        alphas = 1 - betas                           # α_t = 1 - β_t
        alpha_bar = torch.cumprod(alphas, 0)         # ᾱ_t = Π α_i (cumulative product)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()       # for noising: x_t = √ᾱ * x0 + ...
        self.sqrt_1m_alpha_bar = (1 - alpha_bar).sqrt()  # ... + √(1-ᾱ) * noise

    def loss(self, x0, rel, weights=None):
        """
        Compute the diffusion training loss on a batch of real transitions.

        Args:
            x0:      (batch, dim) — clean transition vectors from the replay buffer
            rel:     (batch, 1) — relevance scores for each transition
            weights: (batch,) — optional per-sample weights (used by PGR+Memory
                     to upweight rare transitions)

        Returns:
            scalar loss (MSE between predicted and actual noise)
        """
        B = x0.shape[0]

        # Random timestep per sample — each transition gets noised differently
        t = torch.randint(0, self.T, (B,), device=DEVICE)

        # The noise we're going to add (ground truth for the network to predict)
        noise = torch.randn_like(x0)

        # Forward diffusion: add noise to x0 to get x_t
        # x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * ε
        xt = self.sqrt_alpha_bar[t, None] * x0 + self.sqrt_1m_alpha_bar[t, None] * noise

        # CFG: randomly drop the relevance condition (replace with zeros)
        # This teaches the model what "unconditional" generation looks like
        # mask = 1 (keep condition) with prob (1-p_uncond), else 0 (drop it)
        mask = (torch.rand(B, 1, device=DEVICE) > self.p_uncond).float()
        rel_masked = rel * mask    # zeros where condition is dropped

        # Ask the network: "what noise was added?"
        pred = self.model(xt, t, rel_masked)

        # Per-sample MSE between predicted and actual noise
        per_sample = ((pred - noise) ** 2).mean(dim=-1)

        # Apply per-sample weights if provided (rare events get higher weight)
        if weights is not None:
            return (weights * per_sample).mean()
        return per_sample.mean()

    @torch.no_grad()
    def generate(self, shape, rel):
        """
        Generate synthetic transitions via iterative denoising with CFG.

        Args:
            shape: (n_samples, transition_dim) — how many and how big
            rel:   (n_samples, 1) — relevance values to condition on

        Returns:
            (n_samples, transition_dim) — generated transition vectors
        """
        # Start from pure Gaussian noise
        x = torch.randn(shape, device=DEVICE)
        null_rel = torch.zeros_like(rel)    # the "no condition" input
        omega = self.guidance_scale

        # Reverse diffusion: denoise step by step from t=T-1 to t=0
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)

            # CFG: get both conditional and unconditional noise predictions
            eps_cond = self.model(x, t, rel)          # "what noise if relevance = rel"
            eps_uncond = self.model(x, t, null_rel)    # "what noise if no condition"

            # Amplify the difference: guided = uncond + ω * (cond - uncond)
            # Equivalent to: ω * cond + (1-ω) * uncond
            # When ω > 1, we push BEYOND what the condition says — stronger steering
            eps = omega * eps_cond + (1 - omega) * eps_uncond

            # Reverse diffusion step: remove predicted noise
            alpha = self.alphas[t, None]
            beta = self.betas[t, None]
            mean = (x - beta / self.sqrt_1m_alpha_bar[t, None] * eps) / alpha.sqrt()

            # Add fresh noise for all steps except the last (t=0)
            if i > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean    # final step: just take the mean (clean output)

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ICM (Intrinsic Curiosity Module)
#
# The ICM provides the relevance function F for PGR. Here's the idea:
#
#   1. Encode states into a latent space:       h(s) = StateEncoder(s)
#   2. Predict next latent from current + action: ĥ = ForwardModel(h(s), a)
#   3. Curiosity = prediction error:            F = ||ĥ - h(s')||²
#
# High curiosity = the ICM can't predict what happens next = this transition
# is novel/surprising. PGR conditions the diffusion model on this score
# to generate MORE transitions in novel regions.
#
# The problem our project addresses: once the ICM learns to predict hazard
# transitions well, their curiosity drops to zero, and PGR stops generating
# them. Our rare-event buffer prevents this forgetting.
# ═══════════════════════════════════════════════════════════════════════════════

class StateEncoder(nn.Module):
    """
    Encode raw states into a learned latent representation.

    3 layers with LayerNorm (deeper than original code's 1 hidden layer).
    This matters because the curiosity signal is computed in this latent
    space — if the encoder is too shallow, it can't capture the nonlinear
    structure around hazard boundaries, and all transitions look equally
    "boring" to the ICM.

    Input:  raw state (batch, state_dim)       e.g. (128, 6)
    Output: latent    (batch, latent_dim)       e.g. (128, 32)
    """

    def __init__(self, state_dim: int, latent_dim: int = LATENT_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),      # stabilize as input distribution shifts
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),  # no activation on output — latent can be any value
        )

    def forward(self, s):
        """Map raw state to latent representation."""
        return self.net(s)


class ForwardModel(nn.Module):
    """
    Predict the next latent state from (current latent state, action).

    This models the environment dynamics in latent space:
        ĥ(s') = ForwardModel(h(s), a)

    The prediction error ||ĥ(s') - h(s')||² becomes the curiosity score.
    If this error is high, the transition is "surprising" — the ICM hasn't
    seen enough similar transitions to predict them accurately.

    Input:  h(s) concatenated with action  → [h(s); a]
    Output: predicted h(s') in latent space
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, h_s, a):
        """Predict next latent from current latent + action."""
        return self.net(torch.cat([h_s, a], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize curiosity scores to be non-negative with roughly unit variance.

    Raw curiosity scores can vary wildly in magnitude over training
    (early: everything is novel → high scores; late: well-learned → low scores).
    Normalizing keeps the conditioning signal on a consistent scale for
    the diffusion model.

    Steps:
      1. Zero-mean, unit-variance (standard z-score)
      2. Shift so minimum is slightly above zero (relevance should be positive)
    """
    std = scores.std() + 1e-8                  # avoid division by zero
    normed = (scores - scores.mean()) / std    # z-score
    return normed - normed.min() + 1e-6        # shift to be positive
