"""
Agent implementations for the PGR Safety experiment.

Three agents, each building on the last:

  1. SACAgent           — vanilla Soft Actor-Critic. Learns from real data only.
                          This is our "no generative replay" baseline.

  2. SACPGRAgent        — SAC + Prioritized Generative Replay. Trains a diffusion
                          model on real transitions, conditions on curiosity, and
                          generates synthetic transitions to augment training.
                          Like the PGR paper (Wang et al., 2025).

  3. SACPGRMemoryAgent  — PGR + our rare-event memory bank. Same as above, but
                          also stores hazardous transitions in a separate buffer
                          and upweights them during diffusion training. This is
                          OUR contribution — preventing forgetting of catastrophic
                          events that the curiosity signal would otherwise
                          deprioritize once they become "familiar."
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import (
    DEVICE, LR, GAMMA, TAU, BATCH_SIZE,
    LATENT_DIM, REPLAY_RATIO, PGR_START_BUFFER,
    RARE_BATCH_RATIO, RARE_WEIGHT,
    CFG_P_UNCOND, CFG_GUIDANCE_SCALE,
    COST_LIMIT, LAMBDA_LR, LAMBDA_INIT,
    # Experiment configs
    ADAPTIVE_RARE_MIN, ADAPTIVE_RARE_MAX, ADAPTIVE_STALE_WINDOW,
    SAFETY_CRITIC_LR, SAFETY_CRITIC_WEIGHT,
    RARE_AUGMENT_NOISE, RARE_AUGMENT_FACTOR,
    RARE_WEIGHT_MIN, RARE_WEIGHT_MAX,
)
from buffers import ReplayBuffer, RareEventBuffer
from networks import (
    QNetwork, GaussianPolicy,
    StateEncoder, ForwardModel,
    NoisePredictor, Diffusion,
    CostQNetwork,
    normalize_scores,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Base SAC Agent
#
# SAC (Soft Actor-Critic) is an off-policy RL algorithm that learns:
#   - A policy π(a|s)       — "what action to take" (the actor)
#   - Q-functions Q(s,a)    — "how good is this state-action pair" (critics)
#   - Temperature α         — balances reward vs exploration (entropy)
#
# The "soft" part: SAC adds an entropy bonus to the reward, encouraging
# the policy to be as random as possible while still achieving high reward.
# This helps with exploration and makes training more stable.
# ═══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    """Vanilla SAC baseline — learns purely from real environment experience."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ── Actor: the policy network ────────────────────────────────────
        self.policy = GaussianPolicy(state_dim, action_dim).to(DEVICE)

        # ── Critics: two Q-networks (double-Q trick to reduce overestimation)
        # We also keep "target" copies that update slowly for stable training.
        # The target networks provide the bootstrap target for Q-learning:
        #   Q_target(s', a') where a' ~ π(·|s')
        self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())  # start identical
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ── Entropy temperature (auto-tuned) ─────────────────────────────
        # α controls the reward-entropy tradeoff: J = E[r + α * entropy]
        # Higher α = more exploration, lower α = more exploitation.
        # SAC auto-tunes α so that the policy's entropy stays near a target.
        self.target_entropy = -action_dim  # heuristic: -dim(action_space)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp()

        # ── Optimizers ───────────────────────────────────────────────────
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

        # ── Lagrangian cost constraint ──────────────────────────────────
        # λ (lambda) penalizes the reward when the agent hits hazards:
        #   effective_reward = reward - λ * cost
        #
        # λ auto-tunes via dual gradient descent:
        #   If recent cost > COST_LIMIT → λ increases (penalize harder)
        #   If recent cost < COST_LIMIT → λ decreases (relax penalty)
        #
        # We store log(λ) and exponentiate so λ stays positive.
        # This is the same trick SAC uses for α (entropy temperature).
        self.log_lambda = torch.tensor(
            np.log(max(LAMBDA_INIT, 1e-8)),  # log(1.0) = 0.0
            requires_grad=True, device=DEVICE, dtype=torch.float32,
        )
        self.lam = self.log_lambda.exp().item()  # current λ value
        self.lambda_opt = optim.Adam([self.log_lambda], lr=LAMBDA_LR)

        # Track recent episode costs for λ update
        self.recent_costs = []  # costs from recent episodes

        # ── Lambda stabilisation ──────────────────────────────────────
        # EMA (exponential moving average) of episode cost smooths out
        # the spiky signal that rare-event replay creates.  Without it,
        # a single replay burst can make λ spike → policy collapse →
        # more cost → λ spikes further (death spiral).
        #
        # EMA update:  ema = decay * ema + (1-decay) * new_value
        # With decay=0.95, a one-off spike of 50 only moves the EMA by
        # 2.5 instead of 50.  Sustained high cost still pushes λ up.
        self.cost_ema = float(COST_LIMIT)  # start at target (neutral)
        self.cost_ema_decay = 0.95         # higher = smoother / slower

        # ── Replay buffer ────────────────────────────────────────────────
        self.buffer = ReplayBuffer(state_dim, action_dim)

    # ── Environment interaction ──────────────────────────────────────────

    def select_action(self, state):
        """Pick an action for the given state (used during env rollouts)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.policy.get_action(s)

    def add_transition(self, s, a, r, c, ns, d):
        """Store a transition in the replay buffer."""
        self.buffer.add(s, a, r, c, ns, d)

    # ── Core SAC update ──────────────────────────────────────────────────

    def _soft_update(self):
        """
        Slowly blend online Q-networks into target Q-networks.

        target = TAU * online + (1 - TAU) * target

        With TAU=0.005, the target network moves very slowly, providing
        a stable training signal. Without this, Q-learning is unstable
        because the target we're chasing keeps changing.
        """
        for tp, sp in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)
        for tp, sp in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

    def _sac_update(self, states, actions, rewards, costs, next_states, dones):
        """
        One full SAC gradient step with Lagrangian cost constraint.

        The key change from vanilla SAC:
          effective_reward = reward - λ * cost
        
        This makes the policy AVOID high-cost transitions. λ auto-tunes
        so the agent finds the best reward while staying under COST_LIMIT.

        Three updates happen:

        1. CRITICS (Q-networks):
           Target: y = (r - λ*c) + γ * (1-done) * (min Q_target(s', a') - α * log π(a'|s'))
           Loss:   MSE(Q(s,a), y)  for both Q1 and Q2

        2. ACTOR (policy):
           Loss:   E[α * log π(a|s) - min Q(s, a)]
           This pushes the policy toward actions with high Q and high entropy.

        3. TEMPERATURE (α):
           Loss:   -α * (log π(a|s) + target_entropy)
           If entropy is too low, α increases (more exploration).
           If entropy is too high, α decreases (more exploitation).

        Args:
            states, actions, rewards, costs, next_states, dones — batch of transitions
            (can be real, synthetic, or a mix)
        """
        # ── Compute cost-penalized reward ──────────────────────────────
        # This is the CORE of the safety mechanism:
        # effective_reward = reward - λ * cost
        # When λ is high and cost > 0, the effective reward becomes very negative,
        # teaching the Q-function that hazard transitions are BAD.
        effective_rewards = rewards - self.lam * costs

        # ── Step 1: Critic update ────────────────────────────────────────
        with torch.no_grad():
            # Sample next actions from current policy (for bootstrapping)
            next_actions, next_log_probs = self.policy.sample(next_states)

            # Double-Q: take the minimum of both target Q-networks
            # This prevents overestimation of Q-values
            q_target = torch.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            )

            # Bellman target using COST-PENALIZED reward:
            # y = (r - λ*c) + γ * (Q_target - α * log_prob)
            target = effective_rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (
                q_target - self.alpha * next_log_probs
            )

        # Update Q1
        self.q1_opt.zero_grad()
        F.mse_loss(self.q1(states, actions), target).backward()
        self.q1_opt.step()

        # Update Q2
        self.q2_opt.zero_grad()
        F.mse_loss(self.q2(states, actions), target).backward()
        self.q2_opt.step()

        # ── Step 2: Actor (policy) update ────────────────────────────────
        # Sample fresh actions from current policy
        new_actions, log_probs = self.policy.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))

        # Policy loss: maximize Q while maximizing entropy
        # = minimize (α * log_prob - Q)
        self.policy_opt.zero_grad()
        (self.alpha.detach() * log_probs - q_new).mean().backward()
        self.policy_opt.step()

        # ── Step 3: Temperature (α) update ───────────────────────────────
        # If entropy > target → α decreases (less exploration needed)
        # If entropy < target → α increases (need more exploration)
        self.alpha_opt.zero_grad()
        (-(self.log_alpha * (log_probs + self.target_entropy).detach())).mean().backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        # Slowly update target networks
        self._soft_update()

    def record_episode_cost(self, episode_cost):
        """
        Called at the end of each episode to record how much cost was incurred.
        After enough episodes, we update λ (the Lagrange multiplier).

        Dual gradient descent on λ:
          ∇λ = (avg_recent_cost - COST_LIMIT)
          If avg_cost > COST_LIMIT → gradient positive → λ increases → penalize more
          If avg_cost < COST_LIMIT → gradient negative → λ decreases → relax penalty

        We update every 10 episodes to reduce noise.
        """
        self.recent_costs.append(episode_cost)

        # Update λ every 10 episodes (smooths out noise)
        if len(self.recent_costs) >= 10:
            avg_cost = np.mean(self.recent_costs)
            self.recent_costs = []  # reset

            # ── EMA smoothing ─────────────────────────────────────────
            # Instead of feeding raw avg_cost into the λ gradient, we
            # update an exponential moving average first.  This prevents
            # a single burst of replayed rare events from spiking λ.
            self.cost_ema = (self.cost_ema_decay * self.cost_ema
                            + (1 - self.cost_ema_decay) * avg_cost)

            # Dual gradient ASCENT: push λ toward satisfying the constraint
            # We negate because the optimizer does gradient DESCENT (minimizes),
            # but for the Lagrange dual we need ASCENT (maximize).
            #
            # FIX: Use log_lambda directly, NOT log_lambda.exp().
            # With .exp(), the gradient = -λ * (cost - limit). When λ gets small,
            # the gradient vanishes and λ can never recover (death spiral).
            # Without .exp(), the gradient = -(cost - limit), a constant that
            # responds immediately regardless of λ's current value.
            # This matches how SAC tunes its entropy temperature α.
            self.lambda_opt.zero_grad()
            lambda_loss = -self.log_lambda * (self.cost_ema - COST_LIMIT)
            lambda_loss.backward()
            self.lambda_opt.step()

            # Update the cached value (used in _sac_update)
            self.lam = self.log_lambda.exp().item()

            # Clamp to prevent λ from going crazy
            # Max tightened from 100→20: even λ=20 makes cost 10× more
            # important than reward, which is already very aggressive.
            # The old cap of 100 allowed the death spiral we observed in
            # Cheetah (PGR+Memory seeds 123, 456) and Ant (seed 42 @ 16k).
            self.lam = np.clip(self.lam, 0.01, 20.0)
            self.log_lambda.data.copy_(torch.tensor(np.log(self.lam), device=DEVICE))

    def train_step(self):
        """One training iteration: sample batch from buffer → SAC update."""
        if len(self.buffer) < BATCH_SIZE:
            return  # not enough data yet
        states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)
        self._sac_update(states, actions, rewards, costs, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC + PGR (Prioritized Generative Replay)
#
# This extends SAC with a conditional diffusion model that:
#   1. Learns to generate transitions from the replay buffer
#   2. Uses curiosity (ICM prediction error) as a relevance signal
#   3. Generates synthetic transitions biased toward high-curiosity regions
#   4. Mixes synthetic + real data for SAC training
#
# Key insight from the paper: conditioning on curiosity generates MORE DIVERSE
# transitions, which REDUCES overfitting of the Q-function to synthetic data.
# This is NOT about generating higher-quality data — it's about generating
# the RIGHT KIND of data.
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRAgent(SACAgent):
    """SAC + curiosity-conditioned diffusion replay (PGR paper approach)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)

        # Transition dimension: the diffusion model generates entire transitions
        # as flat vectors: [s, a, r, c, s', d] all concatenated
        # The done flag is included so the model learns episode boundaries
        # (without it, synthetic goal transitions get done=False, causing
        # infinite Q-value loops in the Bellman equation)
        self.transition_dim = state_dim + action_dim + 1 + 1 + state_dim + 1  # s,a,r,c,s',d

        # ── ICM (Intrinsic Curiosity Module) ─────────────────────────────
        # Provides the relevance function F(τ) = prediction error
        self.encoder = StateEncoder(state_dim).to(DEVICE)
        self.fwd_model = ForwardModel(LATENT_DIM, action_dim).to(DEVICE)

        # ── Conditional diffusion model ──────────────────────────────────
        # Learns to generate transitions conditioned on curiosity scores
        self.noise_pred = NoisePredictor(self.transition_dim).to(DEVICE)
        self.diffusion = Diffusion(
            self.noise_pred,
            p_uncond=CFG_P_UNCOND,           # 25% of time, drop the condition
            guidance_scale=CFG_GUIDANCE_SCALE, # amplify conditioning at gen time
        )

        # ── Optimizers for the new components ────────────────────────────
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=LR)
        self.fwd_opt = optim.Adam(self.fwd_model.parameters(), lr=LR)
        self.diff_opt = optim.Adam(self.noise_pred.parameters(), lr=LR)

        # ── Normalization stats for diffusion ─────────────────────────────
        # Diffusion models assume input features are roughly N(0,1).
        # But our transitions mix unbounded states, bounded actions [-1,1],
        # and high-variance rewards — all different scales.
        # We dynamically track mean/std and standardize before diffusion.
        self.trans_mean = torch.zeros(self.transition_dim, device=DEVICE)
        self.trans_std = torch.ones(self.transition_dim, device=DEVICE)

        # Track how many gradient steps the diffusion model has taken.
        # We don't generate synthetic data until it's had enough training
        # (burn-in), otherwise we feed garbage to the Q-network.
        self.diffusion_updates = 0

    # ── Curiosity scoring ────────────────────────────────────────────────

    def _compute_curiosity(self, states, actions, next_states):
        """
        Compute ICM curiosity scores for a batch of transitions.

        Curiosity = forward model prediction error in latent space:
            F(s, a, s') = 0.5 * ||g(h(s), a) - h(s')||²

        High score = the ICM can't predict this transition = it's novel.
        Low score = the ICM has seen enough similar transitions = familiar.

        Returns: (batch_size,) tensor of curiosity scores (detached — no grad)
        """
        with torch.no_grad():
            h_s = self.encoder(states)        # encode current state
            h_ns = self.encoder(next_states)  # encode next state (ground truth)
            # Prediction error: how far off is the forward model?
            pred_h_ns = self.fwd_model(h_s, actions)
            return 0.5 * ((pred_h_ns - h_ns) ** 2).mean(dim=-1)

    def _train_icm(self):
        """
        One gradient step to improve the ICM's forward prediction.

        We train the encoder and forward model jointly to minimize:
            ||ForwardModel(Encoder(s), a) - Encoder(s')||²

        As the ICM gets better at predicting, curiosity scores for familiar
        transitions drop — which is exactly what we want. Novel transitions
        remain high-curiosity because the ICM hasn't learned them yet.
        """
        s, a, _, _, ns, _ = self.buffer.sample(BATCH_SIZE)
        h_s, h_ns = self.encoder(s), self.encoder(ns)

        self.encoder_opt.zero_grad()
        self.fwd_opt.zero_grad()
        F.mse_loss(self.fwd_model(h_s, a), h_ns).backward()
        self.encoder_opt.step()
        self.fwd_opt.step()

    # ── Diffusion training ───────────────────────────────────────────────

    def _train_diffusion(self, transitions, scores, weights=None):
        """
        One gradient step on the conditional diffusion model.

        Teaches the diffusion model: "given this curiosity score as a condition,
        learn to generate transitions that look like the real ones with that
        score level."

        FIX: We standardize transitions to ~N(0,1) before feeding to diffusion.
        Without this, the loss is dominated by high-variance features and the
        model generates out-of-bounds actions that destroy the Q-function.

        Args:
            transitions: (batch, transition_dim) — flat real transition vectors
            scores:      (batch,) — normalized curiosity scores for each
            weights:     (batch,) — optional per-sample loss weights
        """
        x0 = torch.FloatTensor(transitions).to(DEVICE)

        # Standardize to ~N(0,1) so diffusion sees balanced features
        x0 = (x0 - self.trans_mean) / self.trans_std

        rel = torch.FloatTensor(scores[:, None]).to(DEVICE)  # (batch, 1)
        w = torch.FloatTensor(weights).to(DEVICE) if weights is not None else None

        self.diff_opt.zero_grad()
        self.diffusion.loss(x0, rel, weights=w).backward()
        self.diff_opt.step()
        self.diffusion_updates += 1

    # ── Synthetic generation ─────────────────────────────────────────────

    def _generate_synthetic(self, n_syn, scores_np):
        """
        Generate n_syn synthetic transitions using the diffusion model,
        conditioned on HIGH curiosity scores.

        Strategy (from PGR paper — "prompting"):
          1. Find the top 10% of curiosity scores in the current pool
          2. Randomly sample from these top scores + small noise
          3. Use these as the relevance condition for diffusion generation

        Post-processing fixes applied:
          - Un-standardize back to physical env bounds
          - Clamp actions to [-1,1] (prevents Q-function extrapolation)
          - Binarize cost at 0.5 (prevents micro-cost minefield effect)
          - Binarize done at 0.5 (prevents infinite Bellman loops)

        Returns: tuple of (states, actions, rewards, costs, next_states, dones)
                 all as GPU tensors
        """
        # Pick conditioning values from the top 10% of curiosity scores
        top_k = max(1, int(len(scores_np) * 0.1))
        top_scores = scores_np[np.argsort(scores_np)[-top_k:]]

        # Sample conditions + small Gaussian noise for variety
        conds = np.maximum(
            0,
            np.random.choice(top_scores, n_syn) + np.random.normal(0, 0.1, n_syn),
        )

        # Generate! This runs the full reverse diffusion with CFG
        syn = self.diffusion.generate(
            (n_syn, self.transition_dim),
            torch.FloatTensor(conds[:, None]).to(DEVICE),
        )

        # Un-standardize: convert back from N(0,1) to physical environment scale
        syn = syn * self.trans_std + self.trans_mean

        # Slice the flat vector back into individual components
        # Layout: [s (state_dim) | a (action_dim) | r (1) | c (1) | s' (state_dim) | d (1)]
        syn_s  = syn[:, :self.state_dim]
        # Clamp actions to [-1,1] — prevents Q-function extrapolation catastrophe
        syn_a  = torch.clamp(syn[:, self.state_dim : self.state_dim + self.action_dim], -1.0, 1.0)
        syn_r  = syn[:, self.state_dim + self.action_dim]
        # Binarize cost: diffusion outputs continuous values, but cost is 0 or 1.
        # Without this, micro-costs like 0.05 make the agent think everywhere is dangerous.
        syn_c  = (syn[:, self.state_dim + self.action_dim + 1] > 0.5).float()
        syn_ns = syn[:, self.state_dim + self.action_dim + 2 : -1]
        # Binarize done: the diffusion model learned when episodes end.
        # Without this, goal transitions get done=False → Q-function thinks
        # it can collect +10 reward infinitely → Q-values explode.
        syn_d  = (syn[:, -1] > 0.5).float()

        return syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d

    # ── Diffusion hazard probe ────────────────────────────────────────────

    def probe_diffusion_hazard_rate(self, n_samples=1000, n_rounds=3):
        """
        Diagnostic: measure what fraction of synthetic transitions contain
        hazards (cost > 0.5 before binarization).

        This directly tests the core claim: PGR's diffusion model
        compresses out rare hazard transitions over time as safe data
        dominates the buffer. PGR+Memory's rare buffer should prevent
        this compression by upweighting hazard data in the diffusion loss.

        Expected results:
          PGR:        hazard rate declines over training (compression)
          PGR+Memory: hazard rate stays stable (rare buffer prevents compression)

        Stability improvements over naive probing:
          - Uses FIXED conditioning values (percentiles 70/80/90/95/99 of
            normalized scores) instead of random sampling, so the same
            "question" is asked every probe.
          - Runs n_rounds and averages, reducing single-probe noise.
          - Uses n_samples=1000 for better statistics.

        Returns: (hazard_rate, mean_raw_cost)
          hazard_rate: fraction of generated transitions with cost > 0.5
          mean_raw_cost: average raw (pre-binarization) cost value
          Returns (None, None) if diffusion isn't ready yet.
        """
        if self.diffusion_updates < 2000:
            return None, None  # diffusion not trained enough

        if len(self.buffer) < 500:
            return None, None

        all_hz_rates = []
        all_raw_costs = []

        for _ in range(n_rounds):
            # Compute curiosity scores on a fresh sample
            s, a, ns, _, _, _, idx = self.buffer.sample_with_idx(
                min(3000, len(self.buffer))
            )
            scores_np = normalize_scores(
                self._compute_curiosity(s, a, ns).cpu().numpy()
            )

            # Use FIXED percentiles of curiosity scores as conditioning values.
            # This ensures the same "question" is asked each time, reducing
            # variance from random score sampling. We use high percentiles
            # because _generate_synthetic also conditions on top scores.
            percentiles = [70, 80, 90, 95, 99]
            pct_values = np.percentile(scores_np, percentiles)
            # Assign each sample a conditioning value from these percentiles
            conds = np.repeat(pct_values, n_samples // len(percentiles) + 1)[:n_samples]

            # Generate synthetic transitions
            syn = self.diffusion.generate(
                (n_samples, self.transition_dim),
                torch.FloatTensor(conds[:, None]).to(DEVICE),
            )
            # Un-standardize
            syn = syn * self.trans_std + self.trans_mean

            # Extract raw cost values (before binarization)
            cost_idx = self.state_dim + self.action_dim + 1
            raw_costs = syn[:, cost_idx].cpu().numpy()

            all_hz_rates.append(float((raw_costs > 0.5).mean()))
            all_raw_costs.append(float(raw_costs.mean()))

        hazard_rate = float(np.mean(all_hz_rates))
        mean_raw_cost = float(np.mean(all_raw_costs))

        return hazard_rate, mean_raw_cost

    # ── Main training step ───────────────────────────────────────────────

    def train_step(self):
        """
        One PGR training iteration. This is the core algorithm (Algorithm 1 in paper):

        Phase 1 — Score & learn (when enough data):
          a) Sample a pool of transitions from the buffer
          b) Compute curiosity scores for the pool
          c) Train the ICM (so it gets better at predicting)
          d) Train the diffusion model on real transitions + their scores

        Phase 2 — Generate & mix:
          e) Generate synthetic transitions conditioned on high curiosity
          f) Mix real + synthetic data (REPLAY_RATIO controls the split)
          g) Run one SAC update on the mixed batch
        """
        if len(self.buffer) < BATCH_SIZE:
            return  # need enough data first

        # Only use PGR once we have enough data for the diffusion model
        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            # ── Phase 1: Score a pool and train components ───────────
            # Sample a large pool of transitions (up to 5000)
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            # Compute and normalize curiosity for the pool
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            # Update normalization stats from the current pool
            # This keeps the diffusion model seeing standardized ~N(0,1) data
            pool_trans = self.buffer.get_transitions(idx)
            self.trans_mean = torch.FloatTensor(pool_trans.mean(axis=0)).to(DEVICE)
            # Clamp std to at least 0.01 to prevent normalization explosion.
            # When cost is 0 everywhere (agent learned safety), std → 0,
            # so a single cost=1 from rare buffer becomes 1/1e-8 = 100 million.
            # Also hardcode binary features (cost, done) to std=1.0 to preserve scale.
            std = np.maximum(pool_trans.std(axis=0), 1e-2)
            cost_idx = self.state_dim + self.action_dim + 1
            std[cost_idx] = 1.0   # cost is binary 0/1, don't normalize
            std[-1] = 1.0         # done is binary 0/1, don't normalize
            self.trans_std = torch.FloatTensor(std).to(DEVICE)

            # Train ICM to get better at prediction (makes curiosity more accurate)
            self._train_icm()

            # Train diffusion model: "learn to generate transitions like these,
            # given their curiosity scores as conditioning"
            batch_idx = np.random.choice(len(idx), BATCH_SIZE, replace=True)
            self._train_diffusion(
                self.buffer.get_transitions(idx[batch_idx]),
                scores_np[batch_idx],
            )

            # ── Phase 2: Generate synthetic data and mix ─────────────
            # Only generate after enough diffusion training (burn-in).
            # Before this, the diffusion model outputs near-random garbage
            # that poisons the Q-network during critical early exploration.
            diffusion_ready = self.diffusion_updates > 2000
            n_syn = int(BATCH_SIZE * REPLAY_RATIO) if diffusion_ready else 0

            if n_syn > 0:
                syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)
                # Fill the rest of the batch with real data
                real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(BATCH_SIZE - n_syn)
                # Concatenate real + synthetic
                states      = torch.cat([real_s, syn_s])
                actions     = torch.cat([real_a, syn_a])
                rewards     = torch.cat([real_r, syn_r])
                costs       = torch.cat([real_c, syn_c])
                next_states = torch.cat([real_ns, syn_ns])
                dones       = torch.cat([real_d, syn_d])
            else:
                # Diffusion not ready yet — train on real data only
                states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)
        else:
            # Before PGR kicks in, just train on real data (same as vanilla SAC)
            states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Run SAC update on the (possibly augmented) batch
        self._sac_update(states, actions, rewards, costs, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC + PGR + Rare-event memory  (OUR CONTRIBUTION)
#
# Problem: PGR's curiosity relevance function prioritizes NOVEL transitions.
# Once the ICM learns to predict hazard transitions, they stop being novel
# and get deprioritized. The diffusion model then stops generating
# hazard-adjacent synthetic data, and the policy "forgets" to avoid hazards.
#
# Solution: maintain a separate small buffer of catastrophic transitions
# and inject them into diffusion training with upweighted loss. This forces
# the diffusion model to always know how to generate near-hazard transitions,
# even when the ICM no longer considers them novel.
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRMemoryAgent(SACPGRAgent):
    """PGR + rare-event memory bank for hazardous transitions."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        # Small dedicated buffer for catastrophic transitions
        self.rare_buffer = RareEventBuffer(state_dim, action_dim)

    def add_transition(self, s, a, r, c, ns, d):
        """
        Store transition in main buffer, AND if it's hazardous (cost > 0),
        also store it in the rare-event buffer.
        """
        self.buffer.add(s, a, r, c, ns, d)
        if c > 0:   # this transition hit a hazard!
            self.rare_buffer.add(s, a, r, c, ns, d)

    def train_step(self):
        """
        PGR+Memory training: rare events injected at TWO levels.

        Level 1 — Diffusion training (same as before):
          80% normal buffer + 20% rare buffer with 5x loss weight.
          Forces the diffusion model to remember hazard transitions.

        Level 2 — SAC policy training (NEW — Fix 3):
          The SAC batch is split three ways: real + synthetic + rare.
          This guarantees the policy sees actual hazard data every batch,
          bypassing the diffusion quality bottleneck entirely.

        The batch composition:
          - n_real = BATCH_SIZE - n_syn - n_rare  (roughly 50%)
          - n_syn  = BATCH_SIZE * REPLAY_RATIO    (30% synthetic)
          - n_rare = BATCH_SIZE * RARE_BATCH_RATIO (20% rare)
        """
        if len(self.buffer) < BATCH_SIZE:
            return

        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            # Score a pool of transitions (same as base PGR)
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            # Update normalization stats from the current pool
            # Bug 2 fix: clamp std and hardcode binary features
            pool_trans = self.buffer.get_transitions(idx)
            self.trans_mean = torch.FloatTensor(pool_trans.mean(axis=0)).to(DEVICE)
            std = np.maximum(pool_trans.std(axis=0), 1e-2)
            cost_idx = self.state_dim + self.action_dim + 1
            std[cost_idx] = 1.0   # cost is binary, don't normalize
            std[-1] = 1.0         # done is binary, don't normalize
            self.trans_std = torch.FloatTensor(std).to(DEVICE)

            # Train ICM
            self._train_icm()

            # ── Level 1: Diffusion training with rare-event injection ────
            n_rare_diff = int(BATCH_SIZE * RARE_BATCH_RATIO)    # 20% rare
            n_normal = BATCH_SIZE - n_rare_diff                  # 80% normal

            # Normal transitions from main buffer
            batch_idx = np.random.choice(len(idx), n_normal, replace=True)
            normal_trans = self.buffer.get_transitions(idx[batch_idx])
            normal_scores = scores_np[batch_idx]
            normal_weights = np.ones(n_normal)  # weight = 1.0

            # Inject rare transitions if available
            if len(self.rare_buffer) > 0:
                rare_trans = self.rare_buffer.get_transitions(min(n_rare_diff, len(self.rare_buffer)))
                if rare_trans is not None:
                    # Bug 5 fix: use scores_np.max() NOT * 2
                    # With * 2, hazards are mapped to unreachable prompt scores
                    rare_scores = np.ones(len(rare_trans)) * scores_np.max()
                    rare_weights = np.ones(len(rare_trans)) * RARE_WEIGHT
                    all_trans = np.vstack([normal_trans, rare_trans])
                    all_scores = np.concatenate([normal_scores, rare_scores])
                    all_weights = np.concatenate([normal_weights, rare_weights])
                else:
                    all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights
            else:
                all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights

            all_weights = all_weights / all_weights.mean()
            self._train_diffusion(all_trans, all_scores, all_weights)

            # ── Level 2: SAC batch = real + synthetic + rare ─────────────
            # Bug 4 fix: only generate synthetic after 2000 diffusion updates
            diffusion_ready = self.diffusion_updates > 2000
            n_syn = int(BATCH_SIZE * REPLAY_RATIO) if diffusion_ready else 0
            n_rare_sac = int(BATCH_SIZE * RARE_BATCH_RATIO) if len(self.rare_buffer) > 0 else 0
            n_real = BATCH_SIZE - n_syn - n_rare_sac

            real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(n_real)

            # Start building the batch components
            parts_s  = [real_s]
            parts_a  = [real_a]
            parts_r  = [real_r]
            parts_c  = [real_c]
            parts_ns = [real_ns]
            parts_d  = [real_d]

            if n_syn > 0:
                syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)
                parts_s.append(syn_s)
                parts_a.append(syn_a)
                parts_r.append(syn_r)
                parts_c.append(syn_c)
                parts_ns.append(syn_ns)
                parts_d.append(syn_d)

            # Inject rare transitions directly into SAC batch
            if n_rare_sac > 0:
                rare_sample = self.rare_buffer.sample(n_rare_sac)
                if rare_sample is not None:
                    rs, ra, rr, rc, rns, rd = rare_sample
                    parts_s.append(rs)
                    parts_a.append(ra)
                    parts_r.append(rr)
                    parts_c.append(rc)
                    parts_ns.append(rns)
                    parts_d.append(rd)

            states      = torch.cat(parts_s)
            actions     = torch.cat(parts_a)
            rewards     = torch.cat(parts_r)
            costs       = torch.cat(parts_c)
            next_states = torch.cat(parts_ns)
            dones       = torch.cat(parts_d)
        else:
            states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)

        self._sac_update(states, actions, rewards, costs, next_states, dones)

    def record_episode_cost(self, episode_cost):
        """
        Phantom cost fix: when the environment reports cost=0 but the rare
        buffer has data, inject a small phantom cost to keep λ from collapsing.

        Without this fix, during safe phases (no hazards), λ decays to 0.01
        because the agent sees zero cost. When hazards return, the agent has
        no safety penalty and spikes hard before λ can recover.

        The phantom cost is RARE_BATCH_RATIO * avg_rare_cost ≈ 0.2, which
        represents the cost the agent "would" see if it were replaying rare
        events during training (which it is, via the rare buffer).
        """
        if episode_cost == 0 and len(self.rare_buffer) > 0:
            # Compute average cost in rare buffer
            avg_rare_cost = np.mean([t['cost'] for t in self.rare_buffer.buffer])
            phantom_cost = RARE_BATCH_RATIO * avg_rare_cost
            super().record_episode_cost(phantom_cost)
        else:
            super().record_episode_cost(episode_cost)

# ═══════════════════════════════════════════════════════════════════════════════
# SAC + Memory  (ABLATION BASELINE)
#
# This answers the reviewer question: "Why not just replay rare events directly
# into SAC without a diffusion model?" If this agent matches PGR+Memory, our
# diffusion model isn't adding value. If it's worse (pessimism collapse) or
# unstable, that proves the diffusion model is necessary.
# ═══════════════════════════════════════════════════════════════════════════════

class SACMemoryAgent(SACAgent):
    """SAC + direct rare-event injection (no diffusion, no ICM).

    Ablation baseline: injects raw hazard transitions into the SAC batch.
    Tests whether the diffusion model is necessary, or if simple replay
    of rare events is sufficient.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        self.rare_buffer = RareEventBuffer(state_dim, action_dim)

    def add_transition(self, s, a, r, c, ns, d):
        self.buffer.add(s, a, r, c, ns, d)
        if c > 0:
            self.rare_buffer.add(s, a, r, c, ns, d)

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        n_rare = int(BATCH_SIZE * RARE_BATCH_RATIO) if len(self.rare_buffer) > 0 else 0
        n_real = BATCH_SIZE - n_rare

        real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(n_real)

        if n_rare > 0:
            rare_sample = self.rare_buffer.sample(n_rare)
            if rare_sample is not None:
                rs, ra, rr, rc, rns, rd = rare_sample
                states      = torch.cat([real_s, rs])
                actions     = torch.cat([real_a, ra])
                rewards     = torch.cat([real_r, rr])
                costs       = torch.cat([real_c, rc])
                next_states = torch.cat([real_ns, rns])
                dones       = torch.cat([real_d, rd])
            else:
                states, actions, rewards, costs, next_states, dones = (
                    real_s, real_a, real_r, real_c, real_ns, real_d
                )
        else:
            states, actions, rewards, costs, next_states, dones = (
                real_s, real_a, real_r, real_c, real_ns, real_d
            )

        self._sac_update(states, actions, rewards, costs, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# PGR+Memory++ — Exploratory Variant with 5 Enhancements
#
# This agent combines all 5 experimental changes on top of PGR+Memory:
#
#   Exp 2: Adaptive Rare Injection — scales injection rate based on
#          how long since the agent last encountered a real hazard.
#
#   Exp 3: Safety Critic — separate cost Q-network gives cleaner
#          safety gradient, trained heavily on rare buffer data.
#
#   Exp 4: Rare Data Augmentation — small Gaussian perturbations of
#          rare buffer states create richer hazard boundary signals.
#
#   Exp 5: Cosine Rare Weight Schedule — diffusion loss weight on
#          rare events increases over training as hazards become stale.
#
# Exp 1 (Two-Phase Environment) is handled at the environment level,
# not inside the agent — any agent can be tested with TwoPhaseWrapper.
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRMemoryPlusAgent(SACPGRAgent):
    """PGR+Memory with all 5 experimental enhancements."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        self.rare_buffer = RareEventBuffer(state_dim, action_dim)

        # ── Exp 2: Adaptive Rare Injection tracking ───────────────────
        self.episodes_since_hazard = 0  # counts episodes without cost>0
        self.total_train_episodes = 0   # total episodes seen

        # ── Exp 3: Safety Critic ──────────────────────────────────────
        # Two cost Q-networks (double-Q trick for cost, same as reward)
        self.qc1 = CostQNetwork(state_dim, action_dim).to(DEVICE)
        self.qc2 = CostQNetwork(state_dim, action_dim).to(DEVICE)
        self.qc1_target = CostQNetwork(state_dim, action_dim).to(DEVICE)
        self.qc2_target = CostQNetwork(state_dim, action_dim).to(DEVICE)
        self.qc1_target.load_state_dict(self.qc1.state_dict())
        self.qc2_target.load_state_dict(self.qc2.state_dict())
        self.qc1_opt = optim.Adam(self.qc1.parameters(), lr=SAFETY_CRITIC_LR)
        self.qc2_opt = optim.Adam(self.qc2.parameters(), lr=SAFETY_CRITIC_LR)

    def add_transition(self, s, a, r, c, ns, d):
        self.buffer.add(s, a, r, c, ns, d)
        if c > 0:
            self.rare_buffer.add(s, a, r, c, ns, d)
            self.episodes_since_hazard = 0  # reset staleness counter

    def record_episode_cost(self, episode_cost):
        """Track episode cost and update hazard staleness for adaptive injection."""
        super().record_episode_cost(episode_cost)
        self.total_train_episodes += 1
        if episode_cost == 0:
            self.episodes_since_hazard += 1
        else:
            self.episodes_since_hazard = 0

    # ── Exp 2: Adaptive rare batch ratio ──────────────────────────────

    def _get_adaptive_rare_ratio(self):
        """
        Scale rare injection rate based on hazard staleness.

        When the agent hasn't seen hazards in a while (episodes_since_hazard
        is high), inject more rare data to prevent forgetting. When hazards
        are fresh, inject less to avoid overwhelming the batch.

        Uses linear interpolation between ADAPTIVE_RARE_MIN and _MAX,
        clamped by ADAPTIVE_STALE_WINDOW.
        """
        staleness = min(self.episodes_since_hazard / max(ADAPTIVE_STALE_WINDOW, 1), 1.0)
        return ADAPTIVE_RARE_MIN + staleness * (ADAPTIVE_RARE_MAX - ADAPTIVE_RARE_MIN)

    # ── Exp 5: Cosine rare weight schedule ────────────────────────────

    def _get_scheduled_rare_weight(self):
        """
        Cosine-anneal the diffusion loss weight for rare events.

        Early training: weight = RARE_WEIGHT_MIN (hazards are fresh, less
        emphasis needed). Late training: weight = RARE_WEIGHT_MAX (hazards
        are stale in the main buffer, need stronger emphasis).

        Uses cosine schedule for smooth transition.
        """
        if self.diffusion_updates <= 0:
            return RARE_WEIGHT_MIN
        progress = min(self.diffusion_updates / 10000.0, 1.0)
        cos_val = 0.5 * (1.0 - np.cos(np.pi * progress))
        return RARE_WEIGHT_MIN + cos_val * (RARE_WEIGHT_MAX - RARE_WEIGHT_MIN)

    # ── Exp 4: Rare data augmentation ─────────────────────────────────

    def _augment_rare_transitions(self, rare_trans):
        """
        Create augmented copies of rare transitions by adding small Gaussian
        noise to state and next_state dimensions.

        This expands the effective hazard boundary the diffusion model learns.
        Only states and next_states are perturbed — actions, rewards, costs,
        and dones stay the same.
        """
        if rare_trans is None or len(rare_trans) == 0:
            return rare_trans

        sd = self.state_dim
        ad = self.action_dim
        augmented = [rare_trans]

        for _ in range(RARE_AUGMENT_FACTOR):
            noisy = rare_trans.copy()
            noisy[:, :sd] += np.random.normal(0, RARE_AUGMENT_NOISE, noisy[:, :sd].shape)
            ns_start = sd + ad + 2  # after s, a, r, c
            ns_end = ns_start + sd
            noisy[:, ns_start:ns_end] += np.random.normal(
                0, RARE_AUGMENT_NOISE, noisy[:, ns_start:ns_end].shape
            )
            augmented.append(noisy)

        return np.vstack(augmented)

    # ── Exp 3: Safety critic update ───────────────────────────────────

    def _update_safety_critic(self, states, actions, costs, next_states, dones):
        """
        Train the cost Q-networks on the current batch.

        Target: y_c = cost + γ * (1-done) * min(Qc1_target, Qc2_target)
        Loss: MSE(Qc(s,a), y_c)
        """
        with torch.no_grad():
            next_actions, _ = self.policy.sample(next_states)
            qc_target = torch.min(
                self.qc1_target(next_states, next_actions),
                self.qc2_target(next_states, next_actions),
            )
            target_c = costs.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * qc_target

        self.qc1_opt.zero_grad()
        F.mse_loss(self.qc1(states, actions), target_c).backward()
        self.qc1_opt.step()

        self.qc2_opt.zero_grad()
        F.mse_loss(self.qc2(states, actions), target_c).backward()
        self.qc2_opt.step()

        for tp, sp in zip(self.qc1_target.parameters(), self.qc1.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)
        for tp, sp in zip(self.qc2_target.parameters(), self.qc2.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

    def _sac_update(self, states, actions, rewards, costs, next_states, dones):
        """
        Extended SAC update with safety critic contribution to policy loss.

        The safety critic adds a direct cost gradient:
           policy_loss = α*log_π - Q_reward + λ_safety * Q_cost
        """
        effective_rewards = rewards - self.lam * costs

        # ── Critic update ─────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q_target = torch.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            )
            target = effective_rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (
                q_target - self.alpha * next_log_probs
            )

        self.q1_opt.zero_grad()
        F.mse_loss(self.q1(states, actions), target).backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        F.mse_loss(self.q2(states, actions), target).backward()
        self.q2_opt.step()

        # ── Safety critic update (Exp 3) ──────────────────────────────
        self._update_safety_critic(states, actions, costs, next_states, dones)

        # ── Actor update with safety penalty ──────────────────────────
        new_actions, log_probs = self.policy.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
        qc_new = torch.min(self.qc1(states, new_actions), self.qc2(states, new_actions))

        self.policy_opt.zero_grad()
        policy_loss = (
            self.alpha.detach() * log_probs
            - q_new
            + SAFETY_CRITIC_WEIGHT * self.lam * qc_new
        ).mean()
        policy_loss.backward()
        self.policy_opt.step()

        # ── Temperature update ────────────────────────────────────────
        self.alpha_opt.zero_grad()
        (-(self.log_alpha * (log_probs + self.target_entropy).detach())).mean().backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        self._soft_update()

    def train_step(self):
        """
        PGR+Memory++ training with all experimental enhancements:
        - Exp 2: adaptive rare injection ratio
        - Exp 4: augmented rare transitions for diffusion
        - Exp 5: cosine-scheduled rare weight
        - Exp 3: safety critic (inside _sac_update override)
        """
        if len(self.buffer) < BATCH_SIZE:
            return

        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            pool_trans = self.buffer.get_transitions(idx)
            self.trans_mean = torch.FloatTensor(pool_trans.mean(axis=0)).to(DEVICE)
            std = np.maximum(pool_trans.std(axis=0), 1e-2)
            cost_idx = self.state_dim + self.action_dim + 1
            std[cost_idx] = 1.0
            std[-1] = 1.0
            self.trans_std = torch.FloatTensor(std).to(DEVICE)

            self._train_icm()

            # ── Exp 2: Adaptive rare ratio for diffusion ──────────────
            rare_ratio = self._get_adaptive_rare_ratio()
            n_rare_diff = int(BATCH_SIZE * rare_ratio)
            n_normal = BATCH_SIZE - n_rare_diff

            batch_idx = np.random.choice(len(idx), n_normal, replace=True)
            normal_trans = self.buffer.get_transitions(idx[batch_idx])
            normal_scores = scores_np[batch_idx]
            normal_weights = np.ones(n_normal)

            if len(self.rare_buffer) > 0:
                rare_trans = self.rare_buffer.get_transitions(min(n_rare_diff, len(self.rare_buffer)))
                if rare_trans is not None:
                    # ── Exp 4: Augment rare transitions ───────────────
                    rare_trans = self._augment_rare_transitions(rare_trans)
                    rare_scores = np.ones(len(rare_trans)) * scores_np.max()
                    # ── Exp 5: Cosine rare weight schedule ────────────
                    scheduled_weight = self._get_scheduled_rare_weight()
                    rare_weights = np.ones(len(rare_trans)) * scheduled_weight
                    all_trans = np.vstack([normal_trans, rare_trans])
                    all_scores = np.concatenate([normal_scores, rare_scores])
                    all_weights = np.concatenate([normal_weights, rare_weights])
                else:
                    all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights
            else:
                all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights

            all_weights = all_weights / all_weights.mean()
            self._train_diffusion(all_trans, all_scores, all_weights)

            # ── SAC batch: real + synthetic + rare (adaptive ratio) ───
            diffusion_ready = self.diffusion_updates > 2000
            n_syn = int(BATCH_SIZE * REPLAY_RATIO) if diffusion_ready else 0
            n_rare_sac = int(BATCH_SIZE * rare_ratio) if len(self.rare_buffer) > 0 else 0
            n_real = BATCH_SIZE - n_syn - n_rare_sac

            real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(n_real)
            parts_s  = [real_s]
            parts_a  = [real_a]
            parts_r  = [real_r]
            parts_c  = [real_c]
            parts_ns = [real_ns]
            parts_d  = [real_d]

            if n_syn > 0:
                syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)
                parts_s.append(syn_s)
                parts_a.append(syn_a)
                parts_r.append(syn_r)
                parts_c.append(syn_c)
                parts_ns.append(syn_ns)
                parts_d.append(syn_d)

            if n_rare_sac > 0:
                rare_sample = self.rare_buffer.sample(n_rare_sac)
                if rare_sample is not None:
                    rs, ra, rr, rc, rns, rd = rare_sample
                    parts_s.append(rs)
                    parts_a.append(ra)
                    parts_r.append(rr)
                    parts_c.append(rc)
                    parts_ns.append(rns)
                    parts_d.append(rd)

            states      = torch.cat(parts_s)
            actions     = torch.cat(parts_a)
            rewards     = torch.cat(parts_r)
            costs       = torch.cat(parts_c)
            next_states = torch.cat(parts_ns)
            dones       = torch.cat(parts_d)
        else:
            states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)

        self._sac_update(states, actions, rewards, costs, next_states, dones)