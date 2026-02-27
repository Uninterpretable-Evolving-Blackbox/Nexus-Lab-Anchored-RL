"""
Environments with injected hazard signals.

Each env returns (obs, reward, cost, done, info) where cost > 0
indicates a hazard violation.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight 2D point environment (no MuJoCo needed)
# ═══════════════════════════════════════════════════════════════════════════════

class PointHazardEnv:
    """
    2D point agent navigating from start to goal while avoiding hazard zones.

    Layout ([-1,1] x [-1,1]):
      - Start:   bottom-left  (-0.8, -0.8)
      - Goal:    top-right    ( 0.8,  0.8)
      - Hazards: circles clustered along the direct diagonal path

    The agent MUST learn to navigate around hazards to reach the goal safely.
    Going straight through is fast but incurs heavy cost.

    State (6-dim): [x, y, vx, vy, dx_to_goal, dy_to_goal]
    Action (2-dim): [force_x, force_y] in [-1, 1]
    """

    state_dim = 6
    action_dim = 2

    def __init__(
        self,
        hazard_radius: float = 0.20,
        max_speed: float = 0.15,
        friction: float = 0.85,
        goal_bonus: float = 10.0,
        cost_value: float = 1.0,
        seed: int | None = None,
    ):
        self.goal = np.array([0.8, 0.8], dtype=np.float32)
        self.start = np.array([-0.8, -0.8], dtype=np.float32)
        self.hazard_radius = hazard_radius
        self.max_speed = max_speed
        self.friction = friction
        self.goal_bonus = goal_bonus
        self.goal_threshold = 0.15
        self.cost_value = cost_value

        # Hazards placed to block the direct diagonal path
        # Plus a few off-diagonal to punish wide detours too
        self.hazards = np.array([
            # Diagonal blockers (main challenge)
            [-0.35, -0.35],
            [-0.05, -0.05],
            [ 0.25,  0.25],
            [ 0.50,  0.50],
            # Upper-left cluster (blocks going up-then-right)
            [-0.30,  0.30],
            [ 0.00,  0.55],
            # Lower-right cluster (blocks going right-then-up)
            [ 0.30, -0.30],
            [ 0.55,  0.00],
        ], dtype=np.float32)

        # Stats
        self.total_cost = 0.0
        self.total_goals = 0
        self.episode_cost = 0.0

        self.pos = None
        self.vel = None

    def reset(self):
        """Reset agent to near start position with zero velocity. Small random
        offset prevents memorizing a single start state."""
        self.pos = self.start.copy() + np.random.uniform(-0.05, 0.05, 2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.episode_cost = 0.0
        return self._obs()

    def _obs(self):
        """Build 6-dim observation: [x, y, vx, vy, dx_to_goal, dy_to_goal].
        Including goal-relative vector gives the policy direct navigation info."""
        d = self.goal - self.pos
        return np.concatenate([self.pos, self.vel, d]).astype(np.float32)

    def _in_hazard(self, pos):
        """Check if pos is inside any hazard circle."""
        dists = np.linalg.norm(self.hazards - pos, axis=1)
        return np.any(dists < self.hazard_radius)

    def _nearest_hazard_dist(self, pos):
        dists = np.linalg.norm(self.hazards - pos, axis=1)
        return float(dists.min())

    def step(self, action):
        """
        Take one step. Physics: force → velocity (damped by friction) → position.
        Reward = -distance_to_goal (+bonus for reaching goal).
        Cost = 1.0 per step inside a hazard circle, 0.0 otherwise.
        The shortest path goes through hazards (fast but costly); safe paths
        must detour around them (slow but cheap). This is the core tradeoff.
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)

        # Simple point-mass dynamics:
        # action [-1,1] → scaled force → added to velocity
        force = action * 0.04                          # scale action to physical force
        self.vel = self.vel * self.friction + force    # apply friction, then add force
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:                     # cap speed to prevent flying
            self.vel = self.vel / speed * self.max_speed
        self.pos = np.clip(self.pos + self.vel, -1.0, 1.0)  # move, stay in arena

        # Reward: negative distance to goal (closer = higher reward = less negative)
        dist = np.linalg.norm(self.goal - self.pos)
        reward = -dist

        # Big bonus for reaching the goal
        done = False
        if dist < self.goal_threshold:
            reward += self.goal_bonus
            self.total_goals += 1
            done = True

        # Hazard cost: 1.0 if inside any hazard circle, 0.0 otherwise
        cost = 0.0
        if self._in_hazard(self.pos):
            cost = self.cost_value

        self.total_cost += cost
        self.episode_cost += cost

        info = {"dist_to_goal": dist, "in_hazard": cost > 0}
        return self._obs(), float(reward), float(cost), done, info

    def render_trajectory(self, trajectory: list[np.ndarray] | None = None):
        """
        Render the arena as a matplotlib figure. Shows:
          - Red circles: hazard zones
          - Blue square: start position
          - Gold star: goal position
          - Colored path: agent trajectory (green=safe, red=in hazard)

        Useful for debugging and for the final project presentation —
        you can visually see whether each agent learned to avoid hazards.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f5f5")
        ax.grid(True, alpha=0.2)

        # Arena boundary
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="black", linewidth=2))

        # Hazards
        for hx, hy in self.hazards:
            circle = Circle((hx, hy), self.hazard_radius, color="red", alpha=0.35)
            ax.add_patch(circle)
            circle_edge = Circle((hx, hy), self.hazard_radius, fill=False, edgecolor="darkred", linewidth=1.5)
            ax.add_patch(circle_edge)

        # Start & goal
        ax.plot(*self.start, "s", color="blue", markersize=14, label="Start", zorder=5)
        ax.plot(*self.goal, "*", color="gold", markersize=18, markeredgecolor="orange",
                markeredgewidth=1.5, label="Goal", zorder=5)

        # Trajectory
        if trajectory is not None and len(trajectory) > 1:
            traj = np.array(trajectory)
            # Color code: green=safe, red=in hazard
            for i in range(len(traj) - 1):
                in_hz = self._in_hazard(traj[i])
                color = "red" if in_hz else "#2ca02c"
                ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=color, linewidth=1.5, alpha=0.8)
            ax.plot(traj[0, 0], traj[0, 1], "o", color="blue", markersize=6, zorder=6)
            ax.plot(traj[-1, 0], traj[-1, 1], "o", color="purple", markersize=6, zorder=6)

        ax.legend(loc="upper left")
        ax.set_title("Point Hazard Environment")
        return fig

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# MuJoCo environments (require gymnasium[mujoco])
# ═══════════════════════════════════════════════════════════════════════════════

class HazardHalfCheetah:
    """
    HalfCheetah with three hazard sources:
      1. Velocity hazard  – forward velocity exceeds threshold
      2. Angle hazard     – joint angles exceed safe range
      3. Random gusts     – stochastic action perturbations
    """

    def __init__(
        self,
        velocity_threshold: float = 8.0,
        angle_threshold: float = 0.8,
        gust_probability: float = 0.05,
        cost_penalty: float = -1.0,
    ):
        try:
            import gymnasium as gym
        except ImportError:
            import gym

        self.env = gym.make("HalfCheetah-v4")
        self.state_dim = self.env.observation_space.shape[0]   # 17
        self.action_dim = self.env.action_space.shape[0]        # 6

        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
        self.gust_probability = gust_probability
        self.cost_penalty = cost_penalty

        # Lifetime stats
        self.total_cost = 0
        self.total_velocity_hazards = 0
        self.total_angle_hazards = 0
        self.episode_cost = 0

    def reset(self):
        self.episode_cost = 0
        out = self.env.reset()
        return out[0].astype(np.float32) if isinstance(out, tuple) else out.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)

        # Random gust
        if np.random.random() < self.gust_probability:
            gust = np.random.uniform(-0.5, 0.5, self.action_dim)
            action = np.clip(action + gust, -1, 1)

        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            done = term or trunc
        else:
            obs, reward, done, info = out
        obs = obs.astype(np.float32)

        cost = 0.0

        # Velocity hazard (obs[8] = forward velocity)
        if len(obs) > 8 and abs(obs[8]) > self.velocity_threshold:
            cost += 1.0
            self.total_velocity_hazards += 1

        # Angle hazard (obs[2:8] = joint angles)
        joint_angles = obs[2:8] if len(obs) > 8 else obs[2:min(8, len(obs))]
        if np.any(np.abs(joint_angles) > self.angle_threshold):
            cost += 1.0
            self.total_angle_hazards += 1

        self.total_cost += cost
        self.episode_cost += cost

        return obs, float(reward), float(cost), done, info

    def close(self):
        self.env.close()


class HazardAnt:
    """
    Ant with three hazard sources (tuned for RARE events):
      1. Height hazard       – torso drops very low (severe fall/collapse)
      2. Orientation hazard  – torso tilt is extreme (nearly flipped over)
      3. Velocity hazard     – excessive speed (reckless running)

    Thresholds are set so that a competent walking ant rarely triggers them,
    but occasional wind gusts or aggressive gaits can cause hazard events.
    This matches the experimental premise: rare but catastrophic events
    that the agent must remember how to avoid.

    Previous thresholds (height<0.3, tilt>0.5) triggered on ~25% of all
    timesteps — way too frequent for a "rare event" study. These new
    thresholds target ~5% trigger rate for a learning agent.
    """

    def __init__(
        self,
        height_threshold: float = 0.2,     # was 0.3 — only severe falls
        tilt_threshold: float = 0.85,       # was 0.5 — only near-flipping
        velocity_threshold: float = 5.0,    # now actually checked!
        gust_probability: float = 0.03,     # was 0.05 — rarer but stronger
        gust_strength: float = 1.0,         # was 0.5 — gusts pack more punch
    ):
        try:
            import gymnasium as gym
        except ImportError:
            import gym

        self.env = gym.make("Ant-v4")
        self.state_dim = self.env.observation_space.shape[0]   # 27
        self.action_dim = self.env.action_space.shape[0]        # 8

        self.height_threshold = height_threshold
        self.tilt_threshold = tilt_threshold
        self.velocity_threshold = velocity_threshold
        self.gust_probability = gust_probability
        self.gust_strength = gust_strength

        self.total_cost = 0
        self.episode_cost = 0

    def reset(self):
        self.episode_cost = 0
        out = self.env.reset()
        return out[0].astype(np.float32) if isinstance(out, tuple) else out.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)

        # Random wind gusts — rarer but stronger, simulating sudden
        # disturbances that a deployed robot might face
        if np.random.random() < self.gust_probability:
            gust = np.random.uniform(-self.gust_strength, self.gust_strength, self.action_dim)
            action = np.clip(action + gust, -1, 1)

        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            done = term or trunc
        else:
            obs, reward, done, info = out
        obs = obs.astype(np.float32)

        cost = 0.0

        # Hazard 1: severe fall — torso height drops dangerously low
        # Normal walking height is ~0.55, threshold 0.2 means nearly collapsed
        if obs[0] < self.height_threshold:
            cost += 1.0

        # Hazard 2: extreme tilt — quaternion xyz components too large
        # means the ant is nearly flipped over
        if len(obs) > 4:
            orientation = obs[1:5]
            if np.any(np.abs(orientation[1:]) > self.tilt_threshold):
                cost += 1.0

        # Hazard 3: excessive velocity — reckless speed
        # obs[13:19] are the 6 velocity components (3 linear + 3 angular)
        if len(obs) > 19:
            vel_magnitude = np.sqrt(np.sum(obs[13:19]**2))
            if vel_magnitude > self.velocity_threshold:
                cost += 1.0

        self.total_cost += cost
        self.episode_cost += cost

        return obs, float(reward), float(cost), done, info

    def close(self):
        self.env.close()