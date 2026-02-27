#!/usr/bin/env python3
"""
Visualise a trained agent.

Usage:
  python animate.py --env cheetah --method PGR+Memory --seed 42
  python animate.py --env cheetah --method SAC --seed 42 --save  # saves mp4
"""

import argparse
import numpy as np
import torch

from config import DEVICE
from agents import SACAgent, SACPGRAgent, SACPGRMemoryAgent, SACMemoryAgent

AGENT_REGISTRY = {"SAC": SACAgent, "SAC+Memory": SACMemoryAgent, "PGR": SACPGRAgent, "PGR+Memory": SACPGRMemoryAgent}

ENV_DIMS = {
    "cheetah": {"gym_id": "HalfCheetah-v4", "state_dim": 17, "action_dim": 6},
    "ant":     {"gym_id": "Ant-v4",          "state_dim": 27, "action_dim": 8},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_DIMS.keys()), default="cheetah")
    parser.add_argument("--method", choices=list(AGENT_REGISTRY.keys()), default="PGR+Memory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save", action="store_true", help="Save as mp4 instead of live window")
    args = parser.parse_args()

    info = ENV_DIMS[args.env]
    agent = AGENT_REGISTRY[args.method](info["state_dim"], info["action_dim"])

    # Load saved weights
    weight_file = f"weights_{args.env}_{args.method.replace('+', '_')}_{args.seed}.pt"
    try:
        checkpoint = torch.load(weight_file, map_location=DEVICE, weights_only=True)
        agent.policy.load_state_dict(checkpoint["policy"])
        print(f"Loaded weights from {weight_file}")
    except FileNotFoundError:
        print(f"ERROR: {weight_file} not found.")
        print(f"Run diagnostic.py first: python diagnostic.py --env {args.env} --seed {args.seed}")
        return

    agent.policy.eval()

    import gymnasium as gym

    if args.save:
        env = gym.make(info["gym_id"], render_mode="rgb_array")
        frames = []
    else:
        env = gym.make(info["gym_id"], render_mode="human")

    obs, _ = env.reset()
    total_reward = 0

    for step in range(args.steps):
        action = np.clip(agent.select_action(obs), -1, 1)
        action = np.clip(action, -1, 1)

        obs, reward, term, trunc, info_dict = env.step(action)
        total_reward += reward

        if args.save:
            frames.append(env.render())

        if term or trunc:
            print(f"Episode ended at step {step+1}, reward={total_reward:.1f}")
            total_reward = 0
            obs, _ = env.reset()

    env.close()

    if args.save:
        try:
            import imageio
            out_path = f"animation_{args.env}_{args.method.replace('+', '_')}_{args.seed}.mp4"
            imageio.mimsave(out_path, frames, fps=30)
            print(f"Saved -> {out_path}")
        except ImportError:
            print("pip install imageio[ffmpeg] to save videos")

    print("Done!")


if __name__ == "__main__":
    main()