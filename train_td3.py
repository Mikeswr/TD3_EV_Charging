"""
TD3 training script for EVChargingEnv (DSCD 614, Option TD3-1).

Runs one seeded training run and writes:
  - logs/td3_seed{N}_episodes.csv   (episode-level reward + length, for
                                      plotting mean/spread across seeds)
  - models/td3_seed{N}.zip          (the trained policy)

Run once per seed for the required protocol, e.g.:
    python train_td3.py --data ev_charging_dataset.csv --seed 0 --timesteps 100000
    python train_td3.py --data ev_charging_dataset.csv --seed 1 --timesteps 100000
    python train_td3.py --data ev_charging_dataset.csv --seed 2 --timesteps 100000

Hyperparameters below are Stable-Baselines3 TD3 defaults plus a modest
exploration noise -- a starting point, not a tuned final configuration.
Document any value you change from these defaults, per the exam's
requirement to state every hyperparameter that differs from library default.
"""

import argparse
import csv
import os

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from ev_charging_env import EVChargingEnv


def train(data_path, seed, total_timesteps, out_dir="."):
    log_dir = os.path.join(out_dir, "logs")
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = Monitor(EVChargingEnv(data_path, seed=seed, split="train"))

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * env.action_space.high
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, log_interval=50)

    model.save(os.path.join(model_dir, f"td3_seed{seed}"))

    rewards = env.get_episode_rewards()
    lengths = env.get_episode_lengths()
    log_path = os.path.join(log_dir, f"td3_seed{seed}_episodes.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length"])
        for i, (r, l) in enumerate(zip(rewards, lengths)):
            writer.writerow([i, r, l])

    print(f"seed {seed}: {len(rewards)} episodes logged to {log_path}")
    print(f"seed {seed}: model saved to {model_dir}/td3_seed{seed}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/user-data/uploads/ev_charging_dataset.csv")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    train(args.data, args.seed, args.timesteps, args.out_dir)
