"""
Seeded TD3 Training Script for EVChargingEnv.
Writes episode logs to logs/ and model checkpoints to models/.
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

    raw_env = EVChargingEnv(data_path, seed=seed, split="train")
    env = Monitor(raw_env)

    # 10% gaussian noise on unit action space [0.0, 1.0]
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.10 * np.ones(n_actions, dtype=np.float32),
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        seed=seed,
        verbose=1,
    )

    print(f"\n--- Starting TD3 Training (Seed {seed}) for {total_timesteps} steps ---")
    model.learn(total_timesteps=total_timesteps, log_interval=50)

    model_path = os.path.join(model_dir, f"td3_seed{seed}")
    model.save(model_path)

    rewards = env.get_episode_rewards()
    lengths = env.get_episode_lengths()
    log_path = os.path.join(log_dir, f"td3_seed{seed}_episodes.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length"])
        for i, (r, l) in enumerate(zip(rewards, lengths)):
            writer.writerow([i, r, l])

    print(f"Seed {seed} finished. Logged: {log_path} | Saved: {model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ev_charging_dataset.csv")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    train(args.data, args.seed, args.timesteps, args.out_dir)