"""
Validation suite for normalized EVChargingEnv.
Checks API compliance, reproducibility, and physical target feasibility.
"""

import argparse
from gymnasium.utils.env_checker import check_env
import numpy as np

from ev_charging_env import EVChargingEnv


def check_api_compliance(data_path):
    print("[1/3] Gymnasium API compliance")
    env = EVChargingEnv(data_path, seed=0, split="train")
    check_env(env, skip_render_check=True)
    print("      PASSED\n")


def check_reproducibility(data_path, seed=123):
    print("[2/3] Reproducibility under fixed seed")
    trajectories = []
    for _ in range(2):
        env = EVChargingEnv(data_path, seed=seed, split="train")
        obs, info = env.reset(seed=seed)
        traj = [obs.copy()]
        rng = np.random.default_rng(0)
        terminated = False
        while not terminated:
            action = np.array([rng.uniform(0.0, 1.0)], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(obs.copy())
        trajectories.append(np.array(traj))

    identical = trajectories[0].shape == trajectories[1].shape and np.allclose(
        trajectories[0], trajectories[1]
    )
    print(f"      {'PASSED' if identical else 'FAILED'}\n")
    if not identical:
        raise AssertionError("Environment is not reproducible under a fixed seed.")


def check_feasibility(data_path, n_episodes=500, seed=42, tolerance=0.5, split="all"):
    print(f"[3/3] Feasibility under 100% power over {n_episodes} episodes ({split})")
    env = EVChargingEnv(data_path, seed=seed, split=split)
    reached = 0
    shortfalls = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            # Action 1.0 corresponds to 100% EV physical limit
            action = np.array([1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
        shortfall = info["target_soc"] - info["soc"]
        shortfalls.append(shortfall)
        if shortfall <= tolerance:
            reached += 1

    success_rate = 100 * reached / n_episodes
    print(f"      Reached target: {reached}/{n_episodes} = {success_rate:.1f}%")
    print(
        f"      Mean shortfall: {np.mean(shortfalls):.2f} pts | Max: {np.max(shortfalls):.2f} pts"
    )

    if success_rate < 95.0:
        raise AssertionError(f"Target unreachable: success rate {success_rate:.1f}%")
    print("      PASSED\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ev_charging_dataset.csv")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    check_api_compliance(args.data)
    check_reproducibility(args.data)
    check_feasibility(args.data, n_episodes=args.episodes, split="train")
    check_feasibility(args.data, n_episodes=args.episodes, split="test")

    print("All validation checks passed successfully.")