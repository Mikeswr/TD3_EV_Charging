"""
Evaluation harness: Compares Immediate Charging Baseline against TD3 models
across the test split.
"""

import argparse
import csv

import numpy as np
from stable_baselines3 import TD3

from ev_charging_env import EVChargingEnv


def run_episode(env, action_fn):
    obs, info = env.reset()

    total_reward = 0.0
    total_cost = 0.0
    total_peak = 0.0
    violations = 0
    actions_taken = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = action_fn(env, obs)
        raw_val = float(np.asarray(action).reshape(-1)[0])
        actions_taken.append(raw_val)

        # Check against unit bounds [0.0, 1.0]
        if raw_val < -1e-5 or raw_val > 1.0 + 1e-5:
            violations += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += -info["cost_term"]
        total_peak += -info["peak_term"]

    met = 1.0 if info["soc"] >= info["target_soc"] - 0.5 else 0.0

    return {
        "reward": total_reward,
        "cost": total_cost,
        "peak": total_peak,
        "met": met,
        "violations": violations,
        "mean_action": float(np.mean(actions_taken)),
    }


def immediate_charging_policy(env, obs):
    # Charge at 100% capacity until complete
    return np.array([1.0], dtype=np.float32)


def agent_policy(model):
    def _policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _policy


def evaluate_policy(data_path, action_fn, seeds, episodes_per_seed):
    per_seed = []
    for seed in seeds:
        env = EVChargingEnv(data_path, seed=seed, split="test")
        episodes = [run_episode(env, action_fn) for _ in range(episodes_per_seed)]

        per_seed.append(
            {
                "seed": seed,
                "cost": np.mean([e["cost"] for e in episodes]),
                "peak": np.mean([e["peak"] for e in episodes]),
                "met": 100.0 * np.mean([e["met"] for e in episodes]),
                "violations": int(np.sum([e["violations"] for e in episodes])),
                "reward": np.mean([e["reward"] for e in episodes]),
                "mean_action": np.mean([e["mean_action"] for e in episodes]),
            }
        )
    return per_seed


def summarize(per_seed, label):
    metrics = ["cost", "peak", "met", "violations", "reward", "mean_action"]
    print(f"\n{label} (n={len(per_seed)} seeds):")
    for metric in metrics:
        values = np.array([row[metric] for row in per_seed])
        print(f"  {metric:14s} mean={values.mean():8.3f}  std={values.std():8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ev_charging_dataset.csv")
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", default="evaluation_results.csv")
    args = parser.parse_args()

    if len(args.model) != len(args.seeds):
        raise ValueError("--model and --seeds must match in length.")

    # Baseline evaluation
    baseline_results = evaluate_policy(
        args.data, immediate_charging_policy, args.seeds, args.episodes
    )
    summarize(baseline_results, "Baseline (Immediate Charging)")

    # TD3 evaluation
    td3_results = []
    for model_path, seed in zip(args.model, args.seeds):
        model = TD3.load(model_path)
        rows = evaluate_policy(
            args.data, agent_policy(model), [seed], args.episodes
        )
        td3_results.extend(rows)
    summarize(td3_results, "Trained TD3 Agent")

    # Write CSV output
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "seed",
                "cost",
                "peak",
                "met",
                "violations",
                "reward",
                "mean_action",
            ]
        )
        for row in baseline_results:
            writer.writerow(
                [
                    "baseline",
                    row["seed"],
                    f"{row['cost']:.4f}",
                    f"{row['peak']:.4f}",
                    f"{row['met']:.2f}",
                    row["violations"],
                    f"{row['reward']:.4f}",
                    f"{row['mean_action']:.4f}",
                ]
            )
        for row in td3_results:
            writer.writerow(
                [
                    "td3",
                    row["seed"],
                    f"{row['cost']:.4f}",
                    f"{row['peak']:.4f}",
                    f"{row['met']:.2f}",
                    row["violations"],
                    f"{row['reward']:.4f}",
                    f"{row['mean_action']:.4f}",
                ]
            )

    print(f"\nFull evaluation metrics saved to {args.out}")


if __name__ == "__main__":
    main()