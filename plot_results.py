"""
Plotting and reporting script for DSCD 614 Option TD3-1.
Generates:
  1. learning_curves.png (mean +/- std shaded reward during training)
  2. metrics_comparison.png (Baseline vs TD3 side-by-side)
  3. Formatted comparison table for report.
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curves(log_dir="logs", out_file="learning_curves.png", window=50):
    files = sorted(glob.glob(os.path.join(log_dir, "td3_seed*_episodes.csv")))
    if not files:
        print("No training logs found in logs/ directory.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    all_curves = []
    min_len = float("inf")

    for f in files:
        df = pd.read_csv(f)
        rolled = df["reward"].rolling(window=window, min_periods=1).mean().values
        all_curves.append(rolled)
        min_len = min(min_len, len(rolled))

    truncated = np.array([c[:min_len] for c in all_curves])
    episodes = np.arange(min_len)
    mean_r = np.mean(truncated, axis=0)
    std_r = np.std(truncated, axis=0)

    ax.plot(episodes, mean_r, color="#1f77b4", lw=2, label="TD3 Mean Reward")
    ax.fill_between(
        episodes,
        mean_r - std_r,
        mean_r + std_r,
        color="#1f77b4",
        alpha=0.25,
        label=r"$\pm 1\ \sigma$ across seeds",
    )

    ax.set_title("TD3 Training Convergence Across Seeds", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"Return ({window}-Episode Rolling Mean)", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_metrics_comparison(
    eval_csv="evaluation_results.csv", out_file="metrics_comparison.png"
):
    if not os.path.exists(eval_csv):
        print(f"Evaluation file {eval_csv} not found.")
        return

    df = pd.read_csv(eval_csv)
    summary = (
        df.groupby("policy")[["reward", "peak", "cost", "met"]]
        .agg(["mean", "std"])
        .reindex(["baseline", "td3"])
    )

    metrics = [
        ("reward", "Cumulative Reward"),
        ("peak", "Peak Contribution"),
        ("cost", "Charging Cost"),
        ("met", "Target Met (%)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=300)
    colors = ["#7f7f7f", "#2ca02c"]

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        means = [
            summary.loc["baseline", (key, "mean")],
            summary.loc["td3", (key, "mean")],
        ]
        stds = [
            summary.loc["baseline", (key, "std")],
            summary.loc["td3", (key, "std")],
        ]

        bars = ax.bar(
            ["Baseline", "TD3"],
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(
        "Performance Comparison: Immediate Baseline vs TD3",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY TABLE FOR REPORT")
    print("=" * 60)
    print(
        f"{'Metric':<20} | {'Baseline (Immediate)':<22} | {'TD3 Agent':<22}"
    )
    print("-" * 68)
    for key, title in metrics:
        b_m = summary.loc["baseline", (key, "mean")]
        b_s = summary.loc["baseline", (key, "std")]
        t_m = summary.loc["td3", (key, "mean")]
        t_s = summary.loc["td3", (key, "std")]
        print(f"{title:<20} | {b_m:7.3f} +/- {b_s:<10.3f} | {t_m:7.3f} +/- {t_s:<10.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    plot_learning_curves()
    plot_metrics_comparison()