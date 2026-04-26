"""
Generate the 3 reward-curve PNGs from training_log.csv.
Run locally or in Colab after training completes.

Usage:
    python plot_curves.py                          # reads training_log.csv
    python plot_curves.py --log smoke_test_log.csv # custom file
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROLL = 20  # rolling-average window
OUT_DIR = "."


def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["main_task_complete"] = df["main_task_complete"].astype(bool)
    df["side_task_attempted"] = df["side_task_attempted"].astype(bool)
    return df


def rolling(series: pd.Series, window: int = ROLL) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def annotate_phases(ax, n_total: int) -> None:
    """Add subtle before/after shading for 500-step training runs."""
    if n_total < 200:
        return
    ax.axvspan(0, n_total * 0.2, alpha=0.06, color="red", label="Exploring")
    ax.axvspan(n_total * 0.2, n_total * 0.6, alpha=0.06, color="yellow", label="Learning")
    ax.axvspan(n_total * 0.6, n_total, alpha=0.06, color="green", label="Converging")


def plot_curve1(df: pd.DataFrame, out_dir: str) -> str:
    """Main Task Completion Rate — should trend UP."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rate = df["main_task_complete"].astype(float)
    ax.plot(df["episode"], rolling(rate), color="#2196F3", linewidth=2, label="Rolling avg (20 ep)")
    ax.scatter(df["episode"], rate, alpha=0.15, s=8, color="#2196F3")

    annotate_phases(ax, len(df))
    ax.set_title("Main Task Completion Rate", fontsize=15, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.8, linestyle="--", color="green", alpha=0.5, label="Target (0.8)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "curve1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_curve2(df: pd.DataFrame, out_dir: str) -> str:
    """Side Task Attempt Rate in no_monitoring mode — should trend DOWN."""
    nm = df[df["mode"] == "no_monitoring"].copy()
    if nm.empty:
        print("Warning: no 'no_monitoring' episodes found — plotting all modes for curve2.")
        nm = df.copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    rate = nm["side_task_attempted"].astype(float)
    ax.plot(nm["episode"], rolling(rate), color="#F44336", linewidth=2, label="Rolling avg (20 ep)")
    ax.scatter(nm["episode"], rate, alpha=0.15, s=8, color="#F44336")

    annotate_phases(ax, len(nm))
    ax.set_title("Side Task Attempt Rate — No Monitoring Mode", fontsize=15, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Attempt rate")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.1, linestyle="--", color="green", alpha=0.5, label="Target (<0.1)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "curve2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_curve3(df: pd.DataFrame, out_dir: str) -> str:
    """Consistency Score Across All Modes — should trend UP."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"evaluation": "#4CAF50", "deployment": "#FF9800", "no_monitoring": "#9C27B0"}
    for mode, grp in df.groupby("mode"):
        score = rolling(grp["consistency_score"].astype(float))
        ax.plot(grp["episode"], score, linewidth=1.5, alpha=0.8,
                color=colors.get(mode, "gray"), label=mode)

    overall = rolling(df["consistency_score"].astype(float))
    ax.plot(df["episode"], overall, color="black", linewidth=2.5, linestyle="--", label="All modes")

    annotate_phases(ax, len(df))
    ax.set_title("Consistency Score Across All Modes", fontsize=15, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Consistency score (1 = no side-task)")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.9, linestyle="--", color="green", alpha=0.5, label="Target (0.9)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "curve3.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_all(log_path: str = "training_log.csv", out_dir: str = OUT_DIR) -> None:
    df = load_log(log_path)
    print(f"Loaded {len(df)} episodes from {log_path}")
    print(df[["mode", "main_task_complete", "side_task_attempted", "consistency_score"]].describe())

    plot_curve1(df, out_dir)
    plot_curve2(df, out_dir)
    plot_curve3(df, out_dir)
    print("\nAll 3 curves saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training_log.csv")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    plot_all(args.log, args.out)
