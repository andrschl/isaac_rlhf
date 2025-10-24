#!/usr/bin/env python3
"""Plot Gridworld regret for vanilla vs ts_last (beta1=0.05, beta2=0.1)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib as mpl

mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {"family": "normal", "weight": "black", "size": 20}
mpl.rc("font", **font)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 200,
})
plt.rcParams["text.usetex"] = True
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}"
                              r"\usepackage{bm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl-gt-reward",
        type=float,
        default=3.28,
        help="Ground-truth reward achieved by the RL baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the figure (default: fig/gridworld_regret.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def collect_regret_curves(
    files: Iterable[Path],
    rl_gt_reward: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load logs and return (steps, regrets) where regrets has shape (runs, steps)."""
    files = sorted(Path(f) for f in files)
    if not files:
        raise FileNotFoundError("No log files found for the given configuration.")

    steps = None
    regrets = []

    for csv_path in files:
        df = pd.read_csv(csv_path)
        if "rlhf/gt_reward" not in df.columns or "step" not in df.columns:
            raise ValueError(f"Missing required columns in {csv_path}")

        run_steps = df["step"].to_numpy()
        if steps is None:
            steps = run_steps
        elif not np.array_equal(steps, run_steps):
            raise ValueError(f"Step mismatch across runs (first mismatch in {csv_path})")

        gt_rewards = df["rlhf/gt_reward"].to_numpy(dtype=float)
        per_step_regret = rl_gt_reward - gt_rewards
        regrets.append(np.cumsum(per_step_regret))

    return steps, np.vstack(regrets)


def quantile_plot(ax, x: np.ndarray, data: np.ndarray, label: str, color: str):
    """Plot the median with a 20-80 percentile ribbon."""
    medians = np.median(data, axis=0)
    lower_q = np.percentile(data, 20, axis=0)
    upper_q = np.percentile(data, 80, axis=0)

    ax.plot(x, medians, label=label, color=color, linewidth=2)
    ax.fill_between(x, lower_q, upper_q, color=color, alpha=0.25)


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    logs_root = project_root / "logs" / "rlhf" / "gridworld"

    vanilla_dir = logs_root / "vanilla"
    ts_last_dir = logs_root / "ts_last" / "beta1_0.05" / "beta2_0.1"

    vanilla_files = sorted(vanilla_dir.glob("seed_*/logs.csv"))
    ts_last_files = sorted(ts_last_dir.glob("seed_*/logs.csv"))

    if not vanilla_files:
        raise FileNotFoundError("No vanilla logs found at logs/rlhf/gridworld/vanilla/seed_*/logs.csv")
    if not ts_last_files:
        raise FileNotFoundError(
            "No ts_last logs found at logs/rlhf/gridworld/ts_last/beta1_0.05/beta2_0.1/seed_*/logs.csv"
        )

    vanilla_steps, vanilla_regrets = collect_regret_curves(vanilla_files, args.rl_gt_reward)
    ts_steps, ts_regrets = collect_regret_curves(ts_last_files, args.rl_gt_reward)

    if not np.array_equal(vanilla_steps, ts_steps):
        raise ValueError("vanilla and ts_last runs have different step schedules.")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    quantile_plot(ax, vanilla_steps, vanilla_regrets, label=r"\texttt{Entropy Exploration}", color="tab:blue")
    quantile_plot(ax, ts_steps, ts_regrets, label=r"\texttt{RPO-Regret}", color="tab:orange")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("RLHF step")
    ax.set_ylabel("Regret")
    ax.legend(frameon=False, handlelength=1.8)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
    fig.tight_layout(pad=0.4)

    output_path = args.output
    if output_path is None:
        output_path = project_root / "fig" / f"gridworld_regret_beta1_{0.05}_beta2_{0.1}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
