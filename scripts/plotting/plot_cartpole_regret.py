#!/usr/bin/env python3
"""Plot Isaac Cartpole regret comparisons for multiple algorithms."""

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

ALGORITHMS = [
    ("ts_last", r"\texttt{RPO-Regret}", "tab:orange"),
    ("ts_last_lazy", r"\texttt{LRPO-Regret}", "tab:blue"),
    ("ts_last_lazy_opt_design", r"\texttt{LRPO-OD-Regret}", "tab:green"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl-gt-reward",
        type=float,
        default=1.25,
        help="Ground-truth reward achieved by the RL baseline.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.001,
        help="Beta1 value used in the experiments.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.1,
        help="Beta2 value used in the experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the regret figure (default: fig/cartpole_regret_beta1_{beta1}_beta2_{beta2}.png).",
    )
    parser.add_argument(
        "--output-queries",
        type=Path,
        default=None,
        help="Where to save the preference query figure (default: fig/cartpole_num_queries_beta1_{beta1}_beta2_{beta2}.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots interactively after saving.",
    )
    return parser.parse_args()


def load_runs(
    files: Iterable[Path],
    rl_gt_reward: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load logs and return (steps, regrets, num_queries) arrays."""
    files = sorted(Path(f) for f in files)
    if not files:
        raise FileNotFoundError("No log files found for the given configuration.")

    steps = None
    regrets = []
    queries = []

    for csv_path in files:
        df = pd.read_csv(csv_path)
        required_cols = {"rlhf/gt_reward", "rlhf/num_queries", "step"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        run_steps = df["step"].to_numpy()
        if steps is None:
            steps = run_steps
        elif not np.array_equal(steps, run_steps):
            raise ValueError(f"Step mismatch across runs (first mismatch in {csv_path})")

        gt_rewards = df["rlhf/gt_reward"].to_numpy(dtype=float)
        per_step_regret = rl_gt_reward - gt_rewards
        regrets.append(np.cumsum(per_step_regret))
        queries.append(df["rlhf/num_queries"].to_numpy(dtype=float))

    return steps, np.vstack(regrets), np.vstack(queries)


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
    logs_root = project_root / "logs" / "rlhf" / "Isaac-Cartpole-v0"

    regret_fig, regret_ax = plt.subplots(figsize=(6.5, 4.0))
    queries_fig, queries_ax = plt.subplots(figsize=(6.5, 4.0))

    steps_reference = None

    beta1_dir = f"beta1_{args.beta1}"
    beta2_dir = f"beta2_{args.beta2}"

    for algorithm, label, color in ALGORITHMS:
        base_dir = logs_root / algorithm / beta1_dir / beta2_dir
        files = sorted(base_dir.glob("seed_*/logs.csv"))
        if not files:
            raise FileNotFoundError(f"No logs found under {base_dir / 'seed_*/logs.csv'}")

        steps, regrets, num_queries = load_runs(files, rl_gt_reward=args.rl_gt_reward)

        if steps_reference is None:
            steps_reference = steps
        elif not np.array_equal(steps_reference, steps):
            raise ValueError("Step mismatch across algorithms.")

        quantile_plot(regret_ax, steps, regrets, label=label, color=color)
        quantile_plot(queries_ax, steps, num_queries, label=label, color=color)

    for ax in (regret_ax, queries_ax):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6) if ax is regret_ax else None
        ax.set_xlabel("RLHF step")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
        ax.legend(frameon=False, handlelength=1.8)

    regret_ax.set_ylabel("Regret")
    queries_ax.set_ylabel("Number of preference queries")

    regret_fig.tight_layout(pad=0.4)
    queries_fig.tight_layout(pad=0.4)

    output_path = args.output
    if output_path is None:
        output_path = project_root / "fig" / f"cartpole_regret_beta1_{args.beta1}_beta2_{args.beta2}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regret_fig.savefig(output_path, dpi=300)

    output_queries = args.output_queries
    if output_queries is None:
        output_queries = project_root / "fig" / f"cartpole_num_queries_beta1_{args.beta1}_beta2_{args.beta2}.png"
    output_queries.parent.mkdir(parents=True, exist_ok=True)
    queries_fig.savefig(output_queries, dpi=300)

    if args.show:
        plt.show()
    else:
        plt.close(regret_fig)
        plt.close(queries_fig)


if __name__ == "__main__":
    main()
