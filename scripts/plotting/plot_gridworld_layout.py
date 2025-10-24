#!/usr/bin/env python3
"""Visualize the 6x6 Gridworld MDP used in the experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Match paper styling from other plotting utilities
font = {"family": "normal", "weight": "black", "size": 20}
mpl.rc("font", **font)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
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
        "--output",
        type=Path,
        default=None,
        help="Where to save the figure (default: fig/gridworld_layout.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def build_layout() -> dict[str, np.ndarray]:
    params = {
        "grid_height": 6,
        "grid_width": 6,
        "noise": 0.0,
        "gamma": 0.9,
    }
    height = params["grid_height"]
    width = params["grid_width"]

    feature_ids = [0, 5, 18, 29, 32, 33]
    reward_states = {5: 0.5, 33: 0.5}
    start_state = 14

    grid = np.zeros((height, width), dtype=int)
    for state in feature_ids:
        row, col = divmod(state, width)
        grid[row, col] = 1  # feature cell

    start_row, start_col = divmod(start_state, width)
    grid[start_row, start_col] = 2  # start cell highlight

    return {
        "grid": grid,
        "reward_states": reward_states,
        "feature_ids": feature_ids,
        "start_state": start_state,
        "params": params,
    }


def plot_layout(grid_info: dict[str, np.ndarray], output_path: Path, show: bool) -> None:
    grid = grid_info["grid"]
    reward_states = grid_info["reward_states"]
    feature_ids = set(grid_info["feature_ids"])
    start_state = grid_info["start_state"]

    height, width = grid.shape
    state_numbers = np.arange(height * width).reshape(height, width)

    cmap = mpl.colors.ListedColormap(["white", "#e0ecff", "#8fd19e"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")

    # Grid lines for clarity
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # add custom grid lines without overlapping axes
    for x in range(width + 1):
        ax.axvline(x - 0.5, color="black", linewidth=1.2, alpha=0.8)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color="black", linewidth=1.2, alpha=0.8)
    border = Rectangle(
        (-0.5, -0.5),
        width,
        height,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        alpha=0.9,
    )
    ax.add_patch(border)

    # Annotate each cell with its state index and reward info if applicable
    for row in range(height):
        for col in range(width):
            state = state_numbers[row, col]
            label_lines = []
            if state == start_state:
                label_lines.append(r"\texttt{start}")
            if state in reward_states:
                reward_val = reward_states[state]
                label_lines.append(rf"$r={reward_val:.1f}$")
            if not label_lines:
                continue
            text = "\n".join(label_lines)
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

    ax.set_xlabel(r"Gridworld environment", labelpad=12)
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
    fig.tight_layout(pad=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    output_path = args.output
    if output_path is None:
        output_path = project_root / "fig" / "gridworld_layout.png"

    layout = build_layout()
    plot_layout(layout, output_path=output_path, show=args.show)


if __name__ == "__main__":
    main()
