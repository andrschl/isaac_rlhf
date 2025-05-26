# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections import defaultdict

import GPUtil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_freest_gpu():
    """Get the GPU with the most free memory."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    return gpus[0].id


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

import wandb
from collections import defaultdict

def load_wandb_logs(project_name: str, run_id: str) -> dict[str, list]:
    """
    Load wandb logs for a specific run.

    Args:
        project_name: The name of the wandb project.
        run_id: The run ID to load logs from.

    Returns:
        A dictionary mapping metric names to their recorded values (as lists).
    """
    api = wandb.Api()
    run = api.run(f"{project_name}/{run_id}")

    # History: each row is a dict with metric names as keys
    history = run.history(samples=100000)  # increase samples if needed

    data = defaultdict(list)
    for row in history:
        for key, value in row.items():
            # filter out step/time/etc. that you might not care about
            if key not in ("_step", "_timestamp", "_runtime"):
                data[key].append(value)

    return dict(data)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    return data