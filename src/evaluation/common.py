""" Common utility functions for data processing and visualization. """

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 
import random


def save_plot(filename: str, dir_name: str=None):
    """ Saves the current plot to a file."""
    if not filename.endswith('.png'):
        filename += '.png'
    filepath = dir_name / filename
    if not dir_name.exists():
        raise FileNotFoundError(f"Save directory {dir_name} does not exist.")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')  # bbox_inches='tight' to avoid clipping
    print(f"Saved to dir: {dir_name}")


def fix_random_seed(seed: int):
    """Fix random seed across numpy and random modules."""
    np.random.seed(seed)
    random.seed(seed)
    print(f"Fixed random seed: {seed}")



# =========================================================================================================
# Validation functions
# =========================================================================================================

def validate_metric(metric):
    assert isinstance(metric, str), f"metric argument must be a string!"
    metric = metric.lower()
    assert metric in ("mae", "mse", "rmse"), f"Invalid metric: '{metric}'"
    return metric


def validate_dir(directory):
    assert isinstance(directory, str) or isinstance(directory, Path), f"directory be a string or a pathlib.Path!"
    directory = Path(directory).absolute()
    assert directory.exists(), f"directory does not exist: '{directory}'"
    return directory


def validate_step(step: int, n_samples: int) -> int:
    assert isinstance(step, int), "step argument must be an integer!"
    assert step > 0, "step must be greater than zero!"
    assert step <= n_samples, f"step ({step}) must be less than or equal to the number of samples ({n_samples})!"
    return step


def validate_min_remaining(min_remaining: int, n_samples: int) -> int:
    assert isinstance(min_remaining, int), "min_remaining argument must be an integer!"
    assert min_remaining > 0, "min_remaining must be greater than zero!"
    assert min_remaining <= n_samples, f"min_remaining ({min_remaining}) must be less than or equal to the number of samples ({n_samples})!"
    return min_remaining
