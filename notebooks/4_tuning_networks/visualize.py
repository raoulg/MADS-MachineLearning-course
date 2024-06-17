from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from tqdm import tqdm

import plotly.express as px


def plot_timers(timer: Dict[str, float]) -> None:
    x = list(timer.keys())
    y = list(timer.values())
    sns.barplot(x=x, y=y)


def plot_grid(
    img: np.ndarray,
    filepath: Path,
    k: int = 3,
    figsize: Tuple = (10, 10),
    title: str = "",
) -> None:
    fig, axs = plt.subplots(k, k, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    axs = axs.ravel()
    for i in tqdm(range(k * k)):
        axs[i].imshow(img[i], cmap="gray")
        axs[i].axis("off")
    fig.savefig(filepath)
    logger.success(f"saved grid to {filepath}")


# Function to plot images
def plot_categories(
    images, class_names, figsize: Tuple = (16, 15), filepath: Optional[Path] = None
):
    fig, axes = plt.subplots(1, 11, figsize=figsize)
    axes = axes.flatten()

    # Plot an empty canvas
    ax = axes[0]
    dummy_array = np.array([[[0, 0, 0, 0]]], dtype="uint8")
    ax.set_title("reference")
    ax.set_axis_off()
    ax.imshow(dummy_array, interpolation="nearest")

    # Plot an image for every category
    for k, v in images.items():
        ax = axes[k + 1]
        ax.imshow(v, cmap=plt.cm.binary)
        ax.set_title(f"{class_names[k]}")
        ax.set_axis_off()

    if filepath is not None:
        fig.savefig(filepath)
        logger.success(f"saved grid to {filepath}")
    else:
        plt.tight_layout()
        plt.show()


def parallel_plot(analysis, columns: list[str]):
    plot = analysis.results_df
    p = plot[columns].reset_index()
    return px.parallel_coordinates(p, color="accuracy")
