import numpy as np
from typing import Callable, Tuple, List
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from optimizers import OptimizationResult


def sphere(x: np.ndarray) -> float:
    """
    Sphere function: f(x) = Σ x_i²
    Global minimum: f(0,...,0) = 0
    """
    return np.sum(x**2)


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function: f(x) = 10n + Σ(x_i² - 10cos(2πx_i))
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]
    Global minimum: f(1,...,1) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x: np.ndarray) -> float:
    """
    Ackley function: highly multimodal
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def plot_2d_benchmark(
    ax: Axes3D, func: Callable, bounds: Tuple[float, float], title: str
):
    """Helper to plot a 2D version of a benchmark function on a 3D axis"""
    x_lin = np.linspace(bounds[0], bounds[1], 80)
    y_lin = np.linspace(bounds[0], bounds[1], 80)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Calculate Z
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # Plot
    ax.plot_surface(
        X, Y, Z, cmap="viridis", edgecolor="none", rstride=2, cstride=2, alpha=0.9
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")


def plot_convergence_on_ax(
    ax: Axes | Axes3D,
    results: List[OptimizationResult],
    title: str,
    max_iter: int,
):
    """Plot convergence curves for multiple results on a specific axes"""

    for result in results:
        history = result.fitness_history

        # Ensure history has max_iter + 1 elements (for iteration 0)
        if len(history) < (max_iter + 1):
            padding = [history[-1]] * (max_iter + 1 - len(history))
            history = history + padding

        # Trim if history is too long (e.g., from extra callbacks)
        history = history[: max_iter + 1]

        ax.plot(
            range(len(history)), history, label=result.algorithm, linewidth=2, alpha=0.9
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=10)
