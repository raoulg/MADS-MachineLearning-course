import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from benchmarks import (
    ackley,
    plot_2d_benchmark,
    plot_convergence_on_ax,
    rastrigin,
    rosenbrock,
    sphere,
)
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
from optimizers import GWO, PSO, GWOConfig, OptimizationResult, PSOConfig
from scipy.optimize import OptimizeResult as ScipyOptimizeResult
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm


class TqdmSink:
    def write(self, message):
        # Write message to stdout, ensuring it's handled by tqdm
        tqdm.write(message.strip(), file=sys.stdout)


logger.remove()
logger.add(
    TqdmSink(),
    # format="<level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("OPTIMIZATION ALGORITHM COMPARISON")
    logger.info("=" * 70)

    # --- Setup ---
    outputdir = Path("./notebooks/8_extras/swarms/results")
    if not outputdir.exists():
        outputdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {outputdir.resolve()}")

    dimension = 10
    max_iterations = 50
    n_agents = 100

    test_functions = {
        "Sphere": (sphere, (-5.0, 5.0)),
        "Rastrigin": (rastrigin, (-5.0, 5.0)),
        "Rosenbrock": (rosenbrock, (-2.0, 2.0)),
        "Ackley": (ackley, (-5.0, 5.0)),
    }

    # --- 1. Plot 2D versions of functions ---
    logger.info("Generating 2D visualizations of benchmark functions...")
    fig_funcs = plt.figure(figsize=(14, 12))
    fig_funcs.suptitle("2D Benchmark Function Visualizations", fontsize=20, y=0.95)
    ax_map: dict[str, Axes3D] = {
        "Sphere": cast(Axes3D, fig_funcs.add_subplot(2, 2, 1, projection="3d")),
        "Rastrigin": cast(Axes3D, fig_funcs.add_subplot(2, 2, 2, projection="3d")),
        "Rosenbrock": cast(Axes3D, fig_funcs.add_subplot(2, 2, 3, projection="3d")),
        "Ackley": cast(Axes3D, fig_funcs.add_subplot(2, 2, 4, projection="3d")),
    }

    for func_name, (func, bounds) in test_functions.items():
        plot_2d_benchmark(ax_map[func_name], func, bounds, func_name)

    filepath_funcs = outputdir / "benchmark_functions_2d.png"
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(filepath_funcs, dpi=150, bbox_inches="tight")
    plt.close(fig_funcs)
    logger.info(f"Function visualization plot saved to {filepath_funcs}")

    # --- 2. Run optimizations and plot results in a grid ---
    fig_conv, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=False)
    axes_flat = axes.flat

    logger.info("Starting algorithm comparison...")

    # Wrap the main loop in tqdm for an overall progress bar
    main_pbar_desc = "Overall Progress"
    main_loop = tqdm(
        zip(axes_flat, test_functions.items()),
        total=len(test_functions),
        desc=main_pbar_desc,
        ncols=100,
        unit="func",
    )

    for ax, (func_name, (func, bounds)) in main_loop:
        logger.info(f"--- Starting Problem: {func_name} (Dim={dimension}) ---")
        main_loop.set_description(f"Running: {func_name}")

        results_list = []
        scipy_bounds = [bounds] * dimension

        # 1. PSO
        pso_config = PSOConfig(
            n_particles=n_agents,
            max_iterations=max_iterations,
            omega=0.8,
            c1=1.5,
            c2=1.5,
            bounds=bounds,
        )
        logger.info("Running PSO...")
        pso = PSO(pso_config)
        pso_result = pso.optimize(func, dimension, verbose=True)
        results_list.append(pso_result)
        logger.success(f"PSO Result: {pso_result}")

        # 2. GWO
        gwo_config = GWOConfig(
            n_wolves=n_agents, max_iterations=max_iterations, bounds=bounds
        )
        logger.info("Running GWO...")
        gwo = GWO(gwo_config)
        gwo_result = gwo.optimize(func, dimension, verbose=True)
        results_list.append(gwo_result)
        logger.success(f"GWO Result: {gwo_result}")

        # 3. Differential Evolution (DE)
        logger.info("Running Differential Evolution (Scipy)...")
        de_history = []

        np.random.seed(42)
        initial_pop = [
            np.random.uniform(bounds[0], bounds[1], dimension) for _ in range(n_agents)
        ]
        de_history.append(func(np.mean(initial_pop, axis=0)))

        def de_callback(xk, convergence):
            de_history.append(func(xk))
            logger.debug(
                f"DE iter: fitness={de_history[-1]:.4f}, convergence={convergence:.4f}"
            )

        de_result: ScipyOptimizeResult = differential_evolution(
            func,
            bounds=scipy_bounds,
            maxiter=max_iterations,
            popsize=int(n_agents / dimension) + 1,
            callback=de_callback,
        )
        de_opt_result = OptimizationResult(
            best_position=de_result.x,
            best_fitness=de_result.fun,
            fitness_history=de_history,
            algorithm="Diff. Evolution",
            n_iterations=de_result.nit,
        )
        results_list.append(de_opt_result)
        logger.success(f"DE Result: {de_opt_result}")

        # 4. L-BFGS-B
        logger.info("Running L-BFGS-B (Scipy)...")
        lbfgs_history = []
        np.random.seed(42)
        x0 = np.random.uniform(bounds[0], bounds[1], dimension)
        lbfgs_history.append(func(x0))

        def lbfgs_callback(xk):
            lbfgs_history.append(func(xk))
            logger.debug(f"L-BFGS-B iter: fitness={lbfgs_history[-1]:.4f}")

        try:
            lbfgs_result: ScipyOptimizeResult = minimize(
                func,
                x0,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": max_iterations, "maxfun": max_iterations * 2},
                callback=lbfgs_callback,
            )
            lbfgs_opt_result = OptimizationResult(
                best_position=lbfgs_result.x,
                best_fitness=lbfgs_result.fun,
                fitness_history=lbfgs_history,
                algorithm="L-BFGS-B",
                n_iterations=lbfgs_result.nit,
            )
            results_list.append(lbfgs_opt_result)
            logger.success(f"L-BFGS-B Result: {lbfgs_opt_result}")
        except Exception as e:
            logger.error(f"L-BFGS-B failed: {e}")

        # Plot convergence for this function ON ITS ASSIGNED AXIS
        plot_convergence_on_ax(
            ax,
            results_list,
            title=f"Convergence on {func_name}",
            max_iter=max_iterations,
        )
        logger.info(f"--- Finished Problem: {func_name} ---")

    # --- Save the combined grid plot ---
    fig_conv.suptitle(
        f"Algorithm Convergence Comparison (Dimensions={dimension})",
        fontsize=22,
        y=1.03,
    )
    fig_conv.tight_layout(rect=(0, 0, 1, 0.97))

    filepath_conv = outputdir / "convergence_comparison_grid.png"
    plt.savefig(filepath_conv, dpi=150, bbox_inches="tight")
    plt.close(fig_conv)

    logger.info(f"Convergence grid plot saved to {filepath_conv}")
    logger.info("=" * 70)
    logger.success("DEMONSTRATION COMPLETE")
    logger.info("=" * 70)
