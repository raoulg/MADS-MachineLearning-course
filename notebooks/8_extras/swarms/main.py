import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import OptimizeResult as ScipyOptimizeResult  # For typing hints

# Import from our new local files
from optimizers import PSO, PSOConfig, GWO, GWOConfig, OptimizationResult

from benchmarks import (
    sphere,
    rastrigin,
    rosenbrock,
    ackley,
    plot_2d_benchmark,
    plot_convergence_on_ax,
)


if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMIZATION ALGORITHM COMPARISON")
    print("=" * 70)

    # --- Setup ---
    outputdir = Path("./notebooks/8_extras/swarms/results")
    outputdir.mkdir(parents=True, exist_ok=True)

    dimension = 10
    max_iterations = 100
    n_agents = 50  # Common number of particles/wolves

    test_functions = {
        "Sphere": (sphere, (-5.0, 5.0)),
        "Rastrigin": (rastrigin, (-5.12, 5.12)),
        "Rosenbrock": (rosenbrock, (-2.0, 2.0)),
        "Ackley": (ackley, (-5.0, 5.0)),
    }

    # --- 1. Plot 2D versions of functions first ---
    print("\nGenerating 2D visualizations of benchmark functions...")
    fig_funcs = plt.figure(figsize=(14, 12))
    fig_funcs.suptitle("2D Benchmark Function Visualizations", fontsize=20, y=0.95)
    ax_map = {
        "Sphere": fig_funcs.add_subplot(2, 2, 1, projection="3d"),
        "Rastrigin": fig_funcs.add_subplot(2, 2, 2, projection="3d"),
        "Rosenbrock": fig_funcs.add_subplot(2, 2, 3, projection="3d"),
        "Ackley": fig_funcs.add_subplot(2, 2, 4, projection="3d"),
    }
    for func_name, (func, bounds) in test_functions.items():
        plot_2d_benchmark(ax_map[func_name], func, bounds, func_name)

    filepath_funcs = outputdir / "benchmark_functions_2d.png"
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filepath_funcs, dpi=150, bbox_inches="tight")
    plt.close(fig_funcs)
    print(f"Function visualization plot saved to {filepath_funcs}")

    # --- 2. Run optimizations and plot results in a grid ---
    fig_conv, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=False)
    axes_flat = axes.flat

    print("\nStarting algorithm comparison...")

    for ax, (func_name, (func, bounds)) in zip(axes_flat, test_functions.items()):
        print(f"\n{'=' * 70}")
        print(f"Problem: {func_name} Function (dimension={dimension})")
        print(f"{'=' * 70}")

        results_list = []
        scipy_bounds = [bounds] * dimension

        # 1. PSO
        pso_config = PSOConfig(
            n_particles=n_agents,
            max_iterations=max_iterations,
            omega=0.729,
            c1=1.49445,
            c2=1.49445,
            bounds=bounds,
        )
        print(f"\nRunning PSO...")
        pso = PSO(pso_config)
        pso_result = pso.optimize(func, dimension, verbose=False)
        results_list.append(pso_result)
        print(f"PSO Result: {pso_result}")

        # 2. GWO
        gwo_config = GWOConfig(
            n_wolves=n_agents, max_iterations=max_iterations, bounds=bounds
        )
        print(f"\nRunning GWO...")
        gwo = GWO(gwo_config)
        gwo_result = gwo.optimize(func, dimension, verbose=False)
        results_list.append(gwo_result)
        print(f"GWO Result: {gwo_result}")

        # 3. Differential Evolution (DE)
        print(f"\nRunning Differential Evolution (Scipy)...")
        de_history = []

        np.random.seed(42)
        initial_pop = [
            np.random.uniform(bounds[0], bounds[1], dimension) for _ in range(n_agents)
        ]
        de_history.append(func(np.mean(initial_pop, axis=0)))  # Approx start fitness

        def de_callback(xk, convergence):
            de_history.append(func(xk))

        de_result: ScipyOptimizeResult = differential_evolution(
            func,
            bounds=scipy_bounds,
            maxiter=max_iterations,
            popsize=int(n_agents / dimension) + 1,
            callback=de_callback,
            seed=42,
        )
        de_opt_result = OptimizationResult(
            best_position=de_result.x,
            best_fitness=de_result.fun,
            fitness_history=de_history,
            algorithm="Diff. Evolution",
            n_iterations=de_result.nit,
        )
        results_list.append(de_opt_result)
        print(f"DE Result: {de_opt_result}")

        # 4. L-BFGS-B
        print(f"\nRunning L-BFGS-B (Scipy)...")
        lbfgs_history = []
        np.random.seed(42)
        x0 = np.random.uniform(bounds[0], bounds[1], dimension)
        lbfgs_history.append(func(x0))

        def lbfgs_callback(xk):
            lbfgs_history.append(func(xk))

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
            print(f"L-BFGS-B Result: {lbfgs_opt_result}")
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")

        # Plot convergence for this function ON ITS ASSIGNED AXIS
        plot_convergence_on_ax(
            ax,
            results_list,
            title=f"Convergence on {func_name}",
            max_iter=max_iterations,
        )

    # --- Save the combined grid plot ---
    fig_conv.suptitle(
        f"Algorithm Convergence Comparison (Dimensions={dimension})",
        fontsize=22,
        y=1.03,
    )
    fig_conv.tight_layout(rect=[0, 0, 1, 0.97])

    filepath_conv = outputdir / "all_convergence_comparison_grid.png"
    plt.savefig(filepath_conv, dpi=150, bbox_inches="tight")
    plt.close(fig_conv)

    print(f"\nConvergence grid plot saved to {filepath_conv}")
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
