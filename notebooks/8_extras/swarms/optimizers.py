import numpy as np
from typing import Callable, Tuple, List
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Shared Result Class
# --------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Results from optimization run"""

    best_position: np.ndarray
    best_fitness: float
    fitness_history: List[float]
    algorithm: str
    n_iterations: int

    def __repr__(self):
        return (
            f"OptimizationResult(algorithm={self.algorithm}, "
            f"best_fitness={self.best_fitness:.6f}, "
            f"n_iterations={self.n_iterations})"
        )


# --------------------------------------------------------------------------
# Particle Swarm Optimization (PSO) - (Your provided code)
# --------------------------------------------------------------------------


class PSOConfig(BaseModel):
    """Particle Swarm Optimization Configuration"""

    n_particles: int = Field(
        default=30, gt=0, description="Number of particles in swarm"
    )
    max_iterations: int = Field(default=100, gt=0, description="Maximum iterations")
    omega: float = Field(default=0.7, ge=0, le=1, description="Inertia weight")
    c1: float = Field(default=1.5, gt=0, description="Cognitive coefficient")
    c2: float = Field(default=1.5, gt=0, description="Social coefficient")
    bounds: Tuple[float, float] = Field(
        default=(-5.0, 5.0), description="Search space bounds"
    )

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v


class PSO:
    """
    Particle Swarm Optimization

    Mathematical formulation:
    v_i(t+1) = ω·v_i(t) + c1·r1·(p_i - x_i(t)) + c2·r2·(g - x_i(t))
    x_i(t+1) = x_i(t) + v_i(t+1)

    where:
    - ω: inertia weight
    - c1, c2: cognitive and social coefficients
    - r1, r2: random vectors ~ U(0,1)^d
    - p_i: personal best position
    - g: global best position
    """

    def __init__(self, config: PSOConfig) -> None:
        self.config = config
        self.objective = None
        self.positions = None
        self.velocities = None
        self.fitness = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = None
        self.fitness_history = []

    def _initialize_swarm(self, dimension: int) -> None:
        cfg = self.config
        self.positions = np.random.uniform(
            cfg.bounds[0], cfg.bounds[1], (cfg.n_particles, dimension)
        )
        velocity_range = abs(cfg.bounds[1] - cfg.bounds[0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range, velocity_range, (cfg.n_particles, dimension)
        )

    def _evaluate_fitness(self) -> None:
        self.fitness = np.array([self.objective(p) for p in self.positions])

    def _initialize_memory(self):
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = self.fitness.copy()
        global_best_idx = np.argmin(self.fitness)
        self.global_best_position = self.positions[global_best_idx].copy()
        self.global_best_fitness = self.fitness[global_best_idx]
        self.fitness_history = [self.global_best_fitness]

    def _update_velocities(self, dimension: int) -> None:
        cfg = self.config
        r1 = np.random.uniform(0, 1, (cfg.n_particles, dimension))
        r2 = np.random.uniform(0, 1, (cfg.n_particles, dimension))
        inertia = cfg.omega * self.velocities
        cognitive = cfg.c1 * r1 * (self.personal_best_positions - self.positions)
        social = cfg.c2 * r2 * (self.global_best_position - self.positions)
        self.velocities = inertia + cognitive + social

    def _update_positions(self) -> None:
        self.positions = self.positions + self.velocities
        self.positions = np.clip(
            self.positions, self.config.bounds[0], self.config.bounds[1]
        )

    def _update_memory(self):
        improved = self.fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = self.fitness[improved]
        min_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[min_idx] < self.global_best_fitness:
            self.global_best_position = self.personal_best_positions[min_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[min_idx]

    def _log_iteration(self, iteration: int, verbose: bool):
        self.fitness_history.append(self.global_best_fitness)
        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"PSO Iteration {iteration + 1}/{self.config.max_iterations}: "
                f"Best fitness = {self.global_best_fitness:.6f}"
            )

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        dimension: int,
        verbose: bool = True,
    ) -> OptimizationResult:
        self.objective = objective
        self._initialize_swarm(dimension)
        self._evaluate_fitness()
        self._initialize_memory()
        for iteration in range(self.config.max_iterations):
            self._update_velocities(dimension)
            self._update_positions()
            self._evaluate_fitness()
            self._update_memory()
            self._log_iteration(iteration, verbose)
        return OptimizationResult(
            best_position=self.global_best_position,
            best_fitness=self.global_best_fitness,
            fitness_history=self.fitness_history,
            algorithm="PSO",
            n_iterations=self.config.max_iterations,
        )


# --------------------------------------------------------------------------
# Grey Wolf Optimizer (GWO) - (New)
# --------------------------------------------------------------------------


class GWOConfig(BaseModel):
    """Grey Wolf Optimizer Configuration"""

    n_wolves: int = Field(default=30, gt=0, description="Number of wolves in the pack")
    max_iterations: int = Field(default=100, gt=0, description="Maximum iterations")
    bounds: Tuple[float, float] = Field(
        default=(-5.0, 5.0), description="Search space bounds"
    )

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v


class GWO:
    """
    Grey Wolf Optimizer (GWO)
    Vectorized implementation
    """

    def __init__(self, config: GWOConfig) -> None:
        self.config = config
        self.objective = None
        self.positions = None
        self.fitness = None
        self.alpha_pos = None
        self.alpha_fitness = float("inf")
        self.beta_pos = None
        self.beta_fitness = float("inf")
        self.delta_pos = None
        self.delta_fitness = float("inf")
        self.fitness_history = []

    def _initialize_pack(self, dimension: int) -> None:
        """Initialize wolf positions"""
        cfg = self.config
        self.positions = np.random.uniform(
            cfg.bounds[0], cfg.bounds[1], (cfg.n_wolves, dimension)
        )

    def _evaluate_and_update_hierarchy(self) -> None:
        """Evaluate fitness and update Alpha, Beta, Delta"""
        self.fitness = np.array([self.objective(p) for p in self.positions])

        # Build lists of all *valid* positions and fitness scores
        # Start with the current pack
        all_positions_list = [self.positions]
        all_fitness_list = [self.fitness]

        # Add previous leaders *if they exist* (i.e., not None)
        if self.alpha_pos is not None:
            # Ensure leaders are 2D (e.g., (1, 10)) for vstack
            all_positions_list.append(np.atleast_2d(self.alpha_pos))
            all_fitness_list.append(np.atleast_1d(self.alpha_fitness))

        if self.beta_pos is not None:
            all_positions_list.append(np.atleast_2d(self.beta_pos))
            all_fitness_list.append(np.atleast_1d(self.beta_fitness))

        if self.delta_pos is not None:
            all_positions_list.append(np.atleast_2d(self.delta_pos))
            all_fitness_list.append(np.atleast_1d(self.delta_fitness))

        # Combine current wolves with previous leaders to ensure elitism
        all_positions = np.vstack(all_positions_list)
        all_fitness = np.concatenate(all_fitness_list)

        # Remove any 'inf' entries from initialization (should only happen on first run)
        valid_indices = np.isfinite(all_fitness)
        all_positions = all_positions[valid_indices]
        all_fitness = all_fitness[valid_indices]

        # Get unique solutions
        _, unique_indices = np.unique(all_positions, axis=0, return_index=True)
        unique_positions = all_positions[unique_indices]
        unique_fitness = all_fitness[unique_indices]

        # Sort by fitness
        sorted_indices = np.argsort(unique_fitness)

        # Update leaders
        self.alpha_pos = unique_positions[sorted_indices[0]].copy()  # Shape (10,)
        self.alpha_fitness = unique_fitness[sorted_indices[0]]

        # Ensure we have at least 3 unique solutions to pick from
        num_unique = len(sorted_indices)

        self.beta_pos = unique_positions[sorted_indices[min(1, num_unique - 1)]].copy()
        self.beta_fitness = unique_fitness[sorted_indices[min(1, num_unique - 1)]]

        self.delta_pos = unique_positions[sorted_indices[min(2, num_unique - 1)]].copy()
        self.delta_fitness = unique_fitness[sorted_indices[min(2, num_unique - 1)]]

    def _update_positions(self, iteration: int, dimension: int) -> None:
        """Update wolf positions based on Alpha, Beta, and Delta"""
        cfg = self.config

        # Linearly decrease 'a' from 2 to 0
        a = 2 - iteration * (2 / cfg.max_iterations)

        # --- Alpha Influence ---
        r1_alpha = np.random.rand(cfg.n_wolves, dimension)
        r2_alpha = np.random.rand(cfg.n_wolves, dimension)
        A1 = 2 * a * r1_alpha - a
        C1 = 2 * r2_alpha
        D_alpha = np.abs(C1 * self.alpha_pos - self.positions)
        X1 = self.alpha_pos - A1 * D_alpha

        # --- Beta Influence ---
        r1_beta = np.random.rand(cfg.n_wolves, dimension)
        r2_beta = np.random.rand(cfg.n_wolves, dimension)
        A2 = 2 * a * r1_beta - a
        C2 = 2 * r2_beta
        D_beta = np.abs(C2 * self.beta_pos - self.positions)
        X2 = self.beta_pos - A2 * D_beta

        # --- Delta Influence ---
        r1_delta = np.random.rand(cfg.n_wolves, dimension)
        r2_delta = np.random.rand(cfg.n_wolves, dimension)
        A3 = 2 * a * r1_delta - a
        C3 = 2 * r2_delta
        D_delta = np.abs(C3 * self.delta_pos - self.positions)
        X3 = self.delta_pos - A3 * D_delta

        # --- New Position (average of influences) ---
        self.positions = (X1 + X2 + X3) / 3

        # Apply boundary constraints
        self.positions = np.clip(self.positions, cfg.bounds[0], cfg.bounds[1])

    def _log_iteration(self, iteration: int, verbose: bool):
        """Log current iteration state"""
        self.fitness_history.append(self.alpha_fitness)  # Log the best-so-far

        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"GWO Iteration {iteration + 1}/{self.config.max_iterations}: "
                f"Best (Alpha) fitness = {self.alpha_fitness:.6f}"
            )

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        dimension: int,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run GWO optimization"""
        self.objective = objective

        # 1. Initialization
        self._initialize_pack(dimension)

        # Evaluate and find initial leaders
        self._evaluate_and_update_hierarchy()

        # Initialize history with the first Alpha
        self.fitness_history = [self.alpha_fitness]

        # 2. Main optimization loop
        for iteration in range(self.config.max_iterations):
            # Update positions based on leaders
            self._update_positions(iteration, dimension)

            # Evaluate new positions and update leaders (with elitism)
            self._evaluate_and_update_hierarchy()

            # Log
            self._log_iteration(iteration, verbose)

        # 3. Return results
        return OptimizationResult(
            best_position=self.alpha_pos,
            best_fitness=self.alpha_fitness,
            fitness_history=self.fitness_history,
            algorithm="GWO",
            n_iterations=self.config.max_iterations,
        )
