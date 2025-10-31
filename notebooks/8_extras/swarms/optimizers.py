import numpy as np
from typing import Callable, Tuple, List
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass
from loguru import logger
from tqdm import trange


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
        self.objective: Callable[[np.ndarray], float] | None = None
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
        self.personal_best_positions: np.ndarray | None = None
        self.personal_best_fitness: np.ndarray | None = None
        self.global_best_position: np.ndarray | None = None
        # FIX: Initialize fitness to infinity (a float) instead of None.
        # This resolves many type errors related to 'None' being in fitness lists
        # or assigned to 'best_fitness' which expects a float.
        self.global_best_fitness: float = np.inf
        self.fitness_history: List[float] = []

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
        assert self.objective is not None, "Objective function not set"
        assert self.positions is not None, "Swarm positions not initialized"
        self.fitness = np.array([self.objective(p) for p in self.positions])

    def _initialize_memory(self):
        assert self.positions is not None, "Swarm positions not initialized"
        assert self.fitness is not None, "Swarm fitness not evaluated"
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = self.fitness.copy()
        global_best_idx = np.argmin(self.fitness)
        self.global_best_position = self.positions[global_best_idx].copy()
        self.global_best_fitness = self.fitness[global_best_idx]
        self.fitness_history = [self.global_best_fitness]

    def _update_velocities(self, dimension: int) -> None:
        """
        Update particle velocities using PSO formula

        v_i(t+1) = ω·v_i(t) + c1·r1·(p_i - x_i) + c2·r2·(g - x_i)
        """

        cfg = self.config
        r1 = np.random.uniform(0, 1, (cfg.n_particles, dimension))
        r2 = np.random.uniform(0, 1, (cfg.n_particles, dimension))

        assert self.velocities is not None
        assert self.personal_best_positions is not None
        assert self.positions is not None
        assert self.global_best_position is not None

        inertia = cfg.omega * self.velocities
        cognitive = cfg.c1 * r1 * (self.personal_best_positions - self.positions)
        social = cfg.c2 * r2 * (self.global_best_position - self.positions)
        self.velocities = inertia + cognitive + social

    def _update_positions(self) -> None:
        assert self.positions is not None
        assert self.velocities is not None
        self.positions = self.positions + self.velocities
        self.positions = np.clip(
            self.positions, self.config.bounds[0], self.config.bounds[1]
        )

    def _update_memory(self):
        assert self.fitness is not None
        assert self.personal_best_fitness is not None
        assert self.personal_best_positions is not None
        assert self.positions is not None
        # self.global_best_fitness is now a float, so no check needed here.

        improved = self.fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = self.fitness[improved]
        min_idx = np.argmin(self.personal_best_fitness)

        # self.global_best_fitness is guaranteed to be float (from __init__ or _initialize_memory)
        if self.personal_best_fitness[min_idx] < self.global_best_fitness:
            self.global_best_position = self.personal_best_positions[min_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[min_idx]

    def _log_iteration(self, iteration: int, verbose: bool):
        """Log current iteration state"""
        # This append is now type-safe as self.global_best_fitness is always float
        self.fitness_history.append(self.global_best_fitness)
        # Log to debug level; this won't clutter the console unless DEBUG is enabled
        if verbose and (iteration + 1) % 20 == 0:
            logger.debug(
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

        # Use trange for an automatically-handled progress bar
        pbar_desc = f"PSO ({self.config.n_particles} particles)"
        for iteration in trange(
            self.config.max_iterations,
            desc=pbar_desc,
            leave=False,
            ncols=100,
            unit="iter",
        ):
            self._update_velocities(dimension)
            self._update_positions()
            self._evaluate_fitness()
            self._update_memory()
            self._log_iteration(iteration, verbose)

        assert self.global_best_position is not None, "Global best position was not set"

        return OptimizationResult(
            best_position=self.global_best_position,
            best_fitness=self.global_best_fitness,
            fitness_history=self.fitness_history,
            algorithm="PSO",
            n_iterations=self.config.max_iterations,
        )


# --------------------------------------------------------------------------
# Grey Wolf Optimizer (GWO)
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
    """Grey Wolf Optimizer (GWO) - Vectorized implementation"""

    def __init__(self, config: GWOConfig) -> None:
        self.config = config
        self.objective: Callable[[np.ndarray], float] | None = None
        self.positions: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
        self.alpha_pos: np.ndarray | None = None
        self.alpha_fitness: float = float("inf")
        self.beta_pos: np.ndarray | None = None
        self.beta_fitness: float = float("inf")
        self.delta_pos: np.ndarray | None = None
        self.delta_fitness: float = float("inf")
        self.fitness_history: List[float] = []

    def _initialize_pack(self, dimension: int) -> None:
        cfg = self.config
        self.positions = np.random.uniform(
            cfg.bounds[0], cfg.bounds[1], (cfg.n_wolves, dimension)
        )

    def _evaluate_and_update_hierarchy(self) -> None:
        """Evaluate fitness and update Alpha, Beta, Delta (Robust version)"""
        assert self.objective is not None, "Objective function not set"
        assert self.positions is not None, "Pack positions not initialized"
        self.fitness = np.array([self.objective(p) for p in self.positions])

        # This assertion helps with the vstack error later
        assert self.fitness is not None
        all_positions_list = [self.positions]
        all_fitness_list = [self.fitness]

        if self.alpha_pos is not None:
            all_positions_list.append(np.atleast_2d(self.alpha_pos))
            all_fitness_list.append(np.atleast_1d(self.alpha_fitness))
        if self.beta_pos is not None:
            all_positions_list.append(np.atleast_2d(self.beta_pos))
            all_fitness_list.append(np.atleast_1d(self.beta_fitness))
        if self.delta_pos is not None:
            all_positions_list.append(np.atleast_2d(self.delta_pos))
            all_fitness_list.append(np.atleast_1d(self.delta_fitness))

        # The logic is correct, self.positions is asserted and others are checked
        all_positions = np.vstack(all_positions_list)
        all_fitness = np.concatenate(all_fitness_list)

        valid_indices = np.isfinite(all_fitness)
        all_positions = all_positions[valid_indices]
        all_fitness = all_fitness[valid_indices]

        _, unique_indices = np.unique(all_positions, axis=0, return_index=True)
        unique_positions = all_positions[unique_indices]
        unique_fitness = all_fitness[unique_indices]

        sorted_indices = np.argsort(unique_fitness)

        self.alpha_pos = unique_positions[sorted_indices[0]].copy()
        self.alpha_fitness = float(unique_fitness[sorted_indices[0]])

        num_unique = len(sorted_indices)
        self.beta_pos = unique_positions[sorted_indices[min(1, num_unique - 1)]].copy()
        self.beta_fitness = float(
            unique_fitness[sorted_indices[min(1, num_unique - 1)]]
        )

        self.delta_pos = unique_positions[sorted_indices[min(2, num_unique - 1)]].copy()
        self.delta_fitness = float(
            unique_fitness[sorted_indices[min(2, num_unique - 1)]]
        )

    def _update_positions(self, iteration: int, dimension: int) -> None:
        cfg = self.config
        a = 2 - iteration * (2 / cfg.max_iterations)

        assert self.alpha_pos is not None
        assert self.beta_pos is not None
        assert self.delta_pos is not None
        assert self.positions is not None

        r1_alpha = np.random.rand(cfg.n_wolves, dimension)
        r2_alpha = np.random.rand(cfg.n_wolves, dimension)
        A1 = 2 * a * r1_alpha - a
        C1 = 2 * r2_alpha
        D_alpha = np.abs(C1 * self.alpha_pos - self.positions)
        X1 = self.alpha_pos - A1 * D_alpha

        r1_beta = np.random.rand(cfg.n_wolves, dimension)
        r2_beta = np.random.rand(cfg.n_wolves, dimension)
        A2 = 2 * a * r1_beta - a
        C2 = 2 * r2_beta
        D_beta = np.abs(C2 * self.beta_pos - self.positions)
        X2 = self.beta_pos - A2 * D_beta

        r1_delta = np.random.rand(cfg.n_wolves, dimension)
        r2_delta = np.random.rand(cfg.n_wolves, dimension)
        A3 = 2 * a * r1_delta - a
        C3 = 2 * r2_delta
        D_delta = np.abs(C3 * self.delta_pos - self.positions)
        X3 = self.delta_pos - A3 * D_delta

        self.positions = (X1 + X2 + X3) / 3

        self.positions = np.clip(self.positions, cfg.bounds[0], cfg.bounds[1])

    def _log_iteration(self, iteration: int, verbose: bool):
        self.fitness_history.append(self.alpha_fitness)
        if verbose and (iteration + 1) % 20 == 0:
            logger.debug(
                f"GWO Iteration {iteration + 1}/{self.config.max_iterations}: "
                f"Best (Alpha) fitness = {self.alpha_fitness:.6f}"
            )

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        dimension: int,
        verbose: bool = True,
    ) -> OptimizationResult:
        self.objective = objective
        self._initialize_pack(dimension)
        self._evaluate_and_update_hierarchy()
        self.fitness_history = [self.alpha_fitness]

        pbar_desc = f"GWO ({self.config.n_wolves} wolves)"
        for iteration in trange(
            self.config.max_iterations,
            desc=pbar_desc,
            leave=False,
            ncols=100,
            unit="iter",
        ):
            self._update_positions(iteration, dimension)
            self._evaluate_and_update_hierarchy()
            self._log_iteration(iteration, verbose)

        assert self.alpha_pos is not None, "Alpha position was not set"

        return OptimizationResult(
            best_position=self.alpha_pos,
            best_fitness=self.alpha_fitness,
            fitness_history=self.fitness_history,
            algorithm="GWO",
            n_iterations=self.config.max_iterations,
        )
