"""
Classical Optimization Solvers for Portfolio Selection
Provides exact ground-truth baselines (Brute Force, Quadratic/MILP) and
heuristic baselines (Simulated Annealing, Greedy Selection).
"""

import time
import itertools
from typing import Dict, Any, List
import numpy as np
from src.portfolio_qubo import PortfolioQUBO


class ExactBruteForceSolver:
    """
    Exhaustively searches all 2^n binary bitstrings or combinations of size B.
    Guarantees finding the global mathematical minimum.
    """

    def __init__(self, portfolio_model: PortfolioQUBO):
        self.model = portfolio_model

    def solve(self, valid_combinations_only: bool = True) -> Dict[str, Any]:
        start_time = time.perf_counter()
        n = self.model.n_assets
        b = self.model.budget

        best_cost = float("inf")
        best_x = None
        all_evaluations = []

        if valid_combinations_only:
            # Enumerate exactly combinations of size B
            indices = list(range(n))
            for combo in itertools.combinations(indices, b):
                x = np.zeros(n, dtype=int)
                x[list(combo)] = 1
                res = self.model.evaluate_solution(x)
                cost = res["markowitz_cost"]
                all_evaluations.append((x.copy(), cost))
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
        else:
            # Enumerate full 2^n state space
            for combo in itertools.product([0, 1], repeat=n):
                x = np.array(combo, dtype=int)
                res = self.model.evaluate_solution(x)
                cost = res["qubo_cost"]
                all_evaluations.append((x.copy(), cost))
                if cost < best_cost:
                    best_cost = cost
                    best_x = x

        elapsed_time = time.perf_counter() - start_time
        result = self.model.evaluate_solution(best_x)
        result["solver"] = "Classical Exact (Brute Force)"
        result["execution_time_sec"] = elapsed_time
        result["evaluated_states_count"] = len(all_evaluations)
        return result


class SimulatedAnnealingSolver:
    """
    Classical Simulated Annealing metaheuristic on the QUBO objective.
    Stochastically searches the combinatorial binary space with thermal cooling.
    """

    def __init__(
        self,
        portfolio_model: PortfolioQUBO,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.98,
        num_iterations: int = 1000,
        seed: int = 42
    ):
        self.model = portfolio_model
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.seed = seed

    def solve(self) -> Dict[str, Any]:
        np.random.seed(self.seed)
        start_time = time.perf_counter()
        n = self.model.n_assets
        b = self.model.budget

        # Initialize with random valid configuration
        indices = np.random.choice(n, size=b, replace=False)
        current_x = np.zeros(n, dtype=int)
        current_x[indices] = 1

        current_res = self.model.evaluate_solution(current_x)
        current_cost = current_res["qubo_cost"]

        best_x = current_x.copy()
        best_cost = current_cost
        temp = self.initial_temp

        for step in range(self.num_iterations):
            # Neighbor proposal: flip 1 or 2 bits
            neighbor_x = current_x.copy()
            flip_idx = np.random.choice(n)
            neighbor_x[flip_idx] = 1 - neighbor_x[flip_idx]

            # With some probability, swap a 1 and 0 to maintain budget
            if np.random.rand() > 0.5:
                ones = np.where(neighbor_x == 1)[0]
                zeros = np.where(neighbor_x == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    neighbor_x[np.random.choice(ones)] = 0
                    neighbor_x[np.random.choice(zeros)] = 1

            neighbor_res = self.model.evaluate_solution(neighbor_x)
            neighbor_cost = neighbor_res["qubo_cost"]

            delta_e = neighbor_cost - current_cost

            # Metropolis acceptance criterion
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(temp, 1e-6)):
                current_x = neighbor_x
                current_cost = neighbor_cost

                if current_cost < best_cost and neighbor_res["is_valid"]:
                    best_cost = current_cost
                    best_x = current_x.copy()

            temp *= self.cooling_rate

        elapsed_time = time.perf_counter() - start_time
        result = self.model.evaluate_solution(best_x)
        result["solver"] = "Classical Simulated Annealing"
        result["execution_time_sec"] = elapsed_time
        result["iterations"] = self.num_iterations
        return result


class GreedySharpeSolver:
    """
    Greedy heuristic: ranks assets by individual Sharpe ratio and picks top B assets.
    """

    def __init__(self, portfolio_model: PortfolioQUBO):
        self.model = portfolio_model

    def solve(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        n = self.model.n_assets
        b = self.model.budget

        individual_vols = np.sqrt(np.diag(self.model.covariances))
        # Sharpe ratio = (mu - r_f) / sigma
        r_f = 0.03
        individual_sharpes = (self.model.expected_returns - r_f) / np.maximum(individual_vols, 1e-4)

        top_indices = np.argsort(individual_sharpes)[::-1][:b]
        x = np.zeros(n, dtype=int)
        x[top_indices] = 1

        elapsed_time = time.perf_counter() - start_time
        result = self.model.evaluate_solution(x)
        result["solver"] = "Classical Greedy (Top Sharpe)"
        result["execution_time_sec"] = elapsed_time
        return result
