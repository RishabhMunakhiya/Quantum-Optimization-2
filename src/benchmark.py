"""
Benchmarking Suite for Portfolio Optimization at Scale
Compares Classical vs Quantum Algorithms across multiple problem dimensions (n = 4, 6, 8, 10).
Measures accuracy, approximation ratios, constraint satisfaction, runtime, and quantum circuit scaling.
"""

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data_loader import PortfolioDataLoader
from src.portfolio_qubo import PortfolioQUBO
from src.classical_solvers import ExactBruteForceSolver, SimulatedAnnealingSolver, GreedySharpeSolver
from src.quantum_solvers import NumPyQuantumExactSolver, QAOAPortfolioSolver, SamplingVQESolver


class PortfolioBenchmarkSuite:
    """
    Orchestrates comparative benchmarking between classical and quantum solvers across scales.
    """

    def __init__(self, data_loader: PortfolioDataLoader = None, output_dir: str = None):
        self.loader = data_loader if data_loader is not None else PortfolioDataLoader()
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "outputs")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_single_comparison(
        self,
        num_assets: int = 4,
        budget: int = 2,
        risk_factor: float = 0.5,
        qaoa_reps: int = 1,
        qaoa_iters: int = 60,
        include_vqe: bool = True
    ) -> pd.DataFrame:
        """
        Runs all solvers on a single portfolio configuration and compiles a comparison table.
        """
        mu, sigma, _, tickers = self.loader.get_subset(num_assets)
        model = PortfolioQUBO(
            expected_returns=mu,
            covariances=sigma,
            budget=budget,
            risk_factor=risk_factor,
            tickers=tickers
        )

        results = []

        # 1. Classical Exact
        res_exact = ExactBruteForceSolver(model).solve()
        exact_cost = res_exact["markowitz_cost"]
        results.append(self._format_benchmark_row(res_exact, exact_cost))

        # 2. Classical Simulated Annealing
        res_sa = SimulatedAnnealingSolver(model, num_iterations=1200).solve()
        results.append(self._format_benchmark_row(res_sa, exact_cost))

        # 3. Classical Greedy
        res_greedy = GreedySharpeSolver(model).solve()
        results.append(self._format_benchmark_row(res_greedy, exact_cost))

        # 4. Quantum Exact (NumPy Diagonalization)
        res_q_exact = NumPyQuantumExactSolver(model).solve()
        results.append(self._format_benchmark_row(res_q_exact, exact_cost))

        # 5. Quantum QAOA (p=1)
        res_qaoa_p1 = QAOAPortfolioSolver(model, reps=1, max_iter=qaoa_iters).solve()
        results.append(self._format_benchmark_row(res_qaoa_p1, exact_cost))

        # 6. Quantum QAOA (p=2) if small enough
        if num_assets <= 8:
            res_qaoa_p2 = QAOAPortfolioSolver(model, reps=2, max_iter=qaoa_iters).solve()
            results.append(self._format_benchmark_row(res_qaoa_p2, exact_cost))

        # 7. Quantum VQE
        if include_vqe and num_assets <= 6:
            res_vqe = SamplingVQESolver(model, reps=1, max_iter=qaoa_iters).solve()
            results.append(self._format_benchmark_row(res_vqe, exact_cost))

        df_results = pd.DataFrame(results)
        
        # Save CSV and JSON
        csv_path = os.path.join(self.output_dir, f"benchmark_single_n{num_assets}_b{budget}.csv")
        df_results.to_csv(csv_path, index=False)
        return df_results

    def run_scaling_benchmark(
        self,
        asset_sizes: List[int] = None,
        risk_factor: float = 0.5,
        qaoa_iters: int = 50
    ) -> pd.DataFrame:
        """
        Runs the scaling experiment across different asset dimensions n to demonstrate
        how algorithms scale (Track 2: 'Smarter Choices at Scale').
        """
        if asset_sizes is None:
            asset_sizes = [4, 6, 8, 10]

        records = []
        for n in asset_sizes:
            budget = max(1, n // 2)
            print(f"--- Benchmarking scale: n = {n} assets, Budget = {budget} ---")

            mu, sigma, _, tickers = self.loader.get_subset(n)
            model = PortfolioQUBO(
                expected_returns=mu,
                covariances=sigma,
                budget=budget,
                risk_factor=risk_factor,
                tickers=tickers
            )

            # Classical Exact
            t0 = time.perf_counter()
            res_exact = ExactBruteForceSolver(model).solve()
            t_exact = time.perf_counter() - t0
            exact_cost = res_exact["markowitz_cost"]

            # Classical Simulated Annealing
            t0 = time.perf_counter()
            res_sa = SimulatedAnnealingSolver(model, num_iterations=1000).solve()
            t_sa = time.perf_counter() - t0

            # Quantum QAOA (p=1)
            t0 = time.perf_counter()
            res_qaoa = QAOAPortfolioSolver(model, reps=1, max_iter=qaoa_iters).solve()
            t_qaoa = time.perf_counter() - t0

            records.append({
                "num_assets": n,
                "budget": budget,
                "hilbert_space_dim": 2 ** n,
                "exact_cost": exact_cost,
                "exact_time_sec": t_exact,
                "sa_cost": res_sa["markowitz_cost"],
                "sa_time_sec": t_sa,
                "sa_valid": res_sa["is_valid"],
                "qaoa_cost": res_qaoa["markowitz_cost"],
                "qaoa_time_sec": t_qaoa,
                "qaoa_valid": res_qaoa["is_valid"],
                "qaoa_approx_ratio": abs(exact_cost / res_qaoa["markowitz_cost"]) if res_qaoa["markowitz_cost"] != 0 else 1.0,
                "qaoa_circuit_depth": res_qaoa.get("circuit_depth", 0),
                "qaoa_cnot_count": res_qaoa.get("cnot_count", 0),
                "qaoa_two_qubit_count": res_qaoa.get("two_qubit_gate_count", res_qaoa.get("cnot_count", 0))
            })

        df_scaling = pd.DataFrame(records)
        scaling_path = os.path.join(self.output_dir, "scaling_benchmark.csv")
        df_scaling.to_csv(scaling_path, index=False)
        print(f"Scaling benchmark saved to: {scaling_path}")
        return df_scaling

    def _format_benchmark_row(self, res: Dict[str, Any], exact_cost: float) -> Dict[str, Any]:
        cost = res["markowitz_cost"]
        # Approximation ratio = exact_cost / cost (accounting for negative min costs)
        if abs(cost) > 1e-6 and abs(exact_cost) > 1e-6:
            approx_ratio = min(1.0, abs(exact_cost / cost)) if cost <= exact_cost else min(1.0, abs(cost / exact_cost))
        else:
            approx_ratio = 1.0 if abs(cost - exact_cost) < 1e-4 else 0.0

        return {
            "Solver": res.get("solver", "Unknown"),
            "Selected Tickers": ", ".join(res["selected_tickers"]),
            "Objective Cost": round(cost, 5),
            "Expected Return (%)": round(res["expected_return"] * 100, 2),
            "Volatility (%)": round(res["volatility"] * 100, 2),
            "Sharpe Ratio": round(res["sharpe_ratio"], 3),
            "Valid Budget": "Yes" if res["is_valid"] else "No",
            "Violation": res["constraint_violation"],
            "Approx Ratio": round(approx_ratio, 4),
            "Time (s)": round(res["execution_time_sec"], 4),
            "Circuit Depth": res.get("circuit_depth", "-"),
            "CNOT Count": res.get("cnot_count", "-")
        }


if __name__ == "__main__":
    suite = PortfolioBenchmarkSuite()
    print("Running Single Comparison (n=4, B=2)...")
    df_single = suite.run_single_comparison(num_assets=4, budget=2)
    print(tabulate(df_single, headers="keys", tablefmt="grid"))
