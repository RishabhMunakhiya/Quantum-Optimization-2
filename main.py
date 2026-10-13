"""
Main Execution Script for Quantum Portfolio Optimization
Track 2: Quantum Optimization - Smarter Choices at Scale
Qiskit Fall Fest 2026 x IIIT Delhi
"""

import argparse
import sys
import os

# Ensure safe UTF-8 terminal printing across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data_loader import PortfolioDataLoader
from src.portfolio_qubo import PortfolioQUBO
from src.classical_solvers import ExactBruteForceSolver, SimulatedAnnealingSolver, GreedySharpeSolver
from src.quantum_solvers import NumPyQuantumExactSolver, QAOAPortfolioSolver, SamplingVQESolver
from src.benchmark import PortfolioBenchmarkSuite
from src.visualizer import PortfolioVisualizer


def run_pipeline(
    num_assets: int = 4,
    budget: int = 2,
    risk_factor: float = 0.5,
    qaoa_reps: int = 1,
    qaoa_iters: int = 60,
    run_scaling: bool = False
):
    print("=" * 75)
    print("  QUANTUM PORTFOLIO OPTIMIZER: SMARTER CHOICES AT SCALE")
    print("  Qiskit Fall Fest 2026 x IIIT Delhi | Track 2: Quantum Optimization")
    print("=" * 75)

    loader = PortfolioDataLoader()
    visualizer = PortfolioVisualizer()
    suite = PortfolioBenchmarkSuite(data_loader=loader)

    print(f"\n[1/5] Loading asset market data...")
    mu, sigma, corr, tickers = loader.get_subset(num_assets)
    print(f" -> Selected {num_assets} candidate tickers: {tickers}")
    print(f" -> Investment Budget (Stocks to Pick): B = {budget}")
    print(f" -> Risk Aversion Factor: q = {risk_factor}")

    # Generate Correlation Heatmap
    heatmap_file = visualizer.plot_correlation_heatmap(corr, tickers)
    print(f" -> Saved Correlation Matrix Heatmap to: {heatmap_file}")

    print(f"\n[2/5] Formulating Constrained Markowitz Problem as QUBO & Ising Hamiltonian...")
    model = PortfolioQUBO(
        expected_returns=mu,
        covariances=sigma,
        budget=budget,
        risk_factor=risk_factor,
        tickers=tickers
    )
    qp = model.build_quadratic_program()
    qubo = model.to_qubo()
    operator, offset = model.to_ising()

    print(f" -> Quadratic Variables: {qp.get_num_vars()} binary qubits")
    print(f" -> Linear Constraints: sum(x_i) == {budget} (Budget constraint)")
    print(f" -> Quadratic Penalty Multiplier A: {model.penalty:.2f}")
    print(f" -> Ising Pauli Terms: {len(operator)} terms (SparsePauliOp)")
    print(f" -> Ising Energy Offset: {offset:.4f}")

    print(f"\n[3/5] Executing Classical and Quantum Solvers...")
    df_results = suite.run_single_comparison(
        num_assets=num_assets,
        budget=budget,
        risk_factor=risk_factor,
        qaoa_reps=qaoa_reps,
        qaoa_iters=qaoa_iters
    )

    print("\n" + "=" * 75)
    print("  SOLVER COMPARISON RESULTS TABLE")
    print("=" * 75)
    print(tabulate(df_results, headers="keys", tablefmt="grid", showindex=False))

    # Save solver comparison plot
    solver_chart = visualizer.plot_solver_comparison(df_results)
    print(f"\n -> Saved Solver Comparison Plot to: {solver_chart}")

    # Run single detailed QAOA for energy history
    print(f"\n[4/5] Profiling QAOA Parameter Landscape & Convergence...")
    qaoa_detailed = QAOAPortfolioSolver(model, reps=qaoa_reps, max_iter=qaoa_iters).solve()
    if qaoa_detailed.get("energy_history"):
        conv_chart = visualizer.plot_qaoa_convergence(qaoa_detailed["energy_history"])
        print(f" -> Saved QAOA Convergence Trace to: {conv_chart}")

    # Overlay on Efficient Frontier
    exact_res = ExactBruteForceSolver(model).solve()
    exact_indices = np.where(np.array(exact_res["binary_vector"]) == 1)[0].tolist()
    qaoa_indices = np.where(np.array(qaoa_detailed["binary_vector"]) == 1)[0].tolist()
    frontier_chart = visualizer.plot_efficient_frontier(
        mu=mu,
        sigma=sigma,
        tickers=tickers,
        optimal_selections={"Classical Exact": exact_indices, "QAOA": qaoa_indices},
        budget=budget
    )
    print(f" -> Saved Efficient Frontier Plot to: {frontier_chart}")

    # Scaling Benchmark
    if run_scaling:
        print(f"\n[5/5] Running Track 2 'Smarter Choices at Scale' Benchmark Suite (n = 4, 6, 8, 10)...")
        df_scaling = suite.run_scaling_benchmark(asset_sizes=[4, 6, 8, 10], risk_factor=risk_factor)
        print("\n" + "=" * 75)
        print("  SCALING BENCHMARK SUMMARY")
        print("=" * 75)
        print(tabulate(df_scaling, headers="keys", tablefmt="grid", showindex=False))
        scaling_chart = visualizer.plot_scaling_curves(df_scaling)
        print(f" -> Saved Scaling Benchmark Curves to: {scaling_chart}")
    else:
        print("\n[5/5] Tip: Pass '--scale' to run the multi-asset scaling benchmark (n=4, 6, 8, 10).")

    print("\n" + "=" * 75)
    print("  OPTIMIZATION COMPLETE. ALL ARTIFACTS SAVED IN 'outputs/' DIRECTORY.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Quantum Portfolio Optimizer - Qiskit Fall Fest 2026")
    parser.add_argument("--assets", type=int, default=4, help="Number of candidate assets (default: 4)")
    parser.add_argument("--budget", type=int, default=2, help="Target portfolio budget / number of stocks (default: 2)")
    parser.add_argument("--risk", type=float, default=0.5, help="Risk aversion factor q (default: 0.5)")
    parser.add_argument("--reps", type=int, default=1, help="QAOA circuit layer depth p (default: 1)")
    parser.add_argument("--iters", type=int, default=60, help="Optimizer max iterations (default: 60)")
    parser.add_argument("--scale", action="store_true", help="Run the multi-scale scaling benchmark (n = 4, 6, 8, 10)")

    args = parser.parse_args()
    run_pipeline(
        num_assets=args.assets,
        budget=args.budget,
        risk_factor=args.risk,
        qaoa_reps=args.reps,
        qaoa_iters=args.iters,
        run_scaling=args.scale
    )


if __name__ == "__main__":
    main()
