"""
Publication-Quality Visualization Suite for Quantum Portfolio Optimization
Generates asset correlation heatmaps, solver comparison charts, QAOA convergence traces,
efficient frontier scatterplots, and quantum circuit scaling curves.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless file generation
import matplotlib.pyplot as plt
import seaborn as sns


class PortfolioVisualizer:
    """
    Renders and exports high-resolution charts for the project report and dashboard.
    """

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "outputs")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Clean aesthetic style
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    def plot_correlation_heatmap(self, correlations: np.ndarray, tickers: List[str], filename: str = "correlation_heatmap.png") -> str:
        """Plots the asset return correlation matrix."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.heatmap(
            correlations,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=tickers,
            yticklabels=tickers,
            cbar_kws={"label": "Correlation Coefficient"},
            ax=ax
        )
        ax.set_title("Asset Return Correlation Matrix (Markowitz Input)", fontsize=14, weight="bold", pad=12)
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_solver_comparison(self, df_results: pd.DataFrame, filename: str = "solver_comparison.png") -> str:
        """Compares Objective Cost, Sharpe Ratio, and Runtimes across solvers."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        solvers = df_results["Solver"]
        costs = df_results["Objective Cost"]
        sharpes = df_results["Sharpe Ratio"]

        colors = ["#2b5c8f" if "Quantum" in s else "#d95f02" for s in solvers]

        # Objective Cost (Lower is better)
        bars1 = ax1.barh(solvers, costs, color=colors, alpha=0.85, edgecolor="black")
        ax1.set_title("Markowitz Objective Cost (Lower is Better)", fontsize=12, weight="bold")
        ax1.set_xlabel("Objective Value")
        ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        for bar in bars1:
            w = bar.get_width()
            offset = -0.05 if w < 0 else 0.01
            ax1.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="right" if w < 0 else "left", fontsize=9)

        # Sharpe Ratio (Higher is better)
        bars2 = ax2.barh(solvers, sharpes, color=colors, alpha=0.85, edgecolor="black")
        ax2.set_title("Portfolio Sharpe Ratio (Risk-Adjusted Return)", fontsize=12, weight="bold")
        ax2.set_xlabel("Sharpe Ratio")
        for bar in bars2:
            w = bar.get_width()
            ax2.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9)

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_qaoa_convergence(self, energy_history: List[tuple], filename: str = "qaoa_convergence.png") -> str:
        """Plots QAOA expectation energy minimization trajectory over iterations."""
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        iters = [x[0] for x in energy_history]
        energies = [x[1] for x in energy_history]

        ax.plot(iters, energies, marker="o", markersize=4, color="#1f77b4", linewidth=2, label="QAOA Energy <H>")
        ax.set_title("QAOA Classical Optimization Convergence Trajectory", fontsize=13, weight="bold")
        ax.set_xlabel("Optimizer Function Evaluations (COBYLA)")
        ax.set_ylabel("Hamiltonian Expectation Energy")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_efficient_frontier(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        tickers: List[str],
        optimal_selections: Dict[str, List[int]],
        budget: int,
        filename: str = "efficient_frontier.png"
    ) -> str:
        """Plots Monte Carlo portfolio simulations and overlays the chosen quantum/classical portfolios."""
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        n = len(mu)
        np.random.seed(42)

        # Monte Carlo random combinations of size budget
        sim_returns, sim_vols = [], []
        import itertools
        all_combos = list(itertools.combinations(range(n), budget))
        for combo in all_combos:
            w = np.zeros(n)
            w[list(combo)] = 1.0 / budget
            r = np.dot(w, mu)
            v = np.sqrt(max(1e-8, w.T @ sigma @ w))
            sim_returns.append(r * 100)
            sim_vols.append(v * 100)

        scatter = ax.scatter(sim_vols, sim_returns, c=np.array(sim_returns)/np.maximum(sim_vols, 1e-4),
                             cmap="viridis", alpha=0.7, s=40, label="Valid Portfolios (B Assets)")
        plt.colorbar(scatter, ax=ax, label="Return / Volatility Ratio")

        # Overlay specific points
        markers = {"Classical Exact": ("o", "red", 120), "QAOA": ("*", "cyan", 180)}
        for label, idxs in optimal_selections.items():
            if idxs is not None and len(idxs) > 0:
                w = np.zeros(n)
                w[idxs] = 1.0 / len(idxs)
                r = np.dot(w, mu) * 100
                v = np.sqrt(max(1e-8, w.T @ sigma @ w)) * 100
                m, c, s = markers.get(label, ("D", "orange", 100))
                ax.scatter(v, r, marker=m, color=c, edgecolors="black", s=s, label=f"{label} Optimal", zorder=5)

        ax.set_title(f"Markowitz Risk-Return Space & Optimal Selection (n={n}, B={budget})", fontsize=13, weight="bold")
        ax.set_xlabel("Annualized Volatility / Risk (%)")
        ax.set_ylabel("Annualized Expected Return (%)")
        ax.legend(loc="upper left")

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_scaling_curves(self, df_scaling: pd.DataFrame, filename: str = "scaling_curves.png") -> str:
        """Plots runtime, circuit depth, and combinatorial scaling across problem sizes n."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        n_vals = df_scaling["num_assets"]

        # Runtime scaling (Log scale)
        ax1.plot(n_vals, df_scaling["exact_time_sec"], "o-", color="#d95f02", label="Classical Exact (Brute Force)")
        ax1.plot(n_vals, df_scaling["sa_time_sec"], "s--", color="#7570b3", label="Classical Simulated Annealing")
        ax1.plot(n_vals, df_scaling["qaoa_time_sec"], "^-.", color="#1b9e77", label="Quantum QAOA (Simulation)")
        ax1.set_yscale("log")
        ax1.set_title("Runtime Scaling vs Problem Dimension n (Log Scale)", fontsize=12, weight="bold")
        ax1.set_xlabel("Number of Candidate Assets (n = Qubits)")
        ax1.set_ylabel("Runtime (seconds)")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend()

        # Circuit depth and 2-qubit gates
        ax2.plot(n_vals, df_scaling["qaoa_circuit_depth"], "o-", color="#1f77b4", label="QAOA Circuit Depth (p=1)")
        ax2.plot(n_vals, df_scaling["qaoa_cnot_count"], "s--", color="#ff7f0e", label="Two-Qubit (CX/CZ) Gates")
        ax2.set_title("Quantum Resource Scaling: Depth & Entanglers", fontsize=12, weight="bold")
        ax2.set_xlabel("Number of Qubits (Assets n)")
        ax2.set_ylabel("Count")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath
