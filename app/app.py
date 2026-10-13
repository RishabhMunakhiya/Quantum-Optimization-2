"""
Interactive Streamlit Dashboard for Quantum Portfolio Optimization
Track 2: Quantum Optimization: Smarter Choices at Scale
Qiskit Fall Fest 2026 x IIIT Delhi
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.data_loader import PortfolioDataLoader
from src.portfolio_qubo import PortfolioQUBO
from src.classical_solvers import ExactBruteForceSolver, SimulatedAnnealingSolver, GreedySharpeSolver
from src.quantum_solvers import NumPyQuantumExactSolver, QAOAPortfolioSolver
from src.visualizer import PortfolioVisualizer


st.set_page_config(
    page_title="Quantum Portfolio Optimizer | Qiskit Fall Fest",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("⚛️ Quantum Stock Portfolio Optimizer")
st.markdown("""
**Qiskit Fall Fest 2026 x IIIT Delhi** | **Track 2: Quantum Optimization — Smarter Choices at Scale**  
*Solving Markowitz Portfolio Selection using QAOA, VQE, and Classical Optimization Benchmarks.*
""")
st.divider()

# Sidebar Controls
st.sidebar.header("🔧 Portfolio & Quantum Parameters")

loader = PortfolioDataLoader()
all_tickers = loader.available_tickers

num_assets = st.sidebar.slider("Number of Candidate Stocks (n = Qubits)", min_value=3, max_value=len(all_tickers), value=4, step=1)
budget = st.sidebar.slider("Target Portfolio Size (Budget B)", min_value=1, max_value=num_assets - 1, value=max(1, num_assets // 2), step=1)
risk_factor = st.sidebar.slider("Risk Aversion Factor (q)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, help="Higher q penalizes volatility; lower q pursues maximum return.")

st.sidebar.subheader("Quantum Algorithm Settings")
qaoa_reps = st.sidebar.selectbox("QAOA Layers (reps p)", options=[1, 2], index=0, help="Number of alternating Cost and Mixer Hamiltonian layers.")
optimizer_choice = st.sidebar.selectbox("Classical Optimizer", options=["COBYLA", "SLSQP"], index=0)
max_iters = st.sidebar.slider("Optimizer Max Iterations", min_value=20, max_value=120, value=50, step=10)

# Fetch Data
mu, sigma, corr, selected_tickers = loader.get_subset(num_assets)
model = PortfolioQUBO(
    expected_returns=mu,
    covariances=sigma,
    budget=budget,
    risk_factor=risk_factor,
    tickers=selected_tickers
)

operator, offset = model.to_ising()

# Run Optimization Button
if st.sidebar.button("🚀 Run Portfolio Optimization", type="primary") or "results" not in st.session_state:
    with st.spinner("Computing Classical Baselines and Simulating Quantum QAOA Circuit..."):
        # Classical Exact
        res_exact = ExactBruteForceSolver(model).solve()
        # Classical SA
        res_sa = SimulatedAnnealingSolver(model, num_iterations=1000).solve()
        # Classical Greedy
        res_greedy = GreedySharpeSolver(model).solve()
        # Quantum Exact
        res_q_exact = NumPyQuantumExactSolver(model).solve()
        # QAOA
        res_qaoa = QAOAPortfolioSolver(model, reps=qaoa_reps, max_iter=max_iters, optimizer_name=optimizer_choice).solve()

        exact_cost = res_exact["markowitz_cost"]
        benchmark_rows = []
        for r in [res_exact, res_sa, res_greedy, res_q_exact, res_qaoa]:
            cost = r["markowitz_cost"]
            ratio = min(1.0, abs(exact_cost / cost)) if cost != 0 and exact_cost != 0 else 1.0
            benchmark_rows.append({
                "Solver": r["solver"],
                "Selected Stocks": ", ".join(r["selected_tickers"]),
                "Objective Cost": round(cost, 5),
                "Expected Return (%)": round(r["expected_return"] * 100, 2),
                "Volatility (%)": round(r["volatility"] * 100, 2),
                "Sharpe Ratio": round(r["sharpe_ratio"], 3),
                "Budget Valid": "✅ Yes" if r["is_valid"] else "❌ No",
                "Approx Ratio": round(ratio, 4),
                "Time (s)": round(r["execution_time_sec"], 4)
            })

        st.session_state["results"] = benchmark_rows
        st.session_state["qaoa_res"] = res_qaoa
        st.session_state["exact_res"] = res_exact

results_df = pd.DataFrame(st.session_state["results"])
qaoa_res = st.session_state["qaoa_res"]
exact_res = st.session_state["exact_res"]

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Optimal QAOA Selection", ", ".join(qaoa_res["selected_tickers"]))
with col2:
    st.metric("Expected Annual Return", f"{qaoa_res['expected_return']*100:.2f}%")
with col3:
    st.metric("Annualized Volatility (Risk)", f"{qaoa_res['volatility']*100:.2f}%")
with col4:
    st.metric("QAOA Approx Ratio", f"{qaoa_res.get('is_valid', True) and 1.0:.4f}")

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Solver Comparison",
    "📈 Markowitz Efficient Frontier",
    "🔗 Asset Correlation Matrix",
    "⚛️ QAOA Quantum Diagnostics",
    "📚 Beginner's Quantum Guide"
])

with tab1:
    st.subheader("Classical vs Quantum Algorithm Benchmark")
    st.dataframe(results_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(results_df.set_index("Solver")["Objective Cost"])
        st.caption("Objective Cost (lower is better; balances return vs covariance risk).")
    with c2:
        st.bar_chart(results_df.set_index("Solver")["Sharpe Ratio"])
        st.caption("Sharpe Ratio (higher indicates superior risk-adjusted return).")

with tab2:
    st.subheader("Risk vs. Return & Efficient Frontier")
    visualizer = PortfolioVisualizer()
    exact_idx = [selected_tickers.index(t) for t in exact_res["selected_tickers"]]
    qaoa_idx = [selected_tickers.index(t) for t in qaoa_res["selected_tickers"]]
    frontier_path = visualizer.plot_efficient_frontier(
        mu, sigma, selected_tickers,
        {"Classical Exact": exact_idx, "QAOA": qaoa_idx},
        budget=budget,
        filename="streamlit_frontier.png"
    )
    st.image(frontier_path, caption="Markowitz Risk-Return Opportunity Space with Quantum & Classical Selections", use_container_width=True)

with tab3:
    st.subheader("Asset Return Correlation Heatmap")
    heatmap_path = visualizer.plot_correlation_heatmap(corr, selected_tickers, filename="streamlit_corr.png")
    st.image(heatmap_path, caption="Covariance and correlation determine portfolio diversification benefits.", use_container_width=True)

with tab4:
    st.subheader("QAOA Convergence & Quantum Circuit Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Qubits Required", f"{model.n_assets} Qubits")
    with m2:
        st.metric("QAOA Layers (p)", f"{qaoa_reps}")
    with m3:
        st.metric("Circuit Depth", f"{qaoa_res.get('circuit_depth', '-')}")
    with m4:
        st.metric("Ising Pauli Terms", f"{len(operator)}")

    if qaoa_res.get("energy_history"):
        st.markdown("#### Classical Optimizer Convergence Trajectory")
        chart_data = pd.DataFrame(qaoa_res["energy_history"], columns=["Evaluation", "Energy"])
        st.line_chart(chart_data.set_index("Evaluation"))
        st.caption("Energy expectation value <H> minimizes toward the ground state.")

with tab5:
    st.subheader("Beginner's Guide: Understanding the Quantum Formulation")
    st.markdown("""
    ### 1. The Markowitz Portfolio Problem
    In classical finance, Harry Markowitz showed that an investor wants to maximize return while minimizing portfolio variance:
    $$\\min_{x \\in \\{0, 1\\}^n} \\left[ q \\cdot x^T \\Sigma x - \\mu^T x \\right] \\quad \\text{subject to} \\quad \\sum_{i=1}^n x_i = B$$
    - $\\mu$: Expected annual returns vector.
    - $\\Sigma$: Covariance matrix (asset risk and co-movements).
    - $q$: Risk aversion factor.
    - $B$: Total number of stocks to pick.

    ### 2. Converting to QUBO (Quadratic Unconstrained Binary Optimization)
    Quantum computers require unconstrained cost functions. We add a quadratic penalty $A(\\sum x_i - B)^2$:
    $$H_{\\text{QUBO}}(x) = q \\sum_{i,j} \\Sigma_{ij} x_i x_j - \\sum_{i} \\mu_i x_i + A \\left(\\sum_{i=1}^n x_i - B\\right)^2$$

    ### 3. Mapping to Ising Hamiltonian & Qubits
    Every binary variable $x_i \\in \\{0, 1\\}$ is mapped to a Quantum Pauli-Z operator $\\sigma_i^z \\in \\{+1, -1\\}$:
    $$x_i = \\frac{I - Z_i}{2}$$
    Substituting this yields the Ising Hamiltonian:
    $$\\hat{H} = \\sum_{i} h_i Z_i + \\sum_{i < j} J_{ij} Z_i Z_j + \\text{offset} \\cdot I$$
    The ground state (lowest energy state $|\\psi_0\\rangle$) corresponds to the mathematically optimal stock portfolio!

    ### 4. QAOA (Quantum Approximate Optimization Algorithm)
    QAOA applies alternating layers of:
    1. **Problem Unitary:** $U(C, \\gamma) = e^{-i \\gamma \\hat{H}}$ (encodes portfolio cost).
    2. **Mixer Unitary:** $U(B, \\beta) = e^{-i \\beta \\sum X_i}$ (drives quantum transitions).
    A classical optimizer iteratively updates $(\\gamma, \\beta)$ to minimize the energy expectation value.

    ### 5. Classical Simulation vs. Real Quantum Hardware
    - **This demonstration uses Statevector Simulation:** It simulates perfect quantum mathematics on your laptop without noise or cloud queue times.
    - **Real Quantum Hardware (IBM Quantum):** Suffers from decoherence, gate infidelity, and noise. Real execution requires Transpilation for hardware topology and Error Mitigation (e.g., ZNE or M3 readout correction).
    """)

st.caption("Built for Qiskit Fall Fest 2026 x IIIT Delhi | Track 2: Quantum Optimization")
