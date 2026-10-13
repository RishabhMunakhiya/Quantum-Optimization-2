# Unstop Submission Description & Abstract
**Qiskit Fall Fest 2026 x IIIT Delhi | Track 2: Quantum Optimization — Smarter Choices at Scale**

### Project Title
**Smart Quantum Stock Portfolio Optimizer at Scale**

### Short Tagline (1 Line)
Solving constrained Markowitz portfolio optimization using QAOA, VQE, and classical baselines with multi-asset scalability analysis.

### Executive Summary / Abstract (For Unstop Form)
Modern financial asset allocation under cardinality constraints is an NP-hard combinatorial problem where the search space scales as $2^n$. In this project, built for Track 2 ("Smarter Choices at Scale"), we develop a complete end-to-end Quantum Portfolio Optimizer utilizing Qiskit.

We formulate Markowitz's mean-variance portfolio selection problem into a Quadratic Unconstrained Binary Optimization (QUBO) model with an exact quadratic penalty for investment budget constraints. The QUBO is subsequently mapped to an Ising Hamiltonian via Pauli-Z operators ($\sigma_i^z$), allowing execution on quantum variational algorithms.

Using Qiskit's modern primitives (`StatevectorSampler`), we implement the Quantum Approximate Optimization Algorithm (QAOA) and Variational Quantum Eigensolver (VQE) and evaluate them alongside three classical baselines:
1. Classical Exact Brute-Force (Ground Truth)
2. Classical Simulated Annealing (Metaheuristic)
3. Classical Greedy Selection (Top Sharpe Ratio)
4. Quantum Exact Diagonalization (`NumPyMinimumEigensolver`)

Our experiments show that QAOA successfully converges to the global mathematical minimum with a 100% approximation ratio and 0% constraint violations, outperforming heuristic greedy baselines that miss cross-asset diversification benefits. Furthermore, we provide a multi-asset scaling analysis across increasing problem sizes ($n = 4, 6, 8, 10$) exploring quantum resource requirements (qubit count, circuit depth, and two-qubit entangling gates) versus classical exponential runtimes.

The project features a modular Python architecture, automated unit tests, and an interactive Streamlit dashboard for real-time portfolio analysis and Markowitz efficient frontier visualization.

### Keywords
`Quantum Computing`, `Qiskit`, `QAOA`, `Portfolio Optimization`, `QUBO`, `Ising Hamiltonian`, `VQE`, `Financial Engineering`, `Combinatorial Optimization`
