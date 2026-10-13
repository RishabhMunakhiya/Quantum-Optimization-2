# ⚛️ Smart Quantum Stock Portfolio Optimizer at Scale

**Qiskit Fall Fest 2026 x IIIT Delhi**  
**Track 2: Quantum Optimization: Smarter Choices at Scale**  
*Organized by Quantica under the Centre for Quantum Technology (CQT), IIIT Delhi*

---

## 🌟 Overview

The **Smart Quantum Stock Portfolio Optimizer** is a complete, hackathon-ready quantum computing project that solves the Markowitz mean-variance portfolio selection problem under cardinality budget constraints using quantum algorithms in Qiskit.

### Key Features
- **Rigorous Mathematical Formulation:** Maps constrained Markowitz portfolio optimization to a Quadratic Unconstrained Binary Optimization (QUBO) problem and encodes it into an Ising Hamiltonian ($x_i \to \frac{I - Z_i}{2}$).
- **Quantum Variational Solvers:** Implements **QAOA** (Quantum Approximate Optimization Algorithm) and **VQE** (Variational Quantum Eigensolver) with modern Qiskit primitives (`StatevectorSampler`).
- **Comprehensive Classical Baselines:** Benchmarks against **Classical Exact Brute-Force**, **Simulated Annealing**, and **Greedy Sharpe Selection**.
- **Scalability Analysis ("Smarter Choices at Scale"):** Explores combinatorial state explosion ($2^n$), qubit scaling, circuit depth, and two-qubit CNOT gate counts across dimensions ($n = 4, 6, 8, 10$).
- **Interactive Web Dashboard:** Clean, responsive Streamlit dashboard featuring live optimization, parameter tuning, correlation heatmaps, convergence plots, and Markowitz efficient frontiers.
- **Robust & Tested:** Fully verified with unit and integration tests passing in Python 3.11.

---

## 🚀 Interactive Dashboard

The project includes an interactive Streamlit dashboard for real-time portfolio optimization and visualization.

👉 **Launch in your terminal:**
```powershell
.\.venv\Scripts\streamlit.exe run app/app.py
```
Then navigate to [http://localhost:8501](http://localhost:8501) in your browser.

### Features in the App:
1. **Interactive Controls:** Adjust candidate stocks ($n=3$ to $10$), budget ($B$), risk aversion ($q$), and QAOA layers ($p$).
2. **Solver Comparison Table:** Side-by-side comparison of Classical Exact, Simulated Annealing, Greedy, and Quantum QAOA.
3. **Markowitz Efficient Frontier:** Interactive risk-return scatter plot with optimal quantum and classical portfolios highlighted.
4. **Correlation Heatmap:** Visualizing cross-asset covariance and diversification benefits.
5. **QAOA Energy Convergence Trace:** Live view of COBYLA minimizing the Hamiltonian expectation value.
6. **Beginner's Walkthrough:** Simple explanation of qubits, operators, and algorithms.

---

## 📊 Verified Empirical Benchmark Results

### 1. Single Comparison Baseline ($n=4$ Assets, Budget $B=2$, Risk Aversion $q=0.5$)
Measured using real statistical moments on `['AAPL', 'MSFT', 'JPM', 'JNJ']`:

| Solver | Selected Stocks | Markowitz Cost | Expected Return | Volatility | Sharpe Ratio | Valid Budget | Optimality Gap ($\Delta$) | CPU Sim. Time (s) | Logical Depth | CNOT Gates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Exact (Brute Force)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | Yes (0 viol.) | **0.00000** | 0.0003 | - | - |
| **Classical Simulated Annealing** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | Yes (0 viol.) | **0.00000** | 0.1028 | - | - |
| **Classical Greedy (Top Sharpe)** | MSFT, JPM | -0.74829 | 39.86% | 15.63% | 2.358 | Yes (0 viol.) | +0.00730 | 0.0001 | - | - |
| **Quantum Exact (NumPy Diag.)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | Yes (0 viol.) | **0.00000** | 0.0104 | - | - |
| **Quantum QAOA ($p=1$, COBYLA)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | Yes (0 viol.) | **0.00000** | 2.4661 | **19** | **12** |
| **Quantum QAOA ($p=2$, COBYLA)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | Yes (0 viol.) | **0.00000** | 7.1768 | **33** | **24** |
| **Quantum VQE (TwoLocal, $p=1$)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | Yes (0 viol.) | **0.00000** | 0.8819 | **6** | **3** ($CZ$) |

### 2. Multi-Scale Benchmark ("Smarter Choices at Scale", $n = 4, 6, 8, 10$)
Measured from `outputs/scaling_benchmark.csv`:

| Assets ($n$) | Budget ($B$) | State Space ($2^n$) | Exact Cost | Exact Time (s) | SA Cost | SA Gap ($\Delta$) | QAOA Cost ($p=1$) | QAOA Gap ($\Delta$) | QAOA Sim. Time (s) | Logical Depth | CNOT Gates |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 2 | 16 | **-0.75559** | 0.00047 | -0.75559 | **0.00000** | **-0.75559** | **0.00000** | 1.82 | 19 | 12 |
| **6** | 3 | 64 | **-1.06694** | 0.00081 | -1.06694 | **0.00000** | **-1.06694** | **0.00000** | 6.60 | 31 | 30 |
| **8** | 4 | 256 | **-1.33527** | 0.00382 | -1.21562 | **+0.11965** | **-1.33527** | **0.00000** | 55.80 | 43 | 56 |
| **10** | 5 | 1,024 | **-2.85396** | 0.01224 | -1.85569 | **+0.99828** | **-2.85396** | **0.00000** | 949.23 | 55 | 90 |

> **Key Observations:**
> 1. **Quadratic CNOT Scaling:** Two-qubit entanglers scale as $\text{CNOTs} = n(n-1) = \mathcal{O}(n^2)$ due to dense covariance matrix terms ($12 \to 30 \to 56 \to 90$).
> 2. **Optimality Gap $\Delta = 0$:** QAOA consistently converges to the exact mathematical global minimum across all scales ($100\%$ constraint satisfaction).
> 3. **Classical Heuristic Breakdown:** Simulated Annealing misses the optimum at $n=8$ (gap $+0.12$) and $n=10$ (gap $+1.00$).
> 4. **No Quantum Advantage Claimed:** At $n \le 10$, classical exact brute-force executes in $0.012$s, while classical statevector simulation on CPU requires $949.2$s. Quantum advantage is not claimed at this scale.

---

## 🛠️ Project Structure

```
Quantum Optimization/
├── app/
│   └── app.py                      # Interactive Streamlit Web Dashboard
├── data/
│   ├── generate_sample_data.py     # Correlated GBM dataset generator & yfinance fetcher
│   └── stock_data.csv              # Calibrated 252-day market return dataset
├── deliverables/
│   ├── ABSTRACT_SUBMISSION.md      # Concise abstract and keywords for Unstop
│   ├── PRESENTATION_OUTLINE.md     # Pitch deck / slides structure with speaking notes
│   ├── SUBMISSION_CHECKLIST.md     # Evaluation matrix matching hackathon criteria
│   └── TECHNICAL_REPORT.md         # Comprehensive formal research report
├── outputs/                        # High-resolution benchmark plots & CSVs
│   ├── correlation_heatmap.png     # Asset return correlations
│   ├── efficient_frontier.png      # Risk vs Return Monte Carlo frontier
│   ├── qaoa_convergence.png        # QAOA energy minimization trace
│   ├── scaling_curves.png          # Scalability and quantum resource curves
│   └── solver_comparison.png       # Objective cost and Sharpe ratio comparisons
├── src/
│   ├── benchmark.py                # Comparative benchmarking suite
│   ├── classical_solvers.py        # Exact Brute Force, Simulated Annealing, Greedy
│   ├── data_loader.py              # Financial data loader and statistical moments
│   ├── portfolio_qubo.py           # QUBO, Ising Hamiltonian, and metric evaluator
│   ├── quantum_solvers.py          # QAOA, SamplingVQE, NumPy Minimum Eigensolver
│   └── visualizer.py               # Publication-grade plotting utilities
├── tests/
│   └── test_portfolio.py           # Automated unit and integration tests
├── main.py                         # Unified CLI execution script
├── requirements.txt                # Pinned dependencies
└── README.md                       # Complete documentation
```

---

## 💻 How to Run (Step-by-Step for Windows)

### 1. Launch the Interactive Dashboard
If the dashboard is not already running:
```powershell
.\.venv\Scripts\streamlit.exe run app/app.py
```
Then navigate to `http://localhost:8501`.

### 2. Run the Command-Line Pipeline (CLI)
To run the default 4-stock comparison and regenerate all plots in `outputs/`:
```powershell
.\.venv\Scripts\python.exe main.py
```

### 3. Run with Custom Parameters
```powershell
.\.venv\Scripts\python.exe main.py --assets 6 --budget 3 --risk 0.7 --reps 2 --iters 60
```

### 4. Run the Multi-Asset Scaling Benchmark
```powershell
.\.venv\Scripts\python.exe main.py --scale
```

### 5. Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_portfolio.py
```

---

## 🧠 Beginner's Guide: How Quantum Optimization Works

### Step 1: The Financial Goal
We want to pick the best combination of $B$ stocks from $n$ available choices to maximize expected return while minimizing volatility risk.

### Step 2: The QUBO Equation
Quantum hardware minimizes quadratic energy functions of binary variables ($x_i \in \{0, 1\}$). To enforce that we pick exactly $B$ stocks without violating our budget, we add a quadratic penalty term:
$$H(x) = \underbrace{q \cdot x^T \Sigma x}_{\text{Portfolio Risk}} - \underbrace{\mu^T x}_{\text{Expected Return}} + \underbrace{A \left(\sum_{i=1}^n x_i - B\right)^2}_{\text{Budget Penalty}}$$

### Step 3: From Bits to Qubits (Ising Hamiltonian)
We map each binary bit $x_i$ to a quantum spin operator (Pauli-$Z$ matrix):
$$x_i = \frac{I - Z_i}{2}$$
This creates an **Ising Hamiltonian** operator $\hat{H}$. The lowest-energy state (ground state $|0\rangle$) of this Hamiltonian represents the most profitable, risk-balanced stock portfolio!

### Step 4: Solving with QAOA
QAOA puts $n$ qubits into an equal superposition of all $2^n$ possible portfolios at once. It repeatedly alternates between:
1. **Cost Unitary $e^{-i \gamma \hat{H}}$:** Shifts the quantum phases according to portfolio quality.
2. **Mixer Unitary $e^{-i \beta \sum X_i}$:** Drives quantum interference across candidate portfolios.

A classical optimizer iteratively tunes the angles $(\gamma, \beta)$ so that measuring the quantum circuit collapses into the optimal stock selection with highest probability.

---

## 🔬 Classical Simulation vs. Real Quantum Hardware

- **This Project (Statevector Simulation):** Runs an exact simulation of quantum mechanics on your computer using Qiskit's `StatevectorSampler`. It is deterministic, noise-free, and ideal for algorithm verification.
- **Physical IBM Quantum Computers (NISQ Era):** Physical superconducting transmon qubits experience environmental noise, gate errors, and decoherence. Running on real quantum chips requires:
  1. **Transpilation:** Mapping all-to-all QUBO interactions onto hardware connectivity graphs (e.g. IBM Heavy-Hex lattice).
  2. **Error Mitigation:** Using techniques like M3 Readout Error Mitigation or Zero-Noise Extrapolation (ZNE).

---

## ⚠️ Project Limitations & Transparency

1. **No Claims of Quantum Advantage:** At current problem sizes ($n \le 10$), classical brute-force ($0.012$s) and mathematical programming significantly outperform classical simulation of quantum circuits ($949.2$s). This project demonstrates algorithmic correctness, formulation validity, and quantum resource scaling.
2. **Dataset Provenance:** The project utilizes realistic historical price data. In offline environments, an auto-calibrated Geometric Brownian Motion (GBM) model with correlated Cholesky shocks is generated. An automated live fetch via `yfinance` is attempted with graceful fallback.
3. **Cardinality Selection:** Solves binary subset selection ($\sum x_i = B$) with equal allocation ($w_i = 1/B$) within chosen assets. Continuous weight allocation is a subsequent classical quadratic programming step.
4. **Simulation vs Hardware:** Verified using Qiskit local statevector simulation. Due to free-tier cloud queuing delays, physical cloud QPUs were not used for the multi-iteration variational loops.

---

## 🏆 Hackathon Submission Notes (Unstop)

- **Track:** Track 2: Quantum Optimization: Smarter Choices at Scale
- **Institution:** IIIT Delhi x Quantica (Centre for Quantum Technology)
- All required deliverables (Technical Report, Abstract, Presentation Outline, and Verification Checklist) are stored in [`deliverables/`](deliverables/).
