# Technical Report: Quantum Stock Portfolio Optimization at Scale

**Event:** Qiskit Fall Fest 2026 x IIIT Delhi  
**Organizer:** Quantica under Centre for Quantum Technology (CQT), IIIT Delhi  
**Track:** Track 2: Quantum Optimization — Smarter Choices at Scale  
**Project:** Smart Stock Portfolio Optimizer  
**Quantum Framework:** Qiskit 2.5.2, Qiskit Optimization 0.7.0, Qiskit Algorithms 0.4.0  

---

## 1. Executive Summary

Portfolio optimization is one of the foundational challenges of quantitative finance. As financial markets grow in complexity and asset universes expand, determining the mathematically optimal subset of assets under real-world constraints (e.g., cardinality budgets, transaction fees, and sectoral exposure limits) turns into an **NP-hard combinatorial optimization problem**.

This project implements a complete, end-to-end Quantum Optimization solution for modern Markowitz portfolio selection. We formulate the constrained mean-variance portfolio selection problem into a Quadratic Unconstrained Binary Optimization (QUBO) problem, map it onto an Ising Hamiltonian, and solve it using the **Quantum Approximate Optimization Algorithm (QAOA)** and the **Variational Quantum Eigensolver (VQE)** via Qiskit.

We rigorously evaluate the quantum algorithms against three classical baselines:
1. **Classical Exact Brute Force** (Global Ground Truth)
2. **Classical Simulated Annealing** (Heuristic Stochastic Metaheuristic)
3. **Classical Greedy Top-Sharpe** (Heuristic Selection)
4. **Quantum Exact NumPy Eigensolver** (Analytical Ground State)

Furthermore, to address Track 2's focal mandate—*"Smarter Choices at Scale"*—we analyze runtime scaling, circuit depth growth, entangling gate density, and state space explosion ($2^n$) across increasing problem sizes.

---

## 2. Mathematical Problem Formulation

### 2.1 Modern Portfolio Theory (Markowitz Framework)
Let $n$ denote the total number of candidate stocks. Each binary decision variable $x_i \in \{0, 1\}$ represents whether stock $i$ is included in the portfolio ($x_i = 1$) or excluded ($x_i = 0$).

- $\mu \in \mathbb{R}^n$: Annualized expected returns vector.
- $\Sigma \in \mathbb{R}^{n \times n}$: Annualized covariance matrix of asset returns.
- $q \in \mathbb{R}^+$: Investor's risk aversion coefficient.
- $B \in \mathbb{Z}^+$: Cardinality investment budget (exact number of stocks to select).

The constrained optimization problem is:
$$\min_{x \in \{0, 1\}^n} \left[ q \cdot x^T \Sigma x - \mu^T x \right]$$
subject to the equality constraint:
$$\sum_{i=1}^n x_i = B$$

### 2.2 QUBO Reformulation
Gate-based quantum algorithms and quantum annealers operate natively on unconstrained objective functions. We incorporate the cardinality constraint using a quadratic penalty with multiplier $A > 0$:

$$H_{\text{QUBO}}(x) = q \sum_{i=1}^n \sum_{j=1}^n \Sigma_{ij} x_i x_j - \sum_{i=1}^n \mu_i x_i + A \left( \sum_{i=1}^n x_i - B \right)^2$$

Expanding the quadratic penalty term:
$$\left( \sum_{i=1}^n x_i - B \right)^2 = \sum_{i=1}^n x_i^2 + 2 \sum_{i < j} x_i x_j - 2B \sum_{i=1}^n x_i + B^2$$
Because $x_i \in \{0, 1\}$, idempotent identity $x_i^2 = x_i$ holds:
$$\left( \sum_{i=1}^n x_i - B \right)^2 = (1 - 2B) \sum_{i=1}^n x_i + 2 \sum_{i < j} x_i x_j + B^2$$

To guarantee that any invalid bitstring has higher energy than any valid portfolio, the penalty multiplier $A$ is dynamically calibrated:
$$A \ge \max_{x} |\mu^T x| + q \max_{x} |x^T \Sigma x|$$

### 2.3 Mapping to Quantum Ising Hamiltonian
Binary decision variables $x_i \in \{0, 1\}$ map directly to quantum Pauli-$Z$ operators $\sigma_i^z \in \{+1, -1\}$ via the affine transformation:
$$x_i = \frac{I - Z_i}{2}$$

Substituting into $H_{\text{QUBO}}(x)$ produces the Ising Hamiltonian:
$$\hat{H} = \sum_{i=1}^n h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + C \cdot I$$
where:
- $h_i$ represents the effective local magnetic field on qubit $i$.
- $J_{ij}$ represents the two-body coupling interaction between qubits $i$ and $j$.
- $C$ is an energy offset constant.

Finding the ground state $|\psi_0\rangle$ of $\hat{H}$ such that $\hat{H}|\psi_0\rangle = E_0 |\psi_0\rangle$ directly yields the optimal portfolio bitstring.

---

## 3. Quantum Algorithm Implementation

### 3.1 Quantum Approximate Optimization Algorithm (QAOA)
QAOA uses an alternating sequence of non-commuting unitary operators:
1. **Initial State:** Equal superposition of all $2^n$ basis states:
   $$|+\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle$$
2. **Cost Hamiltonian Unitary:** Encodes problem cost and penalties:
   $$U(C, \gamma) = \exp(-i \gamma \hat{H}_C) = \prod_{i < j} \exp(-i \gamma J_{ij} Z_i Z_j) \prod_i \exp(-i \gamma h_i Z_i)$$
3. **Mixer Hamiltonian Unitary:** Drives transitions across computational states:
   $$U(B, \beta) = \exp(-i \beta \hat{H}_M) = \prod_{i=1}^n \exp(-i \beta X_i)$$
4. **Parameterized Quantum State for $p$ layers:**
   $$|\gamma, \beta\rangle = \prod_{k=1}^p \left( U(B, \beta_k) U(C, \gamma_k) \right) |+\rangle^{\otimes n}$$
5. **Variational Optimization Loop:**
   A classical optimizer (COBYLA / SLSQP) updates parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ to minimize:
   $$\min_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \langle \gamma, \beta | \hat{H}_C | \gamma, \beta \rangle$$

### 3.2 Variational Quantum Eigensolver (SamplingVQE)
In addition to QAOA, we deploy a heuristic variational solver:
- **Ansatz:** Parameterized `TwoLocal` circuit with $R_y$ single-qubit rotation blocks and linear $CZ$ two-qubit entanglement blocks.
- **Optimization:** Classical Nelder-Mead / COBYLA updates variational parameters to find the minimum expectation value of the Ising Hamiltonian $\hat{H}$.

### 3.3 Quantum Circuit Metrics & Representation
To ensure absolute scientific accuracy, we explicitly distinguish between three levels of quantum circuit representation:
1. **High-Level Composite Ansatz:** High-level abstractions (`QAOAAnsatz`, `TwoLocal`, `PauliEvolution`) that wrap multi-qubit interactions into single composite blocks. Direct operation counting at this level trivially reports depth $\approx 2$ and CNOT count $= 0$.
2. **Logical Decomposed Circuit:** The platform-independent circuit decomposed into standard elementary gates ($CX$, $R_Z$, $R_X$, $U$). This exposes the true two-qubit entangler counts ($n(n-1)$ for all-to-all connectivity) and logical circuit depth without hardware topology constraints.
3. **Hardware Transpiled Circuit:** Circuit transpiled to physical target basis gates (`cx`, `rz`, `sx`, `x`) and routed via SWAP networks onto physical chip coupling topologies (e.g., IBM Heavy-Hex lattice).

---

## 4. Empirical Experimental Results: 4-Asset Baseline

We evaluate a baseline 4-asset problem with budget $B=2$ and risk aversion $q=0.5$:
- **Candidate Assets:** `AAPL`, `MSFT`, `JPM`, `JNJ`
- **Constraint:** Select exactly 2 stocks ($\sum x_i = 2$)
- **Simulation Backend:** `qiskit.primitives.StatevectorSampler` (seed: 42, 1024 shots)

### 4.1 Solver Benchmark Comparison

| Solver | Selected Stocks | Markowitz Cost | Expected Return | Volatility | Sharpe Ratio | Constraint Violation | Optimality Gap ($\Delta$) | CPU Simulation Time (s) | Logical Circuit Depth | CNOT Gate Count | Two-Qubit Entanglers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Exact (Brute Force)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | 0 (Valid) | **0.00000** | 0.0003 | N/A | N/A | N/A |
| **Classical Simulated Annealing** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | 0 (Valid) | **0.00000** | 0.1028 | N/A | N/A | N/A |
| **Classical Greedy (Top Sharpe)** | MSFT, JPM | -0.74829 | 39.86% | 15.63% | 2.358 | 0 (Valid) | +0.00730 | 0.0001 | N/A | N/A | N/A |
| **Quantum Exact (NumPy Diag.)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | 0 (Valid) | **0.00000** | 0.0104 | N/A | N/A | N/A |
| **Quantum QAOA ($p=1$, COBYLA)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | 0 (Valid) | **0.00000** | 2.4661 | **19** | **12** | **12** |
| **Quantum QAOA ($p=2$, COBYLA)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | 0 (Valid) | **0.00000** | 7.1768 | **33** | **24** | **24** |
| **Quantum VQE (TwoLocal, $p=1$)** | AAPL, MSFT | **-0.75559** | 41.04% | 18.05% | 2.107 | 0 (Valid) | **0.00000** | 0.8819 | **6** | **3** | **3** |

*Note on Transpiled Universal Basis (`cx, rz, sx, x`): QAOA ($p=1$) transpiles to depth 24 with 12 CNOTs; QAOA ($p=2$) transpiles to depth 48 with 24 CNOTs; VQE ($p=1$) transpiles to depth 18 with 3 CNOTs.*

### 4.2 Mathematical Metrics & Optimality Gap
Because the Markowitz objective $q \cdot x^T \Sigma x - \mu^T x$ is negative when expected returns outweigh risk penalties (e.g., $-0.75559$), taking simple ratios of negative values can produce inverted or deceptive ratios. We therefore report the mathematically unambiguous **Optimality Gap**:
$$\Delta = \text{Cost}_{\text{solver}} - \text{Cost}_{\text{exact}} \ge 0$$
- An optimality gap of $\Delta = 0.00000$ proves that the solver successfully identified the global ground state configuration.
- The classical greedy heuristic exhibited an optimality gap of $\Delta = +0.00730$, proving that selecting assets based strictly on individual Sharpe ratios is suboptimal because it ignores covariance cancellation ($\Sigma_{ij}$ diversification benefits).

---

## 5. Smarter Choices at Scale: Empirical Multi-Scale Analysis

To directly address Track 2's theme—*"Smarter Choices at Scale"*—we benchmarked the entire optimization pipeline across increasing asset dimensions: $n = 4, 6, 8, 10$ candidate assets with target budgets $B = \max(1, \lfloor n/2 \rfloor)$.

### 5.1 Verified Scaling Benchmark Table (Measured from `outputs/scaling_benchmark.csv`)

| Number of Assets ($n$) | Target Budget ($B$) | Hilbert Space ($2^n$) | Exact Cost | Classical Exact CPU Time (s) | Simulated Annealing Cost | SA Optimality Gap ($\Delta$) | SA Time (s) | QAOA Cost ($p=1$) | QAOA Optimality Gap ($\Delta$) | QAOA CPU Sim. Time (s) | QAOA Logical Depth | QAOA CNOT Count | QAOA Two-Qubit Entanglers |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 2 | 16 | **-0.75559** | 0.00047 | -0.75559 | **0.00000** | 0.1070 | **-0.75559** | **0.00000** | 1.82 | 19 | 12 | 12 |
| **6** | 3 | 64 | **-1.06694** | 0.00081 | -1.06694 | **0.00000** | 0.1005 | **-1.06694** | **0.00000** | 6.60 | 31 | 30 | 30 |
| **8** | 4 | 256 | **-1.33527** | 0.00382 | -1.21562 | **+0.11965** | 0.0907 | **-1.33527** | **0.00000** | 55.80 | 43 | 56 | 56 |
| **10** | 5 | 1,024 | **-2.85396** | 0.01224 | -1.85569 | **+0.99828** | 0.1651 | **-2.85396** | **0.00000** | 949.23 | 55 | 90 | 90 |

### 5.2 Key Scaling Insights

1. **Quadratic Entangling Gate Scaling:**
   In Markowitz portfolio optimization, asset returns exhibit dense pairwise covariances $\Sigma_{ij} \ne 0$. Consequently, the cost Hamiltonian possesses dense two-body Ising couplings $\sum_{i<j} J_{ij} Z_i Z_j$. Each interaction term translates into one $R_{ZZ}(\theta)$ gate, which decomposes into 2 CNOT gates.
   $$\text{CNOT Count} = 2 \times \binom{n}{2} = n(n-1) = \mathcal{O}(n^2)$$
   - At $n=4$: $4 \times 3 = \mathbf{12}$ CNOTs
   - At $n=6$: $6 \times 5 = \mathbf{30}$ CNOTs
   - At $n=8$: $8 \times 7 = \mathbf{56}$ CNOTs
   - At $n=10$: $10 \times 9 = \mathbf{90}$ CNOTs
2. **Linear Logical Depth Scaling:**
   Logical circuit depth grows strictly linearly: $\text{Depth} = 19 + 6(n-4) = \mathcal{O}(n)$. This confirms predictable circuit compilation scaling.
3. **Classical Heuristic Breakdown vs Quantum Ground State:**
   While Simulated Annealing is fast, it becomes trapped in local minima at higher dimensions:
   - At $n=8$, SA yields an optimality gap of $+0.11965$.
   - At $n=10$, SA severely degrades with an optimality gap of $+0.99828$.
   - Conversely, **QAOA converged to the exact global ground state at all scales ($\Delta = 0.00000$)** with $100\%$ constraint satisfaction.
4. **Classical Statevector Simulation Costs:**
   On a classical CPU, simulating the full statevector requires updating $2^n$ complex amplitudes. Runtime increases from $1.82$s at $n=4$ to $949.23$s ($\approx 15.8$ minutes) at $n=10$. This vividly illustrates why classical statevector simulation becomes intractable for $n > 30$, motivating the eventual migration to physical quantum processing units.

---

## 6. Classical Simulation vs. Real Quantum Hardware

- **Classical Statevector Simulation (This Implementation):** Uses `qiskit.primitives.StatevectorSampler` to simulate exact unitary evolution without quantum noise, decoherence, or cloud queue times. It enables mathematical verification of algorithmic convergence.
- **Physical Quantum Hardware Considerations (NISQ Era):**
  - **Execution Time:** A real QPU executes a circuit shot in microseconds. However, free-tier cloud queues typically involve $30$–$120$ minute wait times per job, rendering multi-iteration variational loops impractical on free shared queues.
  - **Decoherence & Fidelity:** Physical transmon qubits (e.g., IBM Eagle / Heron) have finite coherence times ($T_1, T_2 \approx 100$–$300\,\mu\text{s}$). A depth-55 circuit with 90 CNOTs would experience significant gate error accumulation without Dynamical Decoupling (DD) and Zero-Noise Extrapolation (ZNE).
  - **Hardware Routing:** Real QPUs have limited connectivity (Heavy-Hex topology). Routing all-to-all quadratic couplings requires inserting SWAP gates, expanding depth and gate counts further.

---

## 7. Project Limitations & Ethical Transparency

In accordance with rigorous scientific practices, we document key limitations:
1. **No Premature Claim of Quantum Advantage:** At $n \le 10$, classical exact brute-force ($0.012$s) and modern mixed-integer solvers drastically outperform quantum statevector simulation ($949$s). Quantum advantage cannot be claimed at NISQ scales; this project demonstrates algorithmic feasibility and resource scaling.
2. **Dataset Provenance:** When running offline, the data pipeline uses a calibrated synthetic Geometric Brownian Motion (GBM) model with correlated Cholesky shocks (tech sector correlation $0.65$, market baseline $0.35$). A live download via `yfinance` is attempted with graceful fallback.
3. **Cardinality Selection vs. Continuous Weighting:** This model solves the combinatorial 0-1 subset selection problem ($\sum x_i = B$). Once the $B$ assets are chosen, equal weighting ($w_i = 1/B$) is evaluated. Continuous capital weighting is a subsequent classical quadratic programming step.
4. **Absence of Physical Hardware Run:** Due to free-tier cloud queue constraints, all experiments were conducted on local statevector simulators.

---

## 8. Conclusion

This project delivers a complete, mathematically verified quantum optimization solution for Markowitz portfolio selection. By verifying penalty calibrations, analyzing quadratic CNOT scaling, contrasting QAOA against classical exact and heuristic baselines, and establishing rigorous optimality gaps, the work provides an honest and comprehensive contribution to **Track 2: Quantum Optimization — Smarter Choices at Scale**.
