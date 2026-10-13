# Presentation Outline & Pitch Deck Guide
**Hackathon:** Qiskit Fall Fest 2026 x IIIT Delhi  
**Track 2:** Quantum Optimization: Smarter Choices at Scale  
**Project Title:** Quantum Stock Portfolio Optimizer  

---

### Slide 1: Title & Hook
- **Title:** Smart Quantum Stock Portfolio Optimizer
- **Subtitle:** Solving Real-World Combinatorial Asset Selection under Constraints with QAOA
- **Team / Presenter:** Quantum Optimization Track 2 Team
- **Tagline:** Bringing $2^n$ Financial Combinations into $n$ Qubits at Scale

---

### Slide 2: The Problem: Combinatorial Explosion in Finance
- Markowitz Modern Portfolio Theory balances return ($\mu$) and covariance risk ($\Sigma$).
- When real-world constraints are added (select exactly $B$ stocks from $n$), the problem becomes **NP-hard**.
- For $n = 50$ assets, there are over $1.1 \times 10^{15}$ choices—classical brute force completely stalls.

---

### Slide 3: The Quantum Solution Architecture
- **Step 1:** Formulate mean-variance optimization with cardinality constraints: $\min (q \cdot x^T \Sigma x - \mu^T x)$ s.t. $\sum x_i = B$.
- **Step 2:** Convert to Quadratic Unconstrained Binary Optimization (QUBO) with calibrated penalty $A(\sum x_i - B)^2$.
- **Step 3:** Map binary variables $x_i \to \frac{I - Z_i}{2}$ to generate the Problem Ising Hamiltonian $\hat{H}$.
- **Step 4:** Deploy QAOA with parameterized Cost and Mixer unitaries.

---

### Slide 4: Algorithmic Rigor & Classical Baselines
- We benchmark against three distinct classical baselines:
  1. **Classical Exact Brute-Force** (Global Ground Truth)
  2. **Classical Simulated Annealing** (Heuristic Stochastic Search)
  3. **Classical Greedy Sharpe Selection** (Standard Financial Baseline)
- Ground-truth comparison proves whether the quantum solution achieves the mathematical optimum.

---

### Slide 5: Empirical Results & Verification
- Highlight the Results Table from the actual test runs:
  - QAOA ($p=1$) achieves **100% Approximation Ratio** (identifying AAPL + MSFT).
  - Constraint satisfaction: **0% violation** (exactly $B=2$ stocks picked).
  - Mathematical ground state of Ising Hamiltonian exactly equals the classical brute force minimum.
  - Greedy heuristic failed to find the true global minimum because it neglected pairwise covariance interactions.

---

### Slide 6: Smarter Choices at Scale (Track 2 Focus)
- Hilbert space scaling: $2^n$ classical states represented by $n$ qubits.
- Circuit depth and gate scaling analysis:
  - Fully-connected covariance network requires $\mathcal{O}(n^2)$ two-qubit entangling gates.
  - Discussion of hardware routing (SWAP networks on IBM Heavy-Hex architectures).

---

### Slide 7: Interactive Live Dashboard Demo
- Walk through the interactive Streamlit dashboard (`http://localhost:8501`):
  - Dynamic stock sliders ($n=3$ to $10$) and budget controls.
  - Real-time solver comparison table and bar charts.
  - Markowitz Efficient Frontier visualization.
  - QAOA energy convergence curve.

---

### Slide 8: NISQ Reality vs Future Scalability
- Explain the distinction between classical statevector simulation and real physical quantum chips.
- Roadmap:
  - Hardware execution on IBM Quantum systems (e.g., `ibm_brisbane`, `ibm_kyoto`).
  - Readout error mitigation (M3) and zero-noise extrapolation (ZNE).
  - Warm-starting QAOA with classical continuous relaxations for large $n > 100$.

---

### Slide 9: Conclusion & Q&A
- **Summary:** Complete, working, reproducible quantum optimization pipeline with verified constraint satisfaction and scalable architecture.
- **GitHub / Codebase:** Fully structured repository with tests, CLI runner, and dashboard.
- Open floor for judges' questions.
