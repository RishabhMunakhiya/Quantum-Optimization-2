# Hackathon Submission Checklist & Evaluation Matrix
**Hackathon:** Qiskit Fall Fest 2026 x IIIT Delhi  
**Organizer:** Quantica (Centre for Quantum Technology, IIIT Delhi)  
**Track:** Track 2: Quantum Optimization: Smarter Choices at Scale  

---

### Evaluation Criteria Verification

### 1. Completed Project Engineering & Research Tasks (Verified)

| Component / Criterion | Status | Implementation Proof & Verified Details |
| :--- | :---: | :--- |
| **Track Alignment** | ✅ Completed | Formulates portfolio stock selection under budget constraints and evaluates scaling across $n = 4, 6, 8, 10$ assets. |
| **Problem Formulation** | ✅ Completed | Implemented Markowitz mean-variance optimization with cardinality constraint mapped to QUBO and Ising Hamiltonian ($x_i = \frac{I - Z_i}{2}$). |
| **Quantum Optimization Algorithm** | ✅ Completed | Implemented QAOA (`qiskit_algorithms.QAOA`) and SamplingVQE with modern Qiskit primitives (`StatevectorSampler`). |
| **Circuit Decomposition & Metrics** | ✅ Completed | Fixed ansatz decomposition; correctly reports quadratic CNOT scaling $n(n-1)$ ($12 \to 30 \to 56 \to 90$) and linear depth. |
| **Classical Baselines** | ✅ Completed | Implemented Classical Exact (Brute Force), Classical Simulated Annealing, and Greedy Sharpe heuristics. |
| **Scaling Benchmark ($n=4, 6, 8, 10$)** | ✅ Completed | Measured and validated across 4 scales; QAOA achieved Optimality Gap $\Delta = 0.00000$ at all sizes, whereas SA failed at $n=8$ and $n=10$. |
| **Simulation vs Physical Hardware** | ✅ Completed | Clearly documented CPU statevector simulation overhead (949s at $n=10$) vs physical NISQ noise, gate errors, and decoherence. |
| **No Advantage Disclaimer** | ✅ Completed | Explicitly noted that classical exact brute-force ($0.012$s) outperforms simulation at $n \le 10$; no false superiority claimed. |
| **Automated Unit Tests** | ✅ Completed | 8 automated unit tests passing in `tests/test_portfolio.py` covering math, gates, QUBO, and solvers. |
| **Interactive User Interface** | ✅ Completed | Streamlit dashboard running locally at `http://localhost:8501`; verified HTTP 200 and captured screenshot. |
| **Jupyter Walkthrough Notebook** | ✅ Completed | `submission_walkthrough.ipynb` with 18 cells, executed cleanly from top to bottom with zero errors. |
| **PowerPoint Presentation Deck** | ✅ Completed | `deliverables/Track2_Hackathon_Presentation.pptx` (9 professional widescreen slides, verified via PowerPoint COM). |
| **Code & Secret Hygiene** | ✅ Completed | `.gitignore` active; `.venv`, `.env`, and private secrets excluded; source folder preserved untouched. |

---

### 2. Action Items Requiring User / Team Action (Before Oct 14, 11:59 PM IST)

| Priority | Pending Action Item | Description & Guidance |
| :---: | :--- | :--- |
| **P1** | **Check Unstop Submission Fields** | Review the submission portal to confirm what specific deliverables are required (e.g., GitHub URL, PDF Report, Slides PPTX/PDF, Video Link, Abstract text). |
| **P2** | **Add Personal Team Details** | Replace placeholder `[Team Name / Participant Names Placeholder]` in `deliverables/Track2_Hackathon_Presentation.pptx`, `README.md`, and `submission_walkthrough.ipynb`. |
| **P3** | **Export Technical Report to PDF (If Required)** | If Unstop requires a PDF file upload, export `deliverables/TECHNICAL_REPORT.md` (or print from browser/VS Code markdown preview) to `Technical_Report.pdf`. |
| **P4** | **Record 2-Minute Demo Video (If Required)** | Open `http://localhost:8501` in your browser and record a quick walkthrough of the sliders, solver comparison, and efficient frontier plot using Loom, OBS, or Windows Game Bar (Win+Alt+R). |
| **P5** | **Push to GitHub (If URL Required)** | Initialize git, commit the clean workspace (excluding `.venv`), create a public repo, and push code. |
| **P6** | **Submit on Unstop** | Copy abstract from `deliverables/ABSTRACT_SUBMISSION.md`, attach links/files, and submit before 11:59 PM IST on October 14, 2026. |
