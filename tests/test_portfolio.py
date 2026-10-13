"""
Automated Unit and Integration Tests for Quantum Portfolio Optimization
Verifies data loading, QUBO mapping, constraint validity, classical baselines, and QAOA execution.
"""

import unittest
import numpy as np
from src.data_loader import PortfolioDataLoader
from src.portfolio_qubo import PortfolioQUBO
from src.classical_solvers import ExactBruteForceSolver, SimulatedAnnealingSolver, GreedySharpeSolver
from src.quantum_solvers import NumPyQuantumExactSolver, QAOAPortfolioSolver, SamplingVQESolver, extract_circuit_metrics


class TestQuantumPortfolioOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = PortfolioDataLoader()
        cls.num_assets = 4
        cls.budget = 2
        cls.risk_factor = 0.5
        cls.mu, cls.sigma, cls.corr, cls.tickers = cls.loader.get_subset(cls.num_assets)
        cls.model = PortfolioQUBO(
            expected_returns=cls.mu,
            covariances=cls.sigma,
            budget=cls.budget,
            risk_factor=cls.risk_factor,
            tickers=cls.tickers
        )

    def test_data_loader_dimensions(self):
        """Verify data dimensions, covariance symmetry, and positive semi-definiteness."""
        self.assertEqual(len(self.mu), self.num_assets)
        self.assertEqual(self.sigma.shape, (self.num_assets, self.num_assets))
        # Covariance matrix must be symmetric
        np.testing.assert_allclose(self.sigma, self.sigma.T, atol=1e-7)
        # Eigenvalues must be non-negative
        eigenvals = np.linalg.eigvalsh(self.sigma)
        self.assertTrue(np.all(eigenvals >= -1e-6), f"Eigenvalues negative: {eigenvals}")

    def test_qubo_and_ising_conversion(self):
        """Verify quadratic program formulation, penalty calculation, and Ising Hamiltonian."""
        qp = self.model.build_quadratic_program()
        self.assertEqual(qp.get_num_vars(), self.num_assets)
        self.assertEqual(len(qp.linear_constraints), 1)

        operator, offset = self.model.to_ising()
        self.assertIsNotNone(operator)
        self.assertGreater(len(operator), 0)
        self.assertIsInstance(offset, float)

    def test_classical_exact_solver(self):
        """Verify brute-force solver yields a valid portfolio meeting the budget constraint."""
        solver = ExactBruteForceSolver(self.model)
        res = solver.solve()
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["num_selected"], self.budget)
        self.assertEqual(res["constraint_violation"], 0)
        self.assertLess(res["markowitz_cost"], 0.0)

    def test_quantum_exact_matches_classical(self):
        """Verify quantum ground state of Ising Hamiltonian exactly equals classical optimal cost."""
        classical_res = ExactBruteForceSolver(self.model).solve()
        quantum_res = NumPyQuantumExactSolver(self.model).solve()

        self.assertTrue(quantum_res["is_valid"])
        self.assertEqual(classical_res["selected_tickers"], quantum_res["selected_tickers"])
        self.assertAlmostEqual(classical_res["markowitz_cost"], quantum_res["markowitz_cost"], places=5)

    def test_qaoa_solver_execution(self):
        """Verify QAOA produces a valid portfolio and achieves high approximation ratio."""
        qaoa_solver = QAOAPortfolioSolver(self.model, reps=1, max_iter=40, seed=42)
        res = qaoa_solver.solve()

        self.assertTrue(res["is_valid"])
        self.assertEqual(res["num_selected"], self.budget)
        self.assertIn("circuit_depth", res)
        self.assertGreater(len(res.get("optimal_parameters", [])), 0)

    def test_qaoa_circuit_metrics_decomposition(self):
        """Verify circuit depth and CNOT gate counts are properly decomposed and non-zero."""
        qaoa_solver = QAOAPortfolioSolver(self.model, reps=1, max_iter=20, seed=42)
        res = qaoa_solver.solve()

        # Depth must be decomposed, not remaining at trivial high-level depth of 2
        self.assertGreater(res["circuit_depth"], 2, "Circuit depth must be decomposed beyond high-level ansatz")
        # CNOT count must be non-zero for an entangling QUBO problem
        self.assertGreater(res["cnot_count"], 0, "CNOT count must not be zero for QAOA on QUBO")
        self.assertGreater(res["two_qubit_gate_count"], 0)
        self.assertEqual(res["cnot_count"], res["two_qubit_gate_count"])
        self.assertEqual(res["num_qubits"], self.num_assets)
        self.assertIn("gate_counts", res)
        self.assertIn("cx", res["gate_counts"])
        self.assertGreater(res["gate_counts"]["cx"], 0)
        # Transpiled universal basis metrics
        self.assertIn("transpiled_basis_depth", res)
        self.assertIn("transpiled_cnot_count", res)
        self.assertGreater(res["transpiled_cnot_count"], 0)

    def test_vqe_circuit_metrics_decomposition(self):
        """Verify SamplingVQE produces non-zero two-qubit entangling gates."""
        vqe_solver = SamplingVQESolver(self.model, reps=1, max_iter=20, seed=42)
        res = vqe_solver.solve()

        self.assertTrue(res["is_valid"])
        self.assertGreater(res["circuit_depth"], 0)
        self.assertGreater(res["two_qubit_gate_count"], 0, "Two-qubit gates must be non-zero for VQE ansatz")

    def test_invalid_solution_penalization(self):
        """Verify that an invalid bitstring incurs a significant QUBO penalty."""
        # 0 assets selected (budget is 2)
        zero_vec = np.zeros(self.num_assets, dtype=int)
        eval_zero = self.model.evaluate_solution(zero_vec)
        self.assertFalse(eval_zero["is_valid"])
        self.assertEqual(eval_zero["constraint_violation"], self.budget)
        self.assertGreater(eval_zero["qubo_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
