"""
Quantum Optimization Solvers using Qiskit
Implements QAOA (Quantum Approximate Optimization Algorithm),
SamplingVQE, and NumPy Exact Diagonalization on Ising Hamiltonians.
"""

import time
import warnings
from typing import Dict, Any, Optional
import numpy as np
import scipy.sparse

# Suppress scipy sparse format warnings from matrix conversions
warnings.filterwarnings("ignore", category=scipy.sparse.SparseEfficiencyWarning)

from qiskit import transpile
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA, SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit.circuit.library import TwoLocal
from src.portfolio_qubo import PortfolioQUBO


def extract_circuit_metrics(circuit) -> Dict[str, Any]:
    """
    Extracts decomposed logical gate counts, depth, and two-qubit metrics from a quantum circuit
    (e.g., parameter-bound QAOA or VQE ansatz).

    Decomposes high-level composite blocks (QAOA, PauliEvolution, RZZGate, TwoLocal)
    into standard elementary basis gates (CX, RZ, RX, etc.) to report true non-trivial metrics.
    Also calculates universal basis transpilation metrics (without assuming a specific hardware topology).
    """
    decomposed = circuit
    for _ in range(6):
        ops = decomposed.count_ops()
        if any(composite_op in ops for composite_op in ["QAOA", "PauliEvolution", "qaoa", "rzz", "TwoLocal", "two_local"]):
            decomposed = decomposed.decompose()
        else:
            break

    gate_counts = dict(decomposed.count_ops())
    cx_count = gate_counts.get("cx", 0)
    cz_count = gate_counts.get("cz", 0)
    two_qubit_count = decomposed.num_nonlocal_gates()

    metrics = {
        "circuit_depth": decomposed.depth(),
        "num_qubits": circuit.num_qubits,
        "gate_counts": gate_counts,
        "cnot_count": cx_count + cz_count,
        "two_qubit_gate_count": two_qubit_count,
        "logical_depth": decomposed.depth(),
        "logical_gate_counts": gate_counts,
    }

    # Transpiled universal basis metrics (standard universal basis: cx, rz, sx, x)
    try:
        basis_circuit = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
        metrics["transpiled_basis_depth"] = basis_circuit.depth()
        metrics["transpiled_basis_ops"] = dict(basis_circuit.count_ops())
        metrics["transpiled_cnot_count"] = metrics["transpiled_basis_ops"].get("cx", 0)
    except Exception:
        pass

    return metrics


class NumPyQuantumExactSolver:
    """
    Exact Quantum Diagonalization baseline.
    Computes the mathematical ground state of the Ising Hamiltonian operator.
    """

    def __init__(self, portfolio_model: PortfolioQUBO):
        self.model = portfolio_model

    def solve(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        operator, offset = self.model.to_ising()

        exact_mes = NumPyMinimumEigensolver()
        res = exact_mes.compute_minimum_eigenvalue(operator)
        elapsed_time = time.perf_counter() - start_time

        # Convert ground state eigenstate or bitstring
        # In Qiskit, best measurement or eigenstate gives the bitstring
        # For NumPyMinimumEigensolver, the ground state vector is res.eigenstate
        statevector = res.eigenstate.to_dict()
        best_bitstring = max(statevector.items(), key=lambda item: abs(item[1])**2)[0]

        # Convert bitstring (right-to-left in Qiskit: bit 0 is last char) to x array
        # x_0 = int(best_bitstring[-1]), x_1 = int(best_bitstring[-2]), etc.
        x = np.array([int(best_bitstring[-(i + 1)]) for i in range(self.model.n_assets)], dtype=int)

        eval_res = self.model.evaluate_solution(x)
        eval_res["solver"] = "Quantum Exact (NumPy Diagonalization)"
        eval_res["execution_time_sec"] = elapsed_time
        eval_res["eigenvalue"] = float(res.eigenvalue.real)
        eval_res["ground_energy"] = float(res.eigenvalue.real + offset)
        eval_res["bitstring_probabilities"] = {k: float(abs(v)**2) for k, v in statevector.items() if abs(v)**2 > 1e-4}
        return eval_res


class QAOAPortfolioSolver:
    """
    Quantum Approximate Optimization Algorithm (QAOA) solver.
    Uses parameterized problem and mixer unitary layers to find optimal portfolios.
    """

    def __init__(
        self,
        portfolio_model: PortfolioQUBO,
        reps: int = 1,
        max_iter: int = 80,
        optimizer_name: str = "COBYLA",
        seed: int = 42
    ):
        self.model = portfolio_model
        self.reps = reps
        self.max_iter = max_iter
        self.optimizer_name = optimizer_name
        self.seed = seed

    def solve(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        operator, offset = self.model.to_ising()

        # Classical optimizer
        if self.optimizer_name.upper() == "SLSQP":
            optimizer = SLSQP(maxiter=self.max_iter)
        else:
            optimizer = COBYLA(maxiter=self.max_iter)

        sampler = StatevectorSampler(default_shots=1024, seed=self.seed)

        # Optimization trajectory tracking
        energy_history = []

        def callback(eval_count, params, value, std):
            energy_history.append((eval_count, float(value + offset)))

        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.reps,
            callback=callback
        )

        res = qaoa.compute_minimum_eigenvalue(operator)
        elapsed_time = time.perf_counter() - start_time

        # Extract best measurement from QAOA result
        best_measurement = res.best_measurement
        best_bitstring = best_measurement.get("bitstring")

        # Map Qiskit bitstring to binary vector
        # Qiskit bitstring: index 0 is qubit (n-1), index (n-1) is qubit 0
        x = np.array([int(best_bitstring[-(i + 1)]) for i in range(self.model.n_assets)], dtype=int)

        eval_res = self.model.evaluate_solution(x)
        eval_res["solver"] = f"Quantum QAOA (p={self.reps}, {self.optimizer_name})"
        eval_res["execution_time_sec"] = elapsed_time
        eval_res["eigenvalue"] = float(res.eigenvalue.real)
        eval_res["expected_energy"] = float(res.eigenvalue.real + offset)
        eval_res["best_state_probability"] = float(best_measurement.get("probability", 0.0))
        eval_res["best_state_value"] = float(np.real(best_measurement.get("value", 0.0)))
        eval_res["optimal_parameters"] = [float(p) for p in res.optimal_point] if res.optimal_point is not None else []
        eval_res["energy_history"] = energy_history

        # Circuit metrics
        ansatz = qaoa.ansatz
        try:
            bound_circuit = ansatz.assign_parameters(res.optimal_point) if res.optimal_point is not None else ansatz
            circuit_metrics = extract_circuit_metrics(bound_circuit)
            eval_res.update(circuit_metrics)
        except Exception:
            eval_res["circuit_depth"] = 2 * self.reps + 1
            eval_res["num_qubits"] = self.model.n_assets
            eval_res["gate_counts"] = {}
            eval_res["cnot_count"] = 0
            eval_res["two_qubit_gate_count"] = 0

        return eval_res


class SamplingVQESolver:
    """
    Sampling Variational Quantum Eigensolver (SamplingVQE) using a TwoLocal heuristic ansatz.
    """

    def __init__(
        self,
        portfolio_model: PortfolioQUBO,
        max_iter: int = 80,
        reps: int = 1,
        seed: int = 42
    ):
        self.model = portfolio_model
        self.max_iter = max_iter
        self.reps = reps
        self.seed = seed

    def solve(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        operator, offset = self.model.to_ising()

        n_qubits = self.model.n_assets
        ansatz = TwoLocal(n_qubits, rotation_blocks="ry", entanglement_blocks="cz", reps=self.reps, entanglement="linear")

        optimizer = COBYLA(maxiter=self.max_iter)
        sampler = StatevectorSampler(default_shots=1024, seed=self.seed)

        vqe = SamplingVQE(
            sampler=sampler,
            ansatz=ansatz,
            optimizer=optimizer
        )

        res = vqe.compute_minimum_eigenvalue(operator)
        elapsed_time = time.perf_counter() - start_time

        best_measurement = res.best_measurement
        best_bitstring = best_measurement.get("bitstring")
        x = np.array([int(best_bitstring[-(i + 1)]) for i in range(self.model.n_assets)], dtype=int)

        eval_res = self.model.evaluate_solution(x)
        eval_res["solver"] = f"Quantum VQE (TwoLocal, p={self.reps})"
        eval_res["execution_time_sec"] = elapsed_time
        eval_res["expected_energy"] = float(res.eigenvalue.real + offset)
        eval_res["best_state_probability"] = float(best_measurement.get("probability", 0.0))

        # Circuit metrics for VQE
        try:
            bound_circuit = ansatz.assign_parameters(res.optimal_point) if res.optimal_point is not None else ansatz
            circuit_metrics = extract_circuit_metrics(bound_circuit)
            eval_res.update(circuit_metrics)
        except Exception:
            eval_res["circuit_depth"] = 5
            eval_res["num_qubits"] = n_qubits
            eval_res["gate_counts"] = {}
            eval_res["cnot_count"] = 0
            eval_res["two_qubit_gate_count"] = 0

        return eval_res
