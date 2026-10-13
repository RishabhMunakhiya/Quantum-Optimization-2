"""
Mathematical Formulation: Markowitz Portfolio Optimization to QUBO and Ising Hamiltonian
Supports constraint penalization, Pauli Z mapping, and portfolio metric evaluation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp


class PortfolioQUBO:
    """
    Formulates the Markowitz Mean-Variance Portfolio Selection problem as a
    Quadratic Program, QUBO, and Quantum Ising Hamiltonian.

    Mathematical Objective:
        Minimize: q * x^T * Sigma * x - mu^T * x
        Subject to: sum(x_i) == B,  where x_i in {0, 1}

    QUBO Mapping (with penalty A):
        Minimize: q * x^T * Sigma * x - mu^T * x + A * (sum(x_i) - B)^2
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariances: np.ndarray,
        budget: int,
        risk_factor: float = 0.5,
        penalty: Optional[float] = None,
        tickers: Optional[List[str]] = None
    ):
        self.expected_returns = np.asarray(expected_returns, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        self.n_assets = len(self.expected_returns)
        self.budget = int(budget)
        self.risk_factor = float(risk_factor)
        self.tickers = tickers if tickers is not None else [f"Asset_{i}" for i in range(self.n_assets)]

        if self.covariances.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"Covariance shape {self.covariances.shape} does not match {self.n_assets} assets.")

        if self.budget <= 0 or self.budget > self.n_assets:
            raise ValueError(f"Budget B={self.budget} must be between 1 and {self.n_assets}.")

        # Auto-calibrate penalty parameter A if not provided
        if penalty is None:
            # Upper bound on return variance difference to strictly penalize invalid states
            max_return = np.max(np.abs(self.expected_returns)) * self.n_assets
            max_risk = self.risk_factor * np.max(self.covariances) * (self.n_assets ** 2)
            self.penalty = float(max(2.0, (max_return + max_risk) * 1.5))
        else:
            self.penalty = float(penalty)

        self._qp = None
        self._qubo = None
        self._ising_operator = None
        self._ising_offset = None

    def build_quadratic_program(self) -> QuadraticProgram:
        """Constructs the constrained Quadratic Program."""
        qp = QuadraticProgram("Markowitz_Portfolio_Optimization")

        # Binary decision variables x_0, x_1, ...
        for i in range(self.n_assets):
            qp.binary_var(name=f"x_{i}")

        # Linear objective: -mu_i
        linear_dict = {f"x_{i}": -float(self.expected_returns[i]) for i in range(self.n_assets)}

        # Quadratic objective: q * Sigma_ij
        quadratic_dict = {}
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                quadratic_dict[(f"x_{i}", f"x_{j}")] = float(self.risk_factor * self.covariances[i, j])

        qp.minimize(linear=linear_dict, quadratic=quadratic_dict)

        # Cardinality / Budget Constraint: sum(x_i) == B
        budget_linear = {f"x_{i}": 1.0 for i in range(self.n_assets)}
        qp.linear_constraint(linear=budget_linear, sense="==", rhs=float(self.budget), name="budget_constraint")

        self._qp = qp
        return qp

    def to_qubo(self) -> QuadraticProgram:
        """Converts the constrained problem into an unconstrained QUBO via quadratic penalty."""
        if self._qp is None:
            self.build_quadratic_program()
        conv = QuadraticProgramToQubo(penalty=self.penalty)
        self._qubo = conv.convert(self._qp)
        return self._qubo

    def to_ising(self) -> Tuple[SparsePauliOp, float]:
        """
        Converts the QUBO problem into an Ising Hamiltonian operator and energy offset.
        Uses x_i = (I - Z_i)/2.
        """
        if self._qubo is None:
            self.to_qubo()
        self._ising_operator, self._ising_offset = self._qubo.to_ising()
        return self._ising_operator, self._ising_offset

    def evaluate_solution(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates a binary solution vector x in {0, 1}^n.
        
        Returns full financial and mathematical metrics:
        expected return, volatility, Sharpe ratio, budget violation, cost.
        """
        x = np.asarray(x, dtype=int).flatten()
        if len(x) != self.n_assets:
            raise ValueError(f"Solution vector length {len(x)} does not match {self.n_assets}")

        selected_indices = np.where(x == 1)[0].tolist()
        selected_tickers = [self.tickers[i] for i in selected_indices]
        num_selected = len(selected_indices)

        # Budget constraint
        constraint_violation = abs(num_selected - self.budget)
        is_valid = (constraint_violation == 0)

        # Total expected return & risk of unweighted selection
        # (Assuming equal weights 1/B among selected assets when valid)
        if num_selected > 0:
            weights = np.zeros(self.n_assets)
            weights[selected_indices] = 1.0 / num_selected
            portfolio_expected_return = float(np.dot(weights, self.expected_returns))
            portfolio_variance = float(weights.T @ self.covariances @ weights)
            portfolio_volatility = float(np.sqrt(max(1e-8, portfolio_variance)))
            # Sharpe Ratio with 3% risk-free rate assumption
            risk_free_rate = 0.03
            sharpe_ratio = float((portfolio_expected_return - risk_free_rate) / portfolio_volatility)
        else:
            portfolio_expected_return = 0.0
            portfolio_variance = 0.0
            portfolio_volatility = 0.0
            sharpe_ratio = 0.0

        # Markowitz mathematical objective: q * x^T Sigma x - mu^T x
        unweighted_risk_term = float(self.risk_factor * (x.T @ self.covariances @ x))
        unweighted_return_term = float(np.dot(self.expected_returns, x))
        markowitz_cost = unweighted_risk_term - unweighted_return_term

        # QUBO cost with penalty:
        qubo_cost = markowitz_cost + self.penalty * (num_selected - self.budget) ** 2

        return {
            "binary_vector": x.tolist(),
            "bitstring": "".join(str(b) for b in x[::-1]),  # Qiskit qubit order: x_n-1 ... x_0
            "selected_tickers": selected_tickers,
            "num_selected": num_selected,
            "budget": self.budget,
            "is_valid": is_valid,
            "constraint_violation": constraint_violation,
            "expected_return": portfolio_expected_return,
            "volatility": portfolio_volatility,
            "variance": portfolio_variance,
            "sharpe_ratio": sharpe_ratio,
            "markowitz_cost": markowitz_cost,
            "qubo_cost": qubo_cost
        }
