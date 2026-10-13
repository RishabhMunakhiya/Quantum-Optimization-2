"""
Data Loader & Financial Metrics Calculator
Calculates expected annual returns, covariance matrices, and correlation matrices
for stock portfolio selection.
"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd


class PortfolioDataLoader:
    """
    Loads price history and extracts statistical moments for Markowitz portfolio optimization.
    """

    def __init__(self, csv_path: str = None):
        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, "data", "stock_data.csv")
        self.csv_path = csv_path
        self.df_prices = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        self.daily_returns = self.df_prices.pct_change().dropna()

    @property
    def available_tickers(self) -> List[str]:
        return list(self.df_prices.columns)

    def get_portfolio_data(
        self,
        tickers: List[str] = None,
        annualization_factor: int = 252
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extracts financial statistics for the specified subset of tickers.
        
        Returns:
            expected_returns: Vector mu of annualized expected returns (shape: n)
            covariances: Matrix Sigma of annualized covariances (shape: n x n)
            correlations: Correlation matrix (shape: n x n)
            selected_tickers: List of ticker strings
        """
        if tickers is None:
            tickers = self.available_tickers
        else:
            missing = [t for t in tickers if t not in self.df_prices.columns]
            if missing:
                raise ValueError(f"Tickers not found in dataset: {missing}")

        subset_returns = self.daily_returns[tickers]

        # Annualized expected return: mean daily return * 252
        expected_returns = subset_returns.mean().values * annualization_factor

        # Annualized covariance matrix: cov(daily returns) * 252
        covariances = subset_returns.cov().values * annualization_factor

        # Correlation matrix: cov_ij / (sigma_i * sigma_j)
        correlations = subset_returns.corr().values

        return expected_returns, covariances, correlations, tickers

    def get_subset(
        self,
        num_assets: int = 4,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Picks a diverse subset of n assets for scaling benchmarks.
        """
        total = len(self.available_tickers)
        if num_assets > total:
            raise ValueError(f"Requested {num_assets} assets, but only {total} available.")

        # Fixed deterministic subsets for standard reproducibility
        if num_assets == 3:
            chosen = ["AAPL", "JPM", "XOM"]
        elif num_assets == 4:
            chosen = ["AAPL", "MSFT", "JPM", "JNJ"]
        elif num_assets == 6:
            chosen = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "XOM"]
        elif num_assets == 8:
            chosen = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "PG"]
        else:
            chosen = self.available_tickers[:num_assets]

        return self.get_portfolio_data(chosen)
