"""
Module to generate or fetch stock price datasets for Portfolio Optimization.
Supports offline synthetic generation (realistic Geometric Brownian Motion with correlated assets)
and live fetching via yfinance with seamless fallback.
"""

import os
import numpy as np
import pandas as pd

DEFAULT_TICKERS = [
    "AAPL",   # Apple Inc. (Tech)
    "MSFT",   # Microsoft Corp. (Tech)
    "GOOGL",  # Alphabet Inc. (Tech / Comm)
    "AMZN",   # Amazon.com (Consumer / Cloud)
    "NVDA",   # Nvidia Corp. (Semiconductors / High Volatility)
    "JPM",    # JPMorgan Chase (Financials)
    "JNJ",    # Johnson & Johnson (Healthcare / Defensive)
    "XOM",    # Exxon Mobil (Energy / Commodities)
    "PG",     # Procter & Gamble (Consumer Staples / Low Volatility)
    "TSLA"    # Tesla Inc. (Auto / Tech / Growth)
]


def generate_synthetic_stock_data(
    tickers: list[str] = None,
    num_days: int = 252,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic daily closing prices for stocks using correlated Geometric Brownian Motion (GBM).
    
    Parameters:
        tickers: List of stock ticker symbols
        num_days: Number of trading days (default 252 = ~1 trading year)
        seed: Random seed for exact reproducibility
        
    Returns:
        pd.DataFrame with 'Date' as index and ticker symbols as columns
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
        
    np.random.seed(seed)
    n = len(tickers)
    
    # Realistic annual drift (mu) between 6% and 24%
    annual_returns = np.array([0.18, 0.20, 0.16, 0.15, 0.35, 0.12, 0.08, 0.11, 0.07, 0.28])[:n]
    if len(annual_returns) < n:
        annual_returns = np.pad(annual_returns, (0, n - len(annual_returns)), 'mean')
        
    # Realistic annual volatilities (sigma) between 12% and 48%
    annual_vols = np.array([0.22, 0.20, 0.23, 0.26, 0.42, 0.18, 0.13, 0.21, 0.12, 0.45])[:n]
    if len(annual_vols) < n:
        annual_vols = np.pad(annual_vols, (0, n - len(annual_vols)), 'mean')
        
    # Base correlation matrix structure
    base_corr = 0.35  # Market correlation baseline
    corr_matrix = np.full((n, n), base_corr)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Increase tech intra-correlations if first 5 are tech
    for i in range(min(5, n)):
        for j in range(min(5, n)):
            if i != j:
                corr_matrix[i, j] = 0.65
                
    # Ensure positive semi-definite
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    eigenvals = np.maximum(eigenvals, 1e-4)
    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d, d)
    
    # Cholesky decomposition for correlated shocks
    l_mat = np.linalg.cholesky(corr_matrix)
    
    dt = 1.0 / num_days
    daily_drift = (annual_returns - 0.5 * annual_vols**2) * dt
    daily_vol = annual_vols * np.sqrt(dt)
    
    # Initial prices
    initial_prices = np.array([180.0, 420.0, 160.0, 175.0, 120.0, 195.0, 155.0, 110.0, 165.0, 240.0])[:n]
    if len(initial_prices) < n:
        initial_prices = np.pad(initial_prices, (0, n - len(initial_prices)), constant_values=100.0)
        
    prices = np.zeros((num_days, n))
    prices[0] = initial_prices
    
    for t in range(1, num_days):
        uncorrelated_shocks = np.random.normal(0, 1, n)
        correlated_shocks = l_mat @ uncorrelated_shocks
        prices[t] = prices[t - 1] * np.exp(daily_drift + daily_vol * correlated_shocks)
        
    date_index = pd.date_range(start="2025-01-01", periods=num_days, freq="B")
    df = pd.DataFrame(prices, index=date_index, columns=tickers)
    return df


def fetch_or_load_stock_data(
    tickers: list[str] = None,
    output_path: str = None,
    force_synthetic: bool = False
) -> pd.DataFrame:
    """
    Attempts to download live historical prices via yfinance; falls back automatically
    to calibrated synthetic data if unavailable or offline.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
        
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "stock_data.csv")
        
    # Check if existing CSV is already available
    if os.path.exists(output_path) and not force_synthetic:
        try:
            df = pd.read_csv(output_path, index_col=0, parse_dates=True)
            if all(t in df.columns for t in tickers):
                return df[tickers]
        except Exception:
            pass

    df = None
    if not force_synthetic:
        try:
            import yfinance as yf
            print("Attempting to fetch live market data via yfinance...")
            data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False)
            if "Close" in data.columns:
                temp_df = data["Close"]
            else:
                temp_df = data
            temp_df = temp_df.dropna()
            if len(temp_df) > 50 and all(t in temp_df.columns for t in tickers):
                df = temp_df
                print(f"Successfully fetched live market data for {len(tickers)} tickers.")
        except Exception as exc:
            print(f"Online download failed or offline ({exc}). Falling back to calibrated synthetic dataset.")
            df = None

    if df is None or len(df) < 50:
        print("Generating calibrated, reproducible synthetic stock dataset...")
        df = generate_synthetic_stock_data(tickers=tickers, num_days=252, seed=42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Stock data successfully saved to: {output_path} with {len(df)} rows.")
    return df


if __name__ == "__main__":
    fetch_or_load_stock_data()
