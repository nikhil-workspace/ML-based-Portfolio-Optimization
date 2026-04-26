import numpy as np
import pandas as pd

# -------------------------------
# Load data
# -------------------------------

returns = pd.read_csv("data/returns.csv", index_col="Date", parse_dates=True)
cov_matrix = pd.read_csv("data/cov_matrix.csv", index_col=0)

# Historical mean returns
mean_returns_hist = returns.mean()

# ML predicted returns
from ml_model import predicted_returns
mean_returns_ml = predicted_returns.astype(float)

# -------------------------------
# Portfolio weights
# -------------------------------

# Equal weight
num_assets = len(mean_returns_hist)
weights_equal = np.array([1/num_assets] * num_assets)

# Historical optimization (PASTE your weights)
weights_hist = np.array([
    0.154714,
    0.152203,
    0.098195,
    0.102132,
    0.300000,
    0.073031,
    0.063723,
    0.056002
])

# ML optimization (PASTE your weights)
weights_ml = np.array([
    0.120822,
    0.121830,
    0.068502,
    0.069389,
    0.120496,
    0.135067,
    0.063895,
    0.300000
])

# -------------------------------
# Evaluation function
# -------------------------------

def evaluate(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / risk
    return ret, risk, sharpe

# -------------------------------
# Evaluate all
# -------------------------------

ret_eq, risk_eq, sharpe_eq = evaluate(weights_equal, mean_returns_hist, cov_matrix)
ret_hist, risk_hist, sharpe_hist = evaluate(weights_hist, mean_returns_hist, cov_matrix)
ret_ml, risk_ml, sharpe_ml = evaluate(weights_ml, mean_returns_ml, cov_matrix)

# -------------------------------
# Print results
# -------------------------------

print("\n=== Portfolio Comparison ===\n")

print("Equal Weight:")
print("Return:", ret_eq)
print("Risk:", risk_eq)
print("Sharpe:", sharpe_eq)

print("\nHistorical Optimization:")
print("Return:", ret_hist)
print("Risk:", risk_hist)
print("Sharpe:", sharpe_hist)

print("\nML-Based Optimization:")
print("Return:", ret_ml)
print("Risk:", risk_ml)
print("Sharpe:", sharpe_ml)