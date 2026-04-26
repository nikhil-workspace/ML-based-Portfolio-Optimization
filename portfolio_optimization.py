#How much weight should I assign to each asset?
'''
Find weights w such that:
1.Maximize return
2.Minimize risk
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# -------------------------------
# Load data
# -------------------------------

from ml_model import predicted_returns
mean_returns = predicted_returns.astype(float)

cov_matrix = pd.read_csv("data/cov_matrix.csv", index_col=0)

num_assets = len(mean_returns)

# -------------------------------
# Portfolio performance
# -------------------------------

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    risk = np.sqrt(variance)  # ✅ corrected (std deviation instead of variance)
    return returns, risk


def objective_function(weights, mean_returns, cov_matrix, risk_aversion=0.1):
    returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(returns - risk_aversion * risk)

# -------------------------------
# Constraints
# -------------------------------

constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
)

# Max 30% per asset (for diversification)
bounds = tuple((0, 0.3) for _ in range(num_assets))  

# Initial guess
init_guess = np.array([1. / num_assets] * num_assets)

# -------------------------------
# Optimization
# -------------------------------

result = minimize(
    objective_function,
    init_guess,
    args=(mean_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x

# -------------------------------
# Output
# -------------------------------

portfolio = pd.Series(optimal_weights, index=mean_returns.index)
print("\nOptimal Portfolio Allocation:\n")
print(portfolio)

ret, risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

print("\nExpected Return:", ret)
print("Risk (Std Dev):", risk)
print("Sharpe Ratio:", ret / risk)

'''
TSLA (30%)
- Still highest return driver
- But now controlled

Tech cluster (~50% combined)
AAPL + MSFT + GOOGL + AMZN
- Stable growth backbone

GLD (5.6%)
- Low return but reduces overall risk
- Acts as insurance

JPM & XOM
- Add sector diversification
- Reduce correlation with tech
'''