import pandas as pd

# Load prices
prices = pd.read_csv("data/stock_prices.csv", index_col="Date", parse_dates=True)

# Compute returns
returns = prices.pct_change().dropna()

# Save
returns.to_csv("data/returns.csv")

# PRINT RETURNS 
print("Shape:", returns.shape)
print(returns.head())
print(returns.describe())

'''Mean (expected daily return)
AAPL ≈ 0.00123 → 0.123% per day
TSLA ≈ 0.00243 → highest return (but check risk)

TSLA looks attractive

Standard deviation (risk)
TSLA ≈ 0.040 → very high volatility
GLD ≈ 0.009 → very stable

High return ≠ best asset (risk matters)

Extreme values
TSLA min ≈ -21% → huge downside risk
MSFT min ≈ -14%
'''

#risk and covariance matrix

'''
1.Expected returns vector (μ)
2.Covariance matrix (Σ)
'''
#expected returns
mean_returns = returns.mean()
print("\n",mean_returns)

#covariance matrix
cov_matrix = returns.cov()
print(cov_matrix)

'''
What covariance means
1.Measures how assets move together
2.Helps reduce risk via diversification
'''

#Portfolio Risk = w^T x Σw

mean_returns.to_csv("data/mean_returns.csv")
cov_matrix.to_csv("data/cov_matrix.csv")