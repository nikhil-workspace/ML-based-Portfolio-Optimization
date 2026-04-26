import yfinance as yf
import pandas as pd

# Step 1: Define assets
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "TSLA", "JPM", "XOM", "GLD"
]

# Step 2: Download data
data = yf.download(
    tickers,
    start="2018-01-01",
    end="2024-01-01",
    group_by="ticker",
    auto_adjust=False
)

# Step 3: Extract Adjusted Close
adj_close = pd.DataFrame()

for ticker in tickers:
    adj_close[ticker] = data[ticker]["Adj Close"]

# Step 4: Handle missing values (important)
adj_close = adj_close.dropna()

# Step 5: Save dataset
adj_close.to_csv("data/stock_prices.csv")

# Step 6: Basic checks
print("Shape:", adj_close.shape)
print(adj_close.head())