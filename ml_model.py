import pandas as pd
from xgboost import XGBRegressor

# Load returns
returns = pd.read_csv("data/returns.csv", index_col="Date", parse_dates=True)

# -------------------------------
# Step 1: Create lag features per asset
# -------------------------------
lag = 5
df = returns.copy()

for col in returns.columns:
    for i in range(1, lag + 1):
        df[f"{col}_lag_{i}"] = returns[col].shift(i)

df = df.dropna()

# -------------------------------
# Step 2: Train model per asset
# -------------------------------
models = {}
predicted_returns = {}

# Features (same for all models)
X = df.drop(columns=returns.columns)

for col in returns.columns:
    y = df[col]

    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
    model.fit(X, y)

    models[col] = model

    # Predict next return using latest data
    latest_features = X.iloc[-1:].values
    predicted_returns[col] = model.predict(latest_features)[0]

# -------------------------------
# Step 3: Convert to Series
# -------------------------------
predicted_returns = pd.Series(predicted_returns)

print("\nPredicted Returns (per asset):\n")
print(predicted_returns)

predicted_returns.to_csv("data/predicted_returns.csv")