import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# -------------------------------
# Load data
# -------------------------------
returns = pd.read_csv("data/returns.csv", index_col="Date", parse_dates=True)
cov_matrix = pd.read_csv("data/cov_matrix.csv", index_col=0)

mean_returns_hist = returns.mean()

# ML predicted returns
mean_returns_ml = pd.read_csv("data/predicted_returns.csv", index_col=0).squeeze()

# -------------------------------
# Portfolio functions
# -------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / risk
    return ret, risk, sharpe

def optimize_portfolio(mean_returns, cov_matrix, risk_aversion):
    num_assets = len(mean_returns)

    def objective(weights):
        ret, risk, _ = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(ret - risk_aversion * risk)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.3) for _ in range(num_assets))
    init_guess = np.array([1/num_assets]*num_assets)

    result = minimize(objective, init_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result.x

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Settings")

model_type = st.sidebar.selectbox(
    "Optimization Strategy",
    ["Historical", "ML-Based"]
)

risk_aversion = st.sidebar.slider(
    "Risk Aversion",
    0.01, 1.0, 0.1
)

# -------------------------------
# Select mean returns
# -------------------------------
if model_type == "Historical":
    mean_returns = mean_returns_hist
else:
    mean_returns = mean_returns_ml

# Stability (important for ML)
mean_returns = mean_returns.clip(-0.01, 0.01)

# -------------------------------
# Optimize
# -------------------------------
weights = optimize_portfolio(mean_returns, cov_matrix, risk_aversion)

portfolio = pd.Series(weights, index=mean_returns.index).round(4)

ret, risk, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)

# -------------------------------
# Main Dashboard
# -------------------------------
st.title("📊 ML Portfolio Optimization Dashboard")

col_left, col_right = st.columns([2, 1])

# -------------------------------
# LEFT: Table + Chart
# -------------------------------
with col_left:
    st.subheader("📌 Portfolio Allocation")

    portfolio_df = portfolio.reset_index()
    portfolio_df.columns = ["Asset", "Weight"]

    # Convert to percentage
    portfolio_df["Weight (%)"] = (portfolio_df["Weight"] * 100).round(2)
    portfolio_df = portfolio_df.drop(columns=["Weight"])

    st.dataframe(portfolio_df, use_container_width=True)

    st.subheader("📊 Allocation Chart")
    st.bar_chart(portfolio)

# -------------------------------
# RIGHT: Metrics + Insights
# -------------------------------
with col_right:
    st.subheader("📈 Performance")

    st.metric("Return", f"{ret:.4f}")
    st.metric("Risk", f"{risk:.4f}")
    st.metric("Sharpe Ratio", f"{sharpe:.4f}")

    st.subheader("🧠 Insights")

    if sharpe > 0.07:
        st.success("Strong portfolio (high Sharpe)")
    elif sharpe > 0.05:
        st.warning("Balanced portfolio")
    else:
        st.error("Low performance – adjust risk or model")

# -------------------------------
# Pie Chart (Professional touch)
# -------------------------------
st.subheader("🥧 Portfolio Distribution")
st.pyplot(portfolio.plot.pie(autopct='%1.1f%%').figure)