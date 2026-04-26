# ML-based-Portfolio-Optimization

An interactive **machine learning-powered portfolio optimization system** that helps users allocate assets optimally based on risk and return.

This project combines **financial theory (Mean-Variance Optimization)** with **machine learning (XGBoost)** to create a practical decision-support tool for portfolio management.

---

## 🚀 Features

* 📈 **Portfolio Optimization**

  * Mean-Variance Optimization (Modern Portfolio Theory)
  * Risk-adjusted allocation with constraints

* 🤖 **ML-Based Predictions**

  * Predicts future asset returns using XGBoost
  * Uses lag-based features from historical returns

* ⚙️ **Interactive Dashboard (Streamlit)**

  * Choose optimization strategy:

    * Historical
    * ML-Based
  * Adjust risk aversion dynamically
  * Real-time portfolio allocation updates

* 📊 **Performance Metrics**

  * Expected Return
  * Risk (Volatility)
  * Sharpe Ratio

* 📌 **Visualization**

  * Asset allocation table
  * Bar chart & pie chart
  * Insights based on performance

---

## 🧠 Problem Statement

How should an investor allocate capital across multiple assets to:

* Maximize return
* Minimize risk

Traditional approaches rely on historical averages, which may not reflect future behavior.

👉 This project improves upon that by integrating **machine learning predictions** into the optimization process.

---

## 🏗️ System Architecture

```
Data Collection → Data Preprocessing → ML Model → Predicted Returns → Optimization → Dashboard
```

---

## 📂 Project Structure

```
portfolio-optimization/
│
├── data/
│   ├── stock_prices.csv
│   ├── returns.csv
│   ├── mean_returns.csv
│   ├── cov_matrix.csv
│   ├── predicted_returns.csv
│
├── data_collection.py
├── data_preprocessing.py
├── ml_model.py
├── portfolio_optimization.py
├── evaluation.py
├── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/portfolio-optimization.git
cd portfolio-optimization
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Prepare Data

```bash
python data_collection.py
python data_preprocessing.py
```

---

### Step 2: Train ML Model

```bash
python ml_model.py
```

---

### Step 3: Run Dashboard

```bash
streamlit run app.py
```

---

## 📊 Optimization Model

The portfolio is optimized using:

[
\max \left( w^T \mu - \lambda \cdot w^T \Sigma w \right)
]

Where:

* ( w ) = asset weights
* ( \mu ) = expected returns
* ( \Sigma ) = covariance matrix
* ( \lambda ) = risk aversion

---

## 📈 Results & Insights

| Portfolio Type | Return   | Risk     | Sharpe           |
| -------------- | -------- | -------- | ---------------- |
| Equal Weight   | Moderate | Moderate | Stable           |
| Historical MVO | High     | High     | Best (in-sample) |
| ML-Based       | Lower    | Lower    | Conservative     |

### 🔍 Key Observations

* Historical optimization performs best **in-sample**
* ML-based portfolio produces **lower volatility**
* Predicting daily returns is inherently noisy → ML signals are weak but useful for **risk control**

---

## ⚠️ Limitations

* ML predictions are based on **lag features only**
* No macroeconomic indicators included
* Short-term return prediction is inherently difficult
* No backtesting (future improvement)

---

## 🔮 Future Improvements

* Add **rolling window backtesting**
* Include **macroeconomic indicators**
* Use **LSTM / deep learning models**
* Add **user-selected assets**
* Deploy as a web app

---

## 💡 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn, XGBoost
* SciPy (Optimization)
* Streamlit (UI)

---

## 📌 Key Learnings

* Trade-off between **risk and return**
* Importance of **covariance in diversification**
* ML predictions in finance are **weak but directional**
* Combining ML with optimization creates **practical systems**

---

## ⭐ If you found this useful

Give this repo a ⭐ and feel free to contribute!
