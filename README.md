# 📈 Stock Price Predictor

A machine learning project that predicts stock prices using historical data and technical indicators.

## 🎯 Project Overview

This project demonstrates multiple ML techniques for time-series financial forecasting:

| Model | Type | Description |
|---|---|---|
| **Linear Regression** | Baseline | Simple, interpretable baseline model |
| **Random Forest** | Ensemble | Handles non-linearity and feature interactions |
| **LSTM Neural Network** | Deep Learning | Captures long-term sequential patterns |

---

## 🗂️ Project Structure

```
stock-price-predictor/
│
├── stock_predictor.py       # Main script — runs all models
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── data/
│   └── stock_data.csv       # Generated after running the script
│
├── models/
│   ├── prediction_results.png   # Chart: actual vs predicted
│   ├── model_comparison.png     # Chart: RMSE & R² comparison
│   └── metrics.csv              # Numerical results
│
└── notebooks/
    └── (add Jupyter notebooks here for exploration)
```

---

## ⚙️ Features Used

The models are trained on the following **technical indicators** derived from raw OHLCV data:

- **Moving Averages** — MA-7, MA-20, MA-50
- **Bollinger Bands** — upper and lower bands
- **RSI** — Relative Strength Index (14-day)
- **MACD** — Moving Average Convergence Divergence
- **Momentum** — 5-day price change
- **Volatility** — 10-day rolling standard deviation
- **Day of Week** — captures weekly seasonality

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow (for LSTM) is optional. If not installed, only Linear Regression and Random Forest will run.

### 3. Run the predictor
```bash
python stock_predictor.py
```

### Expected Output
```
==================================================
   STOCK PRICE PREDICTOR
==================================================
✅  Generated 1000 days of synthetic AAPL data → data/stock_data.csv

────────────────────────────────────────
  Linear Regression
  RMSE : 2.3451
  MAE  : 1.8234
  R²   : 0.9823

────────────────────────────────────────
  Random Forest
  RMSE : 1.1023
  MAE  : 0.8901
  R²   : 0.9961
...
```

---

## 📊 Sample Results

After running, you'll find these charts in the `models/` folder:

- **prediction_results.png** — Side-by-side comparison of all models vs actual prices
- **model_comparison.png** — Bar charts comparing RMSE and R² scores

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **scikit-learn** — Linear Regression & Random Forest
- **TensorFlow / Keras** — LSTM Neural Network
- **Matplotlib** — visualizations

---

## 📚 Concepts Learned

| Concept | Applied Where |
|---|---|
| Feature Engineering | Technical indicators (RSI, MACD, Bollinger Bands) |
| Train/Test Split | Time-aware split (no shuffle) |
| Data Normalization | MinMaxScaler for neural network |
| Model Evaluation | RMSE, MAE, R² metrics |
| Deep Learning | LSTM for sequential data |

---

## 🔮 Future Improvements

- [ ] Integrate live data via Yahoo Finance API (`yfinance`)
- [ ] Add sentiment analysis from news headlines
- [ ] Deploy as a web app using Streamlit
- [ ] Add hyperparameter tuning with GridSearchCV

---

## 👨‍💻 Author

**[prasoon khandelwal]**  
Internship Project — AI/ML Domain  
GitHub: [@prasun1509](https://github.com/prasun1509)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
