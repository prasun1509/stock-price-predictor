"""
Stock Price Predictor
=====================
Predicts future stock prices using historical data.
Techniques used:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - LSTM Neural Network (advanced)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Try importing optional deep-learning library ──────────────────────────────
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("TensorFlow not installed – LSTM model will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION  (simulates realistic stock data so no API key is needed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_stock_data(ticker="AAPL", days=1000, start_price=150.0, seed=42):
    """Generate synthetic but realistic stock price data."""
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")

    # Geometric Brownian Motion
    mu, sigma = 0.0003, 0.015
    returns = np.random.normal(mu, sigma, days)
    prices  = start_price * np.cumprod(1 + returns)

    volume = np.random.randint(1_000_000, 50_000_000, days)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   prices * np.random.uniform(0.99, 1.01, days),
        "High":   prices * np.random.uniform(1.00, 1.03, days),
        "Low":    prices * np.random.uniform(0.97, 1.00, days),
        "Close":  prices,
        "Volume": volume,
    })
    df.set_index("Date", inplace=True)
    df.to_csv("data/stock_data.csv")
    print(f"✅  Generated {days} days of synthetic {ticker} data → data/stock_data.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df):
    """Add technical indicators as model features."""
    df = df.copy()

    # Moving averages
    df["MA_7"]  = df["Close"].rolling(7).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # Bollinger Bands
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * df["BB_std"]
    df["BB_lower"] = df["MA_20"] - 2 * df["BB_std"]

    # RSI (14-day)
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12      = df["Close"].ewm(span=12).mean()
    ema26      = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    # Price momentum & volatility
    df["Momentum"]   = df["Close"].pct_change(5)
    df["Volatility"] = df["Close"].rolling(10).std()

    # Day-of-week (0=Mon … 4=Fri)
    df["DayOfWeek"] = df.index.dayofweek

    # Target: next-day close
    df["Target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_20", "MA_50",
    "BB_upper", "BB_lower", "RSI", "MACD",
    "Momentum", "Volatility", "DayOfWeek",
]


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


# ── 3a. Linear Regression ─────────────────────────────────────────────────────

def run_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, evaluate("Linear Regression", y_test, preds)


# ── 3b. Random Forest ─────────────────────────────────────────────────────────

def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, evaluate("Random Forest", y_test, preds)


# ── 3c. LSTM ──────────────────────────────────────────────────────────────────

def build_lstm_sequences(scaled_data, window=60):
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i - window:i])
        y.append(scaled_data[i, 0])          # 0th column = Close (after scaling)
    return np.array(X), np.array(y)


def run_lstm(df):
    if not LSTM_AVAILABLE:
        return None, None, None

    close = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    split   = int(len(scaled) * 0.8)
    train_d = scaled[:split]
    test_d  = scaled[split - 60:]

    X_tr, y_tr = build_lstm_sequences(train_d)
    X_te, y_te = build_lstm_sequences(test_d)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_tr.shape[1], 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0,
              validation_split=0.1)

    preds_scaled = model.predict(X_te, verbose=0)
    preds  = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    metrics = evaluate("LSTM Neural Network", actual, preds)
    return model, preds, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df, y_test, test_index, lr_preds, rf_preds, lstm_preds=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle("Stock Price Predictor — Results", fontsize=16, fontweight="bold")

    # Panel 1: full price history + MAs
    ax = axes[0]
    ax.plot(df.index, df["Close"],  label="Close Price",  linewidth=1.2, color="#1f77b4")
    ax.plot(df.index, df["MA_20"],  label="MA-20",        linewidth=1,   color="#ff7f0e", linestyle="--")
    ax.plot(df.index, df["MA_50"],  label="MA-50",        linewidth=1,   color="#2ca02c", linestyle="--")
    ax.fill_between(df.index, df["BB_lower"], df["BB_upper"], alpha=0.1, color="gray", label="Bollinger Bands")
    ax.set_title("Historical Price with Technical Indicators")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Panel 2: model predictions vs actual
    ax = axes[1]
    ax.plot(test_index, y_test,    label="Actual",           color="black",   linewidth=1.5)
    ax.plot(test_index, lr_preds,  label="Linear Regression",color="#d62728", linewidth=1, linestyle="--")
    ax.plot(test_index, rf_preds,  label="Random Forest",    color="#2ca02c", linewidth=1)
    if lstm_preds is not None:
        # LSTM test set may be shorter
        ax.plot(test_index[-len(lstm_preds):], lstm_preds,
                label="LSTM", color="#9467bd", linewidth=1, linestyle="-.")
    ax.set_title("Model Predictions vs Actual Prices (Test Set)")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Panel 3: RSI
    ax = axes[2]
    ax.plot(df.index, df["RSI"], color="#8c564b", linewidth=1)
    ax.axhline(70, color="red",   linestyle="--", alpha=0.6, label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", alpha=0.6, label="Oversold (30)")
    ax.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.2, color="red")
    ax.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.2, color="green")
    ax.set_title("RSI — Relative Strength Index")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    plt.savefig("models/prediction_results.png", dpi=150, bbox_inches="tight")
    print("\n✅  Chart saved → models/prediction_results.png")
    plt.show()


def plot_metrics(results):
    names = [r["model"] for r in results]
    rmse  = [r["RMSE"] for r in results]
    r2    = [r["R2"]   for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Model Comparison", fontweight="bold")

    colors = ["#1f77b4", "#2ca02c", "#9467bd"][:len(names)]
    ax1.bar(names, rmse, color=colors)
    ax1.set_title("RMSE (lower is better)")
    ax1.set_ylabel("RMSE")

    ax2.bar(names, r2, color=colors)
    ax2.set_title("R² Score (higher is better)")
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("models/model_comparison.png", dpi=150, bbox_inches="tight")
    print("✅  Comparison chart saved → models/model_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   STOCK PRICE PREDICTOR")
    print("=" * 50)

    # Data
    raw_df = generate_stock_data(ticker="AAPL", days=1000)
    df     = add_features(raw_df)

    # Train / test split (no shuffle — time-series!)
    X = df[FEATURE_COLS].values
    y = df["Target"].values
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_index      = df.index[split:]

    # Scale features
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Run models
    results = []
    _, lr_preds, lr_metrics = run_linear_regression(X_train, X_test, y_train, y_test)
    results.append(lr_metrics)

    _, rf_preds, rf_metrics = run_random_forest(X_train, X_test, y_train, y_test)
    results.append(rf_metrics)

    _, lstm_preds, lstm_metrics = run_lstm(df)
    if lstm_metrics:
        results.append(lstm_metrics)

    # Save metrics table
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("models/metrics.csv", index=False)
    print(f"\n✅  Metrics saved → models/metrics.csv")
    print(f"\n{'='*50}")
    print(metrics_df.to_string(index=False))

    # Plots
    plot_results(df, y_test, test_index, lr_preds, rf_preds, lstm_preds)
    plot_metrics(results)

    print("\n🎉  All done! Check the models/ folder for outputs.")


if __name__ == "__main__":
    main()
