import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import time
from datetime import datetime
import sys
import os

# --- COLOR OUTPUT ---
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# DATA FETCH FUNCTION

def fetch_forex_data(symbol='EURUSD=X', interval='1m', period='7d'):
    df = yf.download(tickers=symbol, interval=interval, period=period, auto_adjust=True, progress=False)
@@ -19,113 +31,108 @@
    df.dropna(inplace=True)
    return df

from ta.momentum import StochasticOscillator
# FEATURE ENGINEERING

def engineer_features(df: pd.DataFrame):
    df = df.copy()

    
    close = df['close'].squeeze()
    high = df['high'].squeeze()
    low = df['low'].squeeze()

    df['return'] = close.pct_change()

   
    df['rsi_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    df['return'] = close.pct_change()
    df['rsi_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=2, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    
    df['ema_10'] = ta.trend.EMAIndicator(close=close, window=10).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df['cci'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()

    
    bb = ta.volatility.BollingerBands(close=close, window=2, window_dev=2)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()

    df['atr'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

   
    macd = ta.trend.MACD(close=close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    
    df.dropna(inplace=True)

    return df


# TARGET CREATION

def create_target(df: pd.DataFrame):
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)  
    df.dropna(inplace=True)
    return df

def train_random_forest(df, timeframe='1m'):
    df = engineer_features(df)  
# TRAIN MODEL

def train_random_forest(df):
    df = engineer_features(df)
    df = create_target(df)
    
    X = df.drop(columns=['target'])
    X = df.drop(columns=['target']).select_dtypes(include='number')
    y = df['target']
    
    X = X.select_dtypes(include='number')  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"\n Random Forest - {timeframe} timeframe")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, X_test, y_test
    return model

# INITIAL TRAINING

print("Fetching 5m data...")
print("Fetching initial 5m data...")
df_5m = fetch_forex_data(interval='5m', period='30d')
print("5m data fetched:", df_5m.shape)
model_5m = train_random_forest(df_5m)

print("Fetching 15m data...")
print("Fetching initial 15m data...")
df_15m = fetch_forex_data(interval='15m', period='60d')
print("15m data fetched:", df_15m.shape)


model_5m, X_test_5m, y_test_5m = train_random_forest(df_5m, '5m')
model_15m, X_test_15m, y_test_15m = train_random_forest(df_15m, '15m')

model_15m = train_random_forest(df_15m)

common_index = X_test_5m.index.intersection(X_test_15m.index)
print("\nStarting live prediction loop... Press Ctrl+C to stop.\n")

X_test_5m_aligned = X_test_5m.loc[common_index]
X_test_15m_aligned = X_test_15m.loc[common_index]
# Create CSV if not exists
if not os.path.exists("forex_signals.csv"):
    pd.DataFrame(columns=["time", "prediction", "confidence"]).to_csv("forex_signals.csv", index=False)

y_test_5m_aligned = y_test_5m.loc[common_index]
try:
    while True:
        # Fetch latest candles
        latest_5m = fetch_forex_data(interval='5m', period='3d')
        latest_15m = fetch_forex_data(interval='15m', period='7d')

        # Feature engineering (no target)
        X_5m = engineer_features(latest_5m).select_dtypes(include='number').iloc[-1:]
        X_15m = engineer_features(latest_15m).select_dtypes(include='number').iloc[-1:]

probs_5m = model_5m.predict_proba(X_test_5m_aligned)
probs_15m = model_15m.predict_proba(X_test_15m_aligned)
        # Predictions
        probs_5m = model_5m.predict_proba(X_5m)
        probs_15m = model_15m.predict_proba(X_15m)
        ensemble_probs = 0.6 * probs_5m + 0.4 * probs_15m
        prediction = np.argmax(ensemble_probs, axis=1)[0]
        confidence = max(ensemble_probs[0]) * 100

        
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if prediction == 1:
            print(f"{GREEN}[{now}] Abhiav, next candle will be UP (Confidence: {confidence:.2f}%) {RESET}")
        else:
            print(f"{RED}[{now}] Abhiav, next candle will be DOWN  (Confidence: {confidence:.2f}%) {RESET}")

ensemble_probs = 0.6 * probs_5m + 0.4 * probs_15m
ensemble_preds = np.argmax(ensemble_probs, axis=1)
        log_df = pd.DataFrame([[now, "UP" if prediction == 1 else "DOWN", confidence]],
                              columns=["time", "prediction", "confidence"])
        log_df.to_csv("forex_signals.csv", mode='a', header=False, index=False)

print("\nEnsemble Model Performance:")
print(f"Accuracy: {accuracy_score(y_test_5m_aligned, ensemble_preds):.4f}")
print(classification_report(y_test_5m_aligned, ensemble_preds))
        time.sleep(300)  

except KeyboardInterrupt:
    print("\nStopped live prediction loop.")
    sys.exit()

sf=yf.download(tickers='EURUSD=X', interval='1m', period='7d')
print(sf.tail())
print(sf.head())         # View first few rows
print(sf.info())         # Data types, nulls
print(sf.describe())     # Summary stats: mean, std, min, max
