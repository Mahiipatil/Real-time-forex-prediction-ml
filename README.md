**Real-Time Forex Prediction ML**

A machine learning project to predict forex (foreign exchange) market trends in real-time using historical data and technical indicators.

Features

Predicts currency trends using ML models based on historical forex data.

Utilizes technical indicators such as EMA, RSI, and ATR for better prediction accuracy.

Modular Python code for data preprocessing, feature engineering, and prediction.

Ready for local testing and further extension with real-time data feeds.

**Dataset**

Source: Forex Trading Dataset with EMA, RSI & ATR (2025) – Kaggle

Currency Pairs: EUR/USD, GBP/USD, USD/JPY

Time Period: 2025

Format: CSV

Features: Market prices, EMA, RSI, ATR
**
Installation**


Install dependencies:

pip install -r requirements.txt


Run the main script:

python forex_main.py

Usage

Load the CSV dataset from data/ folder.

The script performs preprocessing and feature engineering.

The ML model predicts forex trends based on input features.

Outputs predictions to console or a results file.

Libraries/Dependencies

Python 3.x

pandas, numpy

scikit-learn

matplotlib / seaborn (for visualization, optional)
