import pandas as pd
import numpy as np
import yfinance as yf

# 1. Define the Top 8 Nifty 50 Stocks and their free-float weights
stocks_data = {
    'HDFCBANK.NS': 0.25,  
    'RELIANCE.NS': 0.20,  
    'ICICIBANK.NS': 0.18, 
    'INFY.NS': 0.10,      
    'BHARTIARTL.NS': 0.10,
    'LT.NS': 0.08,        
    'AXISBANK.NS': 0.05,  
    'SBIN.NS': 0.04       
}

tickers = list(stocks_data.keys())
weights = np.array(list(stocks_data.values()))

print("Fetching intraday data (5-minute interval)...")
# Note: yfinance allows up to 60 days for 5m intraday data
data = yf.download(tickers, period="2d", interval="5m", group_by="ticker", progress=False)

# 2. Extract DataFrames for Open, High, Low, Close, and Volume
open_df = pd.DataFrame({t: data[t]['Open'] for t in tickers})
high_df = pd.DataFrame({t: data[t]['High'] for t in tickers})
low_df = pd.DataFrame({t: data[t]['Low'] for t in tickers})
close_df = pd.DataFrame({t: data[t]['Close'] for t in tickers})
volume_df = pd.DataFrame({t: data[t]['Volume'] for t in tickers})

# Drop rows where all values are NaN
open_df.dropna(how='all', inplace=True)
high_df.dropna(how='all', inplace=True)
low_df.dropna(how='all', inplace=True)
close_df.dropna(how='all', inplace=True)
volume_df.fillna(0, inplace=True)

# 3. Compute Weighted Synthetic OHLC Candles
synthetic_open = open_df.dot(weights)
synthetic_high = high_df.dot(weights)
synthetic_low = low_df.dot(weights)
synthetic_close = close_df.dot(weights)

# Aggregate total volume for the basket across the 8 stocks
synthetic_volume = volume_df.sum(axis=1)

# 4. Calculate Intraday VWAP
# Typical Price of the synthetic basket
synthetic_tp = (synthetic_high + synthetic_low + synthetic_close) / 3

# Extract date to reset cumulative VWAP daily
dates = synthetic_tp.index.date

# Calculate Cumulative (Typical Price * Volume) and Cumulative Volume per day
df_calc = pd.DataFrame({
    'TP_Vol': synthetic_tp * synthetic_volume,
    'Volume': synthetic_volume,
    'Date': dates
}, index=synthetic_tp.index)

# Group by date to ensure VWAP resets each trading day
cum_tp_vol = df_calc.groupby('Date')['TP_Vol'].cumsum()
cum_vol = df_calc.groupby('Date')['Volume'].cumsum()

# Synthetic VWAP calculation
synthetic_vwap = cum_tp_vol / cum_vol

# Combine into a final synthetic intraday dataframe
synthetic_intraday_df = pd.DataFrame({
    'Open': synthetic_open,
    'High': synthetic_high,
    'Low': synthetic_low,
    'Close': synthetic_close,
    'Volume': synthetic_volume,
    'VWAP': synthetic_vwap
})

print("\n--- Latest 5-Minute Synthetic Candles & VWAP ---")
print(synthetic_intraday_df.tail(10))

# 5. Summary of the most recent candle and VWAP state
latest = synthetic_intraday_df.iloc[-1]
print("\n--- Current Status ---")
print(f"Timestamp : {synthetic_intraday_df.index[-1]}")
print(f"Close     : {latest['Close']:.2f}")
print(f"VWAP      : {latest['VWAP']:.2f}")

if latest['Close'] > latest['VWAP']:
    print("Market State: Price is ABOVE VWAP (Bullish intraday bias)")
else:
    print("Market State: Price is BELOW VWAP (Bearish intraday bias)")
