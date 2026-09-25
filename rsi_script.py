import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Select 14 Top Nifty 50 Heavyweight Tickers (Yahoo Finance NSE format)
tickers = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
    'ITC.NS', 'LT.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'AXISBANK.NS',
    'HINDUNILVR.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS'
]

# Step 2: Define corresponding Free-Float Market Cap Weights 
# (These are approximate illustrative weights representing relative free-float dominance)
raw_weights = [
    0.14, 0.13, 0.09, 0.08, 0.08, 
    0.06, 0.06, 0.05, 0.05, 0.05, 
    0.05, 0.05, 0.05, 0.04
]

# Normalize weights so they strictly sum up to 1.0 (100%)
weights = np.array(raw_weights) / sum(raw_weights)

# Download historical Adjusted Close price data (e.g., past 1 year)
print("Downloading historical data for heavyweights...")
data = yf.download(tickers, period="1y", interval="1d")['Close']

# Drop any columns with missing data to keep calculations clean
data = data.dropna(axis=1, how='any')

# Step 3: Function to calculate J. Welles Wilder's RSI accurately
def calculate_wilder_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = np.zeros_like(close_series, dtype=float)
    avg_loss = np.zeros_like(close_series, dtype=float)
    
    # Initialize first average using SMA at the 'period' index
    if len(close_series) <= period:
        return pd.Series(np.nan, index=close_series.index)
        
    avg_gain[period] = gain.iloc[1:period+1].mean()
    avg_loss[period] = loss.iloc[1:period+1].mean()
    
    # Recursive Wilder's smoothing loop for subsequent candles
    for i in range(period + 1, len(close_series)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period
        
    # Prevent division by zero
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Handle strict edge cases
    rsi = np.where(avg_loss == 0, 100.0, rsi)
    rsi = np.where(avg_gain == 0, 0.0, rsi)
    
    # Set warmup period values to NaN
    rsi[:period] = np.nan
    
    return pd.Series(rsi, index=close_series.index)

# Calculate individual RSI for each stock
rsi_df = pd.DataFrame(index=data.index)
for ticker in data.columns:
    rsi_df[ticker] = calculate_wilder_rsi(data[ticker], period=14)

# Align weights with the downloaded columns (in case any ticker failed to download)
active_tickers = list(rsi_df.columns)
active_weights = [weights[tickers.index(t)] for t in active_tickers]
active_weights = np.array(active_weights) / sum(active_weights) # Re-normalize

# Compute the Final Free-Float Weighted Composite RSI
composite_rsi = pd.Series(0.0, index=rsi_df.index)
for i, ticker in enumerate(active_tickers):
    composite_rsi += rsi_df[ticker] * active_weights[i]

# Drop initial NaN rows created during the RSI warmup phase
composite_rsi = composite_rsi.dropna()

print("\n--- Custom Composite RSI Calculated Successfully ---")
print(composite_rsi.tail(5))

# Plotting the Custom Indicator
plt.figure(figsize=(12, 6))
plt.plot(composite_rsi.index, composite_rsi, label='Nifty Heavyweight Composite RSI', color='purple', linewidth=1.5)
plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
plt.title('Custom Free-Float Weighted Heavyweight RSI (Nifty 50 Leaders)')
plt.xlabel('Date')
plt.ylabel('RSI Value')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()
