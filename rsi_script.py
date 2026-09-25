import os
from flask import Flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

app = Flask(__name__)


@app.route("/")
def calculate_synthetic_vwap():
  stocks_data = {
      "HDFCBANK.NS": 0.25,
      "RELIANCE.NS": 0.20,
      "ICICIBANK.NS": 0.18,
      "INFY.NS": 0.10,
      "BHARTIARTL.NS": 0.10,
      "LT.NS": 0.08,
      "AXISBANK.NS": 0.05,
      "SBIN.NS": 0.04,
  }

  tickers = list(stocks_data.keys())
  weights = np.array(list(stocks_data.values()))

  data = yf.download(
      tickers, period="2d", interval="5m", group_by="ticker", progress=False
  )

  open_df = pd.DataFrame({t: data[t]["Open"] for t in tickers})
  high_df = pd.DataFrame({t: data[t]["High"] for t in tickers})
  low_df = pd.DataFrame({t: data[t]["Low"] for t in tickers})
  close_df = pd.DataFrame({t: data[t]["Close"] for t in tickers})
  volume_df = pd.DataFrame({t: data[t]["Volume"] for t in tickers})

  open_df.dropna(how="all", inplace=True)
  high_df.dropna(how="all", inplace=True)
  low_df.dropna(how="all", inplace=True)
  close_df.dropna(how="all", inplace=True)
  volume_df.fillna(0, inplace=True)

  s_open = open_df.dot(weights)
  s_high = high_df.dot(weights)
  s_low = low_df.dot(weights)
  s_close = close_df.dot(weights)
  s_vol = volume_df.sum(axis=1)

  s_tp = (s_high + s_low + s_close) / 3
  dates = s_tp.index.date

  df_calc = pd.DataFrame(
      {
          "Open": s_open,
          "High": s_high,
          "Low": s_low,
          "Close": s_close,
          "TP_Vol": s_tp * s_vol,
          "Volume": s_vol,
          "Date": dates,
      },
      index=s_tp.index,
  )

  df_calc["VWAP"] = (
      df_calc.groupby("Date")["TP_Vol"].cumsum()
      / df_calc.groupby("Date")["Volume"].cumsum()
  )

  latest_close = s_close.iloc[-1]
  latest_vwap = df_calc["VWAP"].iloc[-1]
  timestamp = s_close.index[-1].strftime("%Y-%m-%d %H:%M:%S")

  bias = (
      "Bullish (Above VWAP)"
      if latest_close > latest_vwap
      else "Bearish (Below VWAP)"
  )

  # Plotly కాండిల్ మరియు VWAP చార్ట్ తయారీ
  fig = go.Figure()

  # కాండిల్స్‌టిక్ చార్ట్ జోడించడం
  fig.add_trace(
      go.Candlestick(
          x=df_calc.index,
          open=df_calc["Open"],
          high=df_calc["High"],
          low=df_calc["Low"],
          close=df_calc["Close"],
          name="Synthetic Candles",
      )
  )

  # VWAP లైన్ జోడించడం
  fig.add_trace(
      go.Scatter(
          x=df_calc.index,
          y=df_calc["VWAP"],
          mode="lines",
          name="Synthetic VWAP",
          line=dict(color="#2962FF", width=2),
      )
  )

  fig.update_layout(
      title="Nifty Synthetic Candlestick & VWAP Chart",
      xaxis_title="Time",
      yaxis_title="Price",
      template="plotly_dark",
      xaxis_rangeslider_visible=False,
      height=500,
      dragmode="zoom",
  )

  chart_html = fig.to_html(
    full_html=False, config={"scrollZoom": True, "responsive": True}
)


  return f"""
    <html>
        <head>
            <title>Nifty Synthetic VWAP & Chart</title>
            <meta http-equiv="refresh" content="60">
        </head>
        <body style="font-family: Arial; padding: 20px; background-color: #121212; color: #ffffff;">
            <h2>Nifty 8-Stock Synthetic Candle & VWAP</h2>
            <p><b>Timestamp:</b> {timestamp}</p>
            <p><b>Synthetic Close:</b> {latest_close:.2f}</p>
            <p><b>Synthetic VWAP:</b> {latest_vwap:.2f}</p>
            <p><b>Market State:</b> <span style="color: {"#00E676" if "Bullish" in bias else "#FF5252"};">{bias}</span></p>
            
            <div style="margin-top: 20px;">
                {chart_html}
            </div>

            <p style="color: gray; font-size: 12px; margin-top: 20px;">(Page auto-refreshes every 60 seconds)</p>
        </body>
    </html>
    """


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8080))
  app.run(host="0.0.0.0", port=port)
