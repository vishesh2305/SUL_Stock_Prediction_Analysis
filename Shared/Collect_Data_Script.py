import yfinance as yf
import os

# tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
# tickers = ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"]
# tickers = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"]
# tickers = ["INFY.NS", "TCS.NS", "WIPRO.NS"]

tickers = ["NESTLEIND.NS"]

target_Dir = "Data/nestle"


if not os.path.exists(target_Dir):
    os.makedirs(target_Dir)


for ticker in tickers:
    data = yf.download(ticker, start="2015-01-01", end="2026-03-01")
    file_path = os.path.join(target_Dir, f"{ticker}_data.csv")
    data.to_csv(file_path)
    print(f"Saved {ticker} data to {file_path}")