import yfinance as yf
import os

os.makedirs(r'Data\Nifty', exist_ok=True)

nifty = yf.download('^NSEI', start='2015-01-01', end='2026-02-28', auto_adjust=False)
output_path = r'Data\Nifty\NIFTY50_data.csv'
nifty.to_csv(output_path)

print(f'Downloaded {len(nifty)} rows of Nifty 50 data')
print(f'Date range: {nifty.index.min().date()} to {nifty.index.max().date()}')
print(f'Saved to: {output_path}')
