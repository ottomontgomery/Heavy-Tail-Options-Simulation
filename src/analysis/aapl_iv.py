# Plot implied volatility against moneyness for a given maturity

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

ticker = "AAPL"
S0 = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
r = 0.02

tkr = yf.Ticker(ticker)

def bs_calls(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol_call(price, S, K, T, r):
    if price <= max(S - K, 0):
        return np.nan
    f = lambda s: bs_calls(S, K, T, r, s) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return np.nan

# Find expiry date closest to target maturity (30-90 days)
target_days_min = 40
target_days_max = 50
target_days_optimal = 49  # Aim for ~60 days

best_expiry = None
best_days_diff = float('inf')

for expiry in tkr.options:
    days_to_exp = (np.datetime64(expiry) - np.datetime64('today')).astype('timedelta64[D]').astype(int)
    if target_days_min <= days_to_exp <= target_days_max:
        days_diff = abs(days_to_exp - target_days_optimal)
        if days_diff < best_days_diff:
            best_days_diff = days_diff
            best_expiry = expiry

if best_expiry is None:
    # Fallback: use first expiry if none in range
    best_expiry = tkr.options[0]
    print(f"Warning: No expiry found in {target_days_min}-{target_days_max} day range. Using {best_expiry}")

# Get option chain for selected expiry
chain = tkr.option_chain(best_expiry)
calls = chain.calls

calls['mids'] = (calls['bid'] + calls['ask']) / 2

days_to_exp = (np.datetime64(best_expiry) - np.datetime64('today')).astype('timedelta64[D]').astype(int)
T = max(days_to_exp, 1) / 365.0

calls['iv'] = calls.apply(
    lambda row: implied_vol_call(row['mids'], S0, row['strike'], T, r), 
    axis=1
)

# Calculate moneyness (K/S)
calls['moneyness'] = calls['strike'] / S0

# Filter out NaN values and sort by moneyness
valid_calls = calls.dropna(subset=['iv']).sort_values('moneyness')

# Filter data to the x-axis range for y-axis scaling
x_min, x_max = 0.9, 1.3
filtered_calls = valid_calls[(valid_calls['moneyness'] >= x_min) & (valid_calls['moneyness'] <= x_max)]

# Calculate y-axis limits with padding
if len(filtered_calls) > 0:
    y_min = filtered_calls['iv'].min()
    y_max = filtered_calls['iv'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # 10% padding
    y_lim_min = max(0, y_min - y_padding)
    y_lim_max = y_max + y_padding
else:
    y_lim_min = 0
    y_lim_max = 1

# Plot 2D graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(valid_calls['moneyness'], valid_calls['iv'], 'o-', linewidth=2, markersize=4, label=f'{days_to_exp} days to expiry')
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='ATM (Moneyness = 1)')
ax.set_xlabel('Moneyness (K/S)', fontsize=12)
ax.set_ylabel('Implied Volatility', fontsize=12)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0.18, 0.3)
ax.set_title(f'Implied Volatility vs Moneyness - {ticker}\nExpiry: ({days_to_exp} days)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

