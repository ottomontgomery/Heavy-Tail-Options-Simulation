# Plot simulated implied volatility against moneyness for a given maturity

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

S0 = 100
r = 0.02
true_volatility = 0.2

# Choose a single maturity (similar to fig1.py's approach)
T = 45 / 365.0  # ~40 days to expiry (in years)

# Create strikes and convert to moneyness
strikes = np.linspace(70, 130, 50)
moneyness = strikes / S0

def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol_call(price, S, K, T, r):
    # European call price lower bound
    intrinsic_lb = max(S - K * np.exp(-r*T), 0.0)

    # price must be within arbitrage bounds
    if price < intrinsic_lb or price > S:
        return np.nan

    f = lambda sigma: bs_call_price(S, K, T, r, sigma) - price

    try:
        return brentq(f, 1e-6, 5.0)
    except ValueError:
        return np.nan

# Calculate implied volatilities
ivs = []
for K in strikes:
    # Generate a call price using the true (constant) volatility
    price = bs_call_price(S0, K, T, r, true_volatility)
    # Back out implied vol from that price
    iv = implied_vol_call(price, S0, K, T, r)
    ivs.append(iv)

ivs = np.array(ivs)

mask = ~np.isnan(ivs)
filtered_moneyness = moneyness[mask]
filtered_ivs = ivs[mask]

# Filter data to the x-axis range for y-axis scaling
x_min, x_max = 0.9, 1.3
mask = (moneyness >= x_min) & (moneyness <= x_max)
filtered_moneyness = moneyness[mask]
filtered_ivs = ivs[mask]

# Calculate y-axis limits with padding
if len(filtered_ivs) > 0:
    y_min = filtered_ivs.min()
    y_max = filtered_ivs.max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # 10% padding
    y_lim_min = max(0, y_min - y_padding)
    y_lim_max = y_max + y_padding
else:
    y_lim_min = 0
    y_lim_max = 1

# Plot 2D graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(moneyness, ivs, 'o-', linewidth=2, markersize=4, label=f'{int(T*365)} days to expiry')
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='ATM (Moneyness = 1)')
ax.set_xlabel('Moneyness (K/S)', fontsize=12)
ax.set_ylabel('Implied Volatility', fontsize=12)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0.18, 0.3)
ax.set_title(f'Simulated Implied Volatility vs Moneyness\n(Flat σ = {true_volatility:.1%})', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
