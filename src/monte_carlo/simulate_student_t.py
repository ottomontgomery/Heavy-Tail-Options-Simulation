import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

def bs_calls(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol_call(price, S, K, T, r):
    intrinsic_lb = max(S - K * np.exp(-r*T), 0.0)
    if price <= intrinsic_lb or price >= S:
        return np.nan
    f = lambda s: bs_calls(S, K, T, r, s) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return np.nan

# Define option parameters
S0 = 100
r = 0.02
sigma = 0.2
T = 45/365

# Monte Carlo Parameters
n_paths = 100000
dt = 1/252
n_steps = int(T/dt)

# Sweep over moneyness values
moneyness_grid = np.linspace(0.80, 1.30, 50)
K_grid = moneyness_grid * S0

# Store results for each nu
iv_results = {}

rng = np.random.default_rng(42)

# Loop over nu values (degrees of freedom)
nu_values = np.linspace(2.1, 3.9, 5)  # 5 values between 2.1 and 3.9

for nu in nu_values:
    print(f"Processing nu = {nu:.2f}...")
    
    # Create Student-t distributed random variables
    t_raw = rng.standard_t(df=nu, size=(n_paths, n_steps))
    t_scaled = t_raw / np.sqrt(nu / (nu - 2))  # unit variance
    eps = t_scaled
    
    # Build drift + diffusion
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * eps
    
    # Build log-price process
    log_S = np.log(S0) + np.cumsum(drift + diffusion, axis=1)
    
    # Extract terminal prices
    S_T = np.exp(log_S[:, -1])
    
    # Compute call prices for each strike
    call_prices_std = []
    for K in K_grid:
        payoffs = np.maximum(S_T - K, 0)
        C_mc = np.exp(-r * T) * np.mean(payoffs)
        call_prices_std.append(C_mc)
    
    call_prices_std = np.array(call_prices_std)
    
    # Compute implied volatilities
    iv_grid = []
    for K, price in zip(K_grid, call_prices_std):
        iv = implied_vol_call(price, S0, K, T, r)
        iv_grid.append(iv)
    
    iv_grid = np.array(iv_grid)
    iv_results[nu] = iv_grid

# Plot all IV curves on the same graph
fig, ax = plt.subplots(figsize=(10, 6))

# Use a colormap to distinguish different nu values
colors = plt.cm.viridis(np.linspace(0, 1, len(nu_values)))

for nu, color in zip(nu_values, colors):
    iv_grid = iv_results[nu]
    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(iv_grid)
    valid_moneyness = moneyness_grid[valid_mask]
    valid_iv = iv_grid[valid_mask]
    
    ax.plot(valid_moneyness, valid_iv, 'o-', linewidth=2, markersize=3, 
            color=color, label=f'ν = {nu:.2f}', alpha=0.8)

ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='ATM (Moneyness = 1)')
ax.axhline(y=sigma, color='r', linestyle='--', alpha=0.5, label=f'True Volatility (σ={sigma:.1%})')
ax.set_xlabel('Moneyness (K/S)', fontsize=12)
ax.set_ylabel('Implied Volatility', fontsize=12)
ax.set_xlim(0.80, 1.3)
ax.set_ylim(0.18, 0.3)
ax.set_title(f'Implied Volatility vs Moneyness\nStudent-t Distribution (σ={sigma:.1%}, {n_paths:,} paths)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
