# Implied Volatility Under Heavy-Tailed Returns  
*A Monte Carlo Exploration with Student–t Shocks*

This repository contains the simulation code, analysis scripts, figures, and paper associated with a project examining how heavy-tailed return innovations affect option prices and implied volatilities. The work compares the classical Gaussian Black–Scholes model with a Student–t modification using Monte Carlo methods.

For full context, methodology, and results, see the accompanying paper:

**Montgomery, O. (2025).  
*Implied Volatility Under Heavy-Tailed Returns: A Monte Carlo Exploration with Student–t Shocks*.**  
Available at: `paper/heavy_tail_paper.pdf`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/otto-montgomery/heavy-tail-options.git
cd heavy-tail-options

Create the environment:

conda env create -f environment.yml
conda activate heavy-tail-options

Or install dependencies manually:

pip install -r requirements.txt


⸻

Running the Simulations

All runnable scripts are in src/.

1. Compare distributions (GBM vs. Student–t)

python src/monte_carlo/compare_distributions.py

This script:
	•	Simulates terminal prices under GBM (normal) and Student–t GBM
	•	Prices European calls via Monte Carlo for both distributions
	•	Computes implied volatilities across a range of moneyness values
	•	Plots IV curves comparing normal distribution with multiple Student–t distributions (varying degrees of freedom)
	•	Produces the main comparison figure used in the paper

⸻

2. Simulate GBM (Geometric Brownian Motion)

python src/monte_carlo/simulate_gbm.py

This script:
	•	Simulates stock price paths using standard GBM with normal innovations
	•	Prices European calls via Monte Carlo
	•	Computes implied volatilities across moneyness values
	•	Plots the IV curve for the normal distribution case

⸻

3. Simulate Student–t GBM

python src/monte_carlo/simulate_student_t.py

This script:
	•	Simulates stock price paths using GBM with Student–t innovations
	•	Tests multiple degrees of freedom (ν) values
	•	Prices European calls via Monte Carlo
	•	Computes implied volatilities across moneyness values
	•	Plots IV curves for different Student–t distributions

⸻

4. Generate theoretical implied volatility curve

python src/analysis/theoretical_iv.py

This script:
	•	Generates theoretical Black–Scholes call prices using a constant volatility
	•	Computes implied volatilities from these prices
	•	Plots the theoretical IV curve (should be flat at the true volatility)
	•	Demonstrates the baseline case with no volatility smile

⸻

5. Reproduce the AAPL implied volatility smile

python src/analysis/aapl_iv.py

This script:
	•	Pulls AAPL option-chain data from Yahoo Finance
	•	Selects options with 40-50 days to expiry, aiming for 45 days from expiry.
	•	Computes midprices from bid/ask spreads
	•	Numerically inverts Black–Scholes to obtain implied volatilities
	•	Plots the real-market volatility smile against moneyness

⸻

Figures

Final, paper-ready figures:

paper/figures/

⸻

Reproducibility

Simulations use NumPy’s Generator with fixed seeds for repeatability.
Seeds can be changed directly inside the scripts.

