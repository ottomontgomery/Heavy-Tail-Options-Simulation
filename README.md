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

1. Generate Gaussian and Student–t implied volatility curves

python src/analysis/generate_iv_curves.py

This script:
	•	Simulates terminal prices under GBM and Student–t GBM
	•	Prices European calls via Monte Carlo
	•	Computes implied volatilities
	•	Produces the figures used in the paper

Output figures appear in figures/.

⸻

2. Compare distributions (GBM vs. Student–t)

python src/analysis/compare_distributions.py

Generates histograms/KDE plots showing tail thickness for each distribution.

⸻

3. Reproduce the AAPL implied volatility smile

python src/analysis/real_market_iv.py

This script:
	•	Pulls AAPL option-chain data
	•	Computes midprices
	•	Numerically inverts Black–Scholes to obtain IVs
	•	Outputs the real-market volatility smile

⸻

Notebooks

Exploratory notebooks are stored in:

notebooks/

These document intermediate steps and experiments but are not required for reproducing the core results.

⸻

Figures

Final, paper-ready figures:

paper/figures/

Supplemental and testing plots:

figures/


⸻

Reproducibility

Simulations use NumPy’s Generator with fixed seeds for repeatability.
Seeds can be changed directly inside the scripts.

⸻

License

(Optional: insert MIT License or CC-BY here.)

