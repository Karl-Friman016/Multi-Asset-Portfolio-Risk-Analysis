"""
Multi-Asset Portfolio Risk Analysis
Author: Karl Friman
Date: 2026-03-15

Purpose:
This project measures downside risk in a multi-asset portfolio using Value at Risk (VaR)
and Expected Shortfall (CVaR). The analysis compares historical and parametric risk measures
based on daily portfolio returns.

Financial idea:
Portfolio risk is not only about average volatility, but also about the size of potential losses
in the tail of the return distribution. VaR estimates a loss threshold at a given confidence level,
while CVaR measures the average loss beyond that threshold.

Steps in this script:
1. Import libraries
2. Set project parameters
3. Download historical data and calculate returns
4. Construct portfolio returns
5. Historical VaR and CVaR
6. Estimated Var and CVaR
7. Risk summary table
8. Visualize the return distribution

"""
#%%
# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import t
#%%
# --------------------------------------------------
# 2. Set project parameters
# --------------------------------------------------
tickers = ['SPY', 'QQQ', 'XLF', 'GLD', 'TLT']
start_date = '2015-01-01'
end_date = '2025-12-31'

# Equal-weight portfolio for the baseline analysis
portfolio_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # change based on nr of assets


# %%
#---------------------------------------------------
# 3. Download the data and calculate daily return
#---------------------------------------------------
financial_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
prices= financial_data['Adj Close']
daily_returns = prices.pct_change().dropna()

# %%
# --------------------------------------------------
# 4. Construct portfolio returns
# --------------------------------------------------

# Calculate daily portfolio returns as the weighted sum of asset returns
portfolio_returns = daily_returns.dot(portfolio_weights)

# Quick check
print(portfolio_returns.head())
print(f"Portfolio weights sum to: {portfolio_weights.sum():.2f}")

# %%
# --------------------------------------------------
# 5. Historical VaR and CVaR
# --------------------------------------------------

# 95% historical VaR and CVaR
historical_var_95 = -np.percentile(portfolio_returns, 5)
historical_cvar_95 = -portfolio_returns[portfolio_returns <= -historical_var_95].mean()

# 99% historical VaR and CVaR
historical_var_99 = -np.percentile(portfolio_returns, 1)
historical_cvar_99 = -portfolio_returns[portfolio_returns <= -historical_var_99].mean()

# Display results
print("Historical Risk Measures:")
print(f"95% Historical VaR: {historical_var_95:.2%}")
print(f"95% Historical CVaR: {historical_cvar_95:.2%}")
print(f"99% Historical VaR: {historical_var_99:.2%}")
print(f"99% Historical CVaR: {historical_cvar_99:.2%}")

# %%
# --------------------------------------------------
# 6.  t-VaR and CVaR
# --------------------------------------------------

# Fit t-distribution to portfolio returns
t_df, t_loc, t_scale = t.fit(portfolio_returns)
df= t_df

#Tail probabilities
alpha_95 = 0.05
alpha_99 = 0.01

# t-VaR
t_var_95 = -t.ppf(alpha_95, df=t_df, loc=t_loc, scale=t_scale)
t_var_99 = -t.ppf(alpha_99, df=t_df, loc=t_loc, scale=t_scale)

# Left-tail quantiles from fitted t-distribution
q_95 = t.ppf(alpha_95, df=t_df, loc=t_loc, scale=t_scale)
q_99 = t.ppf(alpha_99, df=t_df, loc=t_loc, scale=t_scale)

# t-distribution CVaR using fitted parameters
t_cvar_95 = -(
    t_loc
    - t_scale
    * ((t_df + ((q_95 - t_loc) / t_scale) ** 2) / (t_df - 1))
    * (t.pdf((q_95 - t_loc) / t_scale, df=t_df) / alpha_95)
)

t_cvar_99 = -(
    t_loc
    - t_scale
    * ((t_df + ((q_99 - t_loc) / t_scale) ** 2) / (t_df - 1))
    * (t.pdf((q_99 - t_loc) / t_scale, df=t_df) / alpha_99)
)

# Display results
print("\nT-Distribution Risk Measures:")
print(f"95% t-VaR: {t_var_95:.4%}")
print(f"95% t-CVaR: {t_cvar_95:.4%}")
print(f"99% t-VaR: {t_var_99:.4%}")
print(f"99% t-CVaR: {t_cvar_99:.4%}")

# %%
# --------------------------------------------------
# 7. Risk summary table
# --------------------------------------------------

risk_summary = pd.DataFrame({
    'Confidence Level': ['95%', '99%'],
    'Historical VaR': [historical_var_95, historical_var_99],
    'Historical CVaR': [historical_cvar_95, historical_cvar_99],
    't-VaR': [t_var_95, t_var_99],
    't-CVaR': [t_cvar_95, t_cvar_99]
})

risk_summary_display = risk_summary.copy()
for col in ['Historical VaR', 'Historical CVaR', 't-VaR', 't-CVaR']:
    risk_summary_display[col] = risk_summary_display[col].map("{:.2%}".format)

print("Risk Summary Table:")
print(risk_summary_display)

#%% 
# --------------------------------------------------
# 8. Plot return distribution and VaR thresholds
# --------------------------------------------------

plt.figure(figsize=(10, 6))

# Histogram of daily portfolio returns
plt.hist(portfolio_returns, bins=50, density=True, alpha=0.9, label ='Portfolio Returns')
plt.xlim(-0.04, 0.02)
x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)

# Compute fitted t-distribution density
t_pdf = t.pdf(x, df=t_df, loc=t_loc, scale=t_scale)

# Plot fitted t-distribution
plt.plot(x, t_pdf, linewidth=2, label='Fitted t-Distribution')

# Add VaR lines
plt.axvline(-historical_var_95, linestyle='--', linewidth=2, color='blue', label='Historical VaR 95%')
plt.axvline(-historical_var_99, linestyle='--', linewidth=2, color='navy', label='Historical VaR 99%')
plt.axvline(-t_var_95, linestyle=':', linewidth=2, color='red', label='t-VaR 95%')
plt.axvline(-t_var_99, linestyle=':', linewidth=2, color='darkred', label='t-VaR 99%')

plt.xlabel('Daily Portfolio Return')
plt.ylabel('Density')
plt.title('Portfolio Return Distribution with VaR Thresholds')
plt.legend()
plt.savefig("portfolio_risk_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
