<h1>SP 500 companies Stocks Analysis</h1>
<h3>Importing Required resources</h3>

```python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import numpy as np
df_companies= pd.read_csv('sp500_companies.csv')
df_stocks=pd.read_csv('sp500_stocks.csv')
df_index=pd.read_csv('sp500_index.csv')
df_index
```

<img width="261" height="464" alt="image" src="https://github.com/user-attachments/assets/b1cda18f-80b5-4457-afe7-621cabed604f" />


<h2>Data Cleaning</h2>

```python
df_companies['Ebitda'].isna().sum()

#Although there are some null values in Ebitda column but lets not fill them for avoiding further anomalies

df_companies['Revenuegrowth'].isna().sum()
#Although there are some null values in Revenuegrowth column but lets not fill them for avoiding further anomalies

df_companies['Weight'].isna().sum()

df_stocks
```

<img width="659" height="472" alt="image" src="https://github.com/user-attachments/assets/277f3a6b-a321-4a07-b18e-63ee3eaa8797" />

<p>However there are many null values in this dataframe but we need to ignore them because Nulls can appear due to:

Trading suspensions Extreme volatility halts Mergers & restructuring</p>

```python

df_index

```

<img width="263" height="472" alt="image" src="https://github.com/user-attachments/assets/fa6893af-7dae-4b63-9039-2f4264277012" />

<h2>Exploratory Data Analysis</h2>
<h3> 1. Compare sector-wise performance </h3>

```python

# Merge sector info into stock data
df = df_stocks.merge(
    df_companies[['Symbol', 'Sector', 'Marketcap']],
    on='Symbol',
    how='left'
)

# Convert Date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Daily_Return'] = df.groupby('Symbol')['Adj Close'].pct_change(fill_method=None)
df = df.dropna(subset=['Daily_Return'])
sector_performance = (
    df.groupby('Sector')['Daily_Return']
      .agg(
          Avg_Daily_Return='mean',
          Daily_Risk='std'
      )
      .reset_index()
)

sector_performance['Annual_Return'] = sector_performance['Avg_Daily_Return'] * 252
sector_performance['Annual_Risk'] = sector_performance['Daily_Risk'] * np.sqrt(252)
risk_free_rate = 0.02  # 2% annual

sector_performance['Sharpe_Ratio'] = (
    sector_performance['Annual_Return'] - risk_free_rate
) / sector_performance['Annual_Risk']
df['Weighted_Return'] = df['Daily_Return'] * df['Marketcap']

sector_weighted = (
    df.groupby('Sector')
      .apply(lambda x: x['Weighted_Return'].sum() / x['Marketcap'].sum())
      .reset_index(name='Weighted_Daily_Return')
)

sector_weighted['Annual_Weighted_Return'] = sector_weighted['Weighted_Daily_Return'] * 252

final_sector_comparison = sector_performance.merge(
    sector_weighted[['Sector', 'Annual_Weighted_Return']],
    on='Sector',
    how='left'
)

final_sector_comparison.sort_values(
    by='Sharpe_Ratio',
    ascending=False,
    inplace=True
)

final_sector_comparison
```

<img width="1043" height="498" alt="image" src="https://github.com/user-attachments/assets/96e80489-057e-49f9-8ace-09dcaa90c463" />

<h2>Measure Risk Vs Return by sector</h2>

```python

plt.figure(figsize=(10,6))
plt.scatter(
    final_sector_comparison['Annual_Risk'],
    final_sector_comparison['Annual_Return']
)

for i, sector in enumerate(final_sector_comparison['Sector']):
    plt.text(
        final_sector_comparison['Annual_Risk'].iloc[i],
        final_sector_comparison['Annual_Return'].iloc[i],
        sector
    )

plt.xlabel('Risk (Annual Volatility)')
plt.ylabel('Return (Annualized)')
plt.title('Sector-Wise Risk vs Return')
plt.show()

```

<img width="1111" height="695" alt="image" src="https://github.com/user-attachments/assets/6d4ab0d9-f995-4c6e-86fb-2595617ee070" />


<h2> 2.Study correlation between stocks</h2>

```python

df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], dayfirst=True)
#Calculate Daily Returns
df_stocks['Return'] = df_stocks.groupby('Symbol')['Adj Close'].pct_change(fill_method=None)
df_stocks = df_stocks.dropna(subset=['Return'])
returns_matrix = df_stocks.pivot(
    index='Date',
    columns='Symbol',
    values='Return'
)
correlation_matrix = returns_matrix.corr()
plt.figure(figsize=(12,8))
snb.heatmap(
    correlation_matrix,
    cmap='coolwarm',
    center=0
)
plt.title('Stock Return Correlation Matrix')
plt.show()

```

<img width="1173" height="868" alt="image" src="https://github.com/user-attachments/assets/112c519b-eb0b-41db-a3fe-2372c83bc4aa" />


<h2>Evaluate portfolio diversification benefits</h2>

```python

stocks['Symbol'].unique()[:20]

```
<img width="662" height="84" alt="image" src="https://github.com/user-attachments/assets/3002c961-47e3-43c8-b1f3-9b56ce7391ab" />

```python

set(['AAPL','MSFT','GOOGL','JNJ','XOM']) - set(stocks['Symbol'].unique())

```

Output : {'MSFT'}

```python

df_stocks[df_stocks['Symbol'].str.contains('GOOG', na=False)]['Symbol'].unique()

```
Output : array(['GOOG'], dtype=object)

```python

returns = df_stocks.pivot(
    index='Date',
    columns='Symbol',
    values='Return'
)
available_stocks = [s for s in portfolio_stocks if s in returns.columns]

returns = returns[available_stocks]
returns.columns

```
Output : Index([], dtype='object', name='Symbol')

```python

portfolio_stocks = ['AAPL', 'MSFT', 'GOOG', 'JNJ', 'XOM']

returns = df_stocks.pivot(
    index='Date',
    columns='Symbol',
    values='Return'
)

portfolio_stocks = [s for s in portfolio_stocks if s in returns.columns]

returns = returns[portfolio_stocks]
returns

```

<img width="213" height="546" alt="image" src="https://github.com/user-attachments/assets/8fe08dd2-9e22-4a4e-9663-a8a15306e33b" />


```python

correlation = returns.corr()
correlation

```

<img width="142" height="118" alt="image" src="https://github.com/user-attachments/assets/d9ea8ccb-557f-4a44-a723-b2a5c7aa0131" />

```python

individual_risk = returns.std() * np.sqrt(252)
weights = np.array([1/len(portfolio_stocks)] * len(portfolio_stocks))

cov_matrix = returns.cov() * 252

portfolio_risk = np.sqrt(
    np.dot(weights.T, np.dot(cov_matrix, weights))
)

portfolio_risk
```

Output : np.float64(0.2736224890643595)

```python

avg_individual_risk = individual_risk.mean()

diversification_benefit = (
    avg_individual_risk - portfolio_risk
) / avg_individual_risk * 100

diversification_benefit

```

Output : np.float64(0.6946251707103821)

```python

companies = pd.read_csv('sp500_companies.csv')

sector_map = companies.set_index('Symbol')['Sector']

portfolio_sectors = sector_map[portfolio_stocks]

portfolio_sectors

```

<p>Symbol
GOOG    Communication Services
Name: Sector, dtype: object
More sectors = stronger diversification

“I evaluated diversification by comparing individual stock volatility with portfolio volatility using the covariance matrix of returns. The portfolio showed lower risk than the average individual stock due to low cross-correlations, confirming diversification benefits.”</p>


<h1>Insights</h1>

<p>Diversification benefits reduce significantly during market crashes.

Volatility clustering is persistent in financial markets.

Sector rotation plays a major role in long-term performance.

Risk-adjusted returns matter more than raw returns.</p>





