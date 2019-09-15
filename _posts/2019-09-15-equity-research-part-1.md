---
title: 'Equity Research with Python, Part 1'
date: 2019-09-15
permalink: /posts/2019/09/equity-research-part-1
tags:
  - Finance
  - Python
  - Artificial Intelligence
---

We will analyze financial statements with the help of financial ratios, these ratios are loosely classified in to four categories:

* Profitability Ratios
* Leverage Ratios
* Valuation Ratios
* Operating Ratios

In this post, we will calculate only **Profitability Ratios**, these set of ratios helps in understanding how good a company is in generating profits and also conveys information about the competitive spirit of the management. In this post, we are going to calculate the following ratios:

1. EBITDA Margin
2. Net Profit Margin
3. Return on Equity (ROE)
4. Return on Asset (ROA)
5. Return on Capital Employed (ROCE)

Let's have a look at the balance sheet, income statement and cash flow statements of "Johnson & Johnson".

**Balance Sheet:**
<iframe width="100%" height="500" src="/images/plots/JNJ_bs.html">Balance Sheet</iframe>

**Income Statement:**
<iframe width="100%" height="500" src="/images/plots/JNJ_inc.html">Income Statement</iframe> 

**Cash Flow Statement:**
<iframe width="100%" height="500" src="/images/plots/JNJ_cs.html">Cash Flow Statement</iframe>

First, we will import all the necessary libraries and create few utility functions.

```python

from bs4 import BeautifulSoup
import pandas as pd
import requests

BASE_URL = "https://finance.yahoo.com/quote/"

#Ticker symbol
symbol = "JNJ"

#Year
t='12/30/2018'


def get_cash_flow(SYMBOL):
    
    URL = BASE_URL + SYMBOL + "/cash-flow"
    return parse_html(URL)


def get_balance_sheet(SYMBOL):

    URL = BASE_URL  + SYMBOL + "/balance-sheet/"  
    return parse_html(URL)


def get_income_statement(SYMBOL):

    URL = BASE_URL + SYMBOL + "/financials"
    return parse_html(URL)


def parse_html (URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find_all('table', {'class':'Lh(1.7) W(100%) M(0)'})
    df = pd.read_html(str(table), header =0, index_col = 0, flavor = 'bs4')[0]   
    return df


def get_val(dataframe,row,col):
    cell_value = int(dataframe.loc[[row], [col]].values)
    return cell_value

def drop_non_numeric(dataframe):
    dataframe = dataframe.convert_objects(convert_numeric=True).dropna()
    return dataframe

def replace_non_numeric_with_nan(dataframe):
    dataframe = dataframe.convert_objects(convert_numeric=True)
    return dataframe

def replace_nan_with_zero(dataframe):
    dataframe = dataframe.fillna(0)
    return dataframe

df_inc = get_income_statement(symbol)
df_bal = get_balance_sheet(symbol)
df_cas = get_cash_flow(symbol)
df_inc_no_str = replace_non_numeric_with_nan(df_inc)
df_bal_no_str = replace_non_numeric_with_nan(df_bal)
df_cas_no_str = replace_non_numeric_with_nan(df_cas)
df_cas_no_str_no_nan = replace_nan_with_zero(df_cas_no_str)
    
```

**1. EBITDA Margin**: The Earnings before Interest Tax Depreciation & Amortization Margin tells us how efficient and profitable a company is at an operating level. It always makes sense to compare the EBITDA margin of the company with its competitors.

EBITDA  =  EBIT + Depreciation & Amortization
EBIDTA Margin = EBITDA / Total Revenue

```python

EBIT = get_val(df_inc,'Earnings Before Interest and Taxes', t)
depr_amor = get_val(df_cas,'Depreciation', t)
EBITDA = EBIT + depr_amor

#We need total revenue to calculate the EBITDA margin.
total_revenue = get_val(df_inc,'Total Revenue', t)
EBITDA_margin = EBITDA /total_revenue
print("EBITDA is ",EBITDA)
print("EBITDA Margin is", round((EBITDA_margin*100),2),"%")

Output:
EBITDA is  28336000
EBITDA Margin is 34.73 %
```

**2. Net Profit Margin** is the percentage of revenue remaining after all the expenses are paid out. This ratio is  calculated as:

Net Profit Margin= Net Income  /  Total Revenues

```python
inc_df_no_nan = drop_non_numeric(df_inc)
net_income = get_val(inc_df_no_nan,'Net Income', t)
net_profit_margin = net_income/total_revenue
print("Net Profit Margin is ", round((net_profit_margin*100),2), "%")

Output:
Net Profit Margin is  18.75 %
```

**3. Return on Equity (RoE):** The Return on Equity (RoE) is a very important ratio, as it helps us in analysing the return, a shareholder earns for every unit of capital invested. It can be calculated as: "_Net Profit / Shareholders Equity * 100_", or one can also use ‘DuPont Model’ to calculate RoE. We will use both methods. First, we will calculate the RoE with **DuPont Model**:
RoE = Net Profit Margin × Asset Turnover × Equity Multiple

```python
previous_year = '12/31/2017'
total_assets_current_year = get_val(df_bal,'Total Assets', t)
total_assets_previous_year = get_val(df_bal,'Total Assets', previous_year)
average_total_assets = (total_assets_current_year + total_assets_previous_year)/2

shareholder_equity_current_year = get_val(df_bal,'Total stockholders\' equity', t)
shareholder_equity_previous_year = get_val(df_bal,'Total stockholders\' equity', previous_year)
average_shareholder_equity = (shareholder_equity_current_year + shareholder_equity_previous_year)/2

asset_turnover = total_revenue / average_total_assets
equity_multiple = average_total_assets / average_shareholder_equity

RoE = (net_profit_margin * asset_turnover * equity_multiple) * 100
print("RoE of", symbol, "is ", round((RoE),2))

Output:
RoE of JNJ is  25.51
```

Now, let's calculate RoE without using DuPont Model:

```python
print("RoE of", symbol, "is ",net_income/shareholder_equity_current_year*100)

Output:
RoE of JNJ is  25.60
```

**4. Return on Asset (RoA)**: This ratio gives us an idea as to how much profit a company was able to generate from its assets. It can be calculated as:

(Net income + interest*(1-tax rate)) / Total Average Assets

```python
income_before_tax = get_val(inc_df_no_nan,'Income Before Tax', t)
income_tax_expense = get_val(inc_df_no_nan,'Income Tax Expense', t)
tax_rate= income_tax_expense / income_before_tax
interest=get_val(inc_df_no_nan,'Interest Expense', t)
RoA = (net_income+interest*(1-tax_rate))/average_total_assets
print(round((RoA*100),2), "%")

Output:
9.31 %
```

**5. Return on Capital Employed (RoCE):** The Return on Capital employed gives us an idea about the profitability of the company considering the overall capital employed or how efficiently a company can generate profits from its capital. This ratio can be calculated as "EBIT/Capital Employed".

```python
EBIT = get_val(inc_df_no_nan,'Earnings Before Interest and Taxes', t)
current_liabilities = get_val(df_bal,'Total Current Liabilities', t)

capital_employed = total_assets_current_year - current_liabilities
RoCE = EBIT / capital_employed
print("RoCE is", round((RoCE*100),2), "%")

Output:
RoCE is 17.59 %
```

In the next post we will explore the "Leverage Ratios".





























































