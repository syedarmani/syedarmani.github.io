---
title: 'Scraping Financial Data from Yahoo Finance.'
date: 2020-01-16
permalink: /posts/2020/01/scraping101/
tags:
  - Finance
  - Python
---

In this post, we will use BeautifulSoup package to scrap the historical financial data from Yahoo Finance.

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd

BASE_URL = "https://finance.yahoo.com/quote/"

def get_cash_flow(SYMBOL):
    
    URL = BASE_URL + SYMBOL + "/cash-flow"
    return parse_html(URL)


def get_balance_sheet(SYMBOL):

    URL = BASE_URL  + SYMBOL + "/balance-sheet/"  
    return parse_html(URL)


def get_income_statement(SYMBOL):

    URL = BASE_URL + SYMBOL + "/financials"
    return parse_html(URL)


def get_ratios(SYMBOL):
    
    URL = BASE_URL + SYMBOL + "/key-statistics"     
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find_all('table', {'class':'table-qsp-stats Mt(10px) '})
    
    print("Valuation Measures")
    df_valuation_measures = pd.read_html(str(table))[0]
    df_valuation_measures.columns=['Key','Value']
    display(df_valuation_measures)
    
    print("\n\n\n Stock Price History")
    df_stock_price_history = pd.read_html(str(table))[1]
    df_stock_price_history.columns=['Key','Value']
    display(df_stock_price_history)
    
    print("\n\n\n Share Statistics")
    df_share_statistics = pd.read_html(str(table))[2]
    df_share_statistics.columns=['Key','Value']
    display(df_share_statistics)
    
    print("\n\n\n Dividends & Splits")
    df_dividends_splits = pd.read_html(str(table))[3]
    df_dividends_splits.columns=['Key','Value']
    display(df_dividends_splits)
    
    print("\n\n\n Fiscal Year")
    df_fiscal_year = pd.read_html(str(table))[4]
    df_fiscal_year.columns=['Key','Value']
    display(df_fiscal_year)
    
    print("\n\n\n Profitability")
    df_profitability = pd.read_html(str(table))[5]
    df_profitability.columns=['Key','Value']
    display(df_profitability)
    
    print("\n\n\n Management Effectiveness")
    df_management_effectiveness = pd.read_html(str(table))[6]
    df_management_effectiveness.columns=['Key','Value']
    display(df_management_effectiveness)

    
def parse_html (URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find_all('table', {'class':'Lh(1.7) W(100%) M(0)'})
    df = pd.read_html(str(table), header =0, index_col = 0, flavor = 'bs4')[0]   
    return df


```
Example:
```python
#Get the balance sheet of Apple.
get_balance_sheet("AAPL")

```

<iframe width="100%" height="500" src="/images/plots/balance_sheet.html">Balance Sheet</iframe>
<br />
