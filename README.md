# Stocks Trading Strategies & Financial Analysis

### Description and methodology

In this project, we will conduct a macro and micro analysis of daily stock data from the beginning of 2019 up to February 2022 to analyze the behavior and movement of stocks and other financial assets over time. In addition to implementing and testing the fast and slow signals trading strategy to aid traders in buying and selling decisions.

We will start at the micro-level by analyzing the overall behavior of our candidate stocks by plotting their daily closing price, daily volume, and log returns. Next, we will calculate and plot the 7-day and 40-day moving average to conduct the Fast & Slow Signals Trading Strategy. We will report the net accumulated profit/loss by following this strategy using past periods as a testing sample.

Afterward, we will calculate the Value at Risk (VaR) to quantify the extent of possible daily financial loss in selected stocks, as investment professionals and portfolio managers commonly use this metric to determine the size and probabilities of potential losses in their institutional portfolios.

Then, moving to the macro analysis part, we will conduct a thorough time series analysis of the relationship between our candidate stocks and macro-variables; in this model, we chose to analyze the relationship with Brent Oil Prices and S&P500. This part will help investors test and understand our candidate stock's relative movements regarding changes in business cycles and the macroeconomy.
In this part, we'll start by doing fundamental statistic analysis and visualization of before & after the pandemic, then we'll analyze the effect of changes in oil prices and S&P500 on selected stocks. To do so, we plot the ACF & PACF to help in model selection. After the regression, we use the augmented Dicky-Fuller test to check for unit roots and cointegration.


### Data sets

In this analysis, we used daily data for stocks from ["Yahoo Finance"](https://finance.yahoo.com/quote/TSLA/history?p=TSLA) and daily macro-variables data (Oil Prices and S&P500) from ["FRED"](https://fred.stlouisfed.org/series/DCOILBRENTEU) for the period starting from January 2019 up to February 2022.

### Reproducibility

You use this code to test this trading strategy and produce similar micro & macro analysis and visualization, using any stock or macro-variable that comes to mind.

**To do so:**


1. First, download your candidate stock daily data of the pre-defined period from ["Yahoo Finance"](https://finance.yahoo.com/quote/TSLA/history?p=TSLA) 
2. Next, download your daily macro-variable data of the pre-defined period from 
    ["FRED"](https://fred.stlouisfed.org/series/DCOILBRENTEU)
3. Make sure to change variable names in the code to match the new data sets file names.
4. Note that you must choose the type of ARIMA model selected in the regression based on the ACF & PACF plots.
5. Install required libraries before running the code by typing the following command in your terminal, ```Python
pip install -r requirements.txt
```

### Project findings and analysis



