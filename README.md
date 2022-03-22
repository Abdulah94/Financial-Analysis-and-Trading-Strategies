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
pip install -r requirements.txt```
5. You can run ```Python jupyter nbconvert --to markdown --execute jubyter-notebook.ipynb``` to export your output to a markdown file.


### Project findings and analysis

##### Macro analysis
We get the result that SP500 and oil price rise steadily before the pandemic, but then drop rapidly until May 2020, and then continue to rise. On the other hand, we can see that after the pandemic, the stock prices of these two high-tech companies(TSLA, NVDA) have soared. We conduct in-depth research on the impact of the pandemic on enterprises’ stocks.

Firstly, we use ARMA (3,3) model to analyze the effect of autocorrelation for different stock’s closed price, then we find their first-order lag autoregressive coefficients are close to 1, it is totally likely that they have unit roots, then I regress the GROWTH RATE	of two stock prices for that of SP500 and oil price, since stock prices might not be affected by oil price and SP500 immediately, I estimate these two variables could have same lag effects on stock price(5 orders), in this case, I assume the lag effect of oil price and SP 500 about five backward periods to stock prices. 

Then we use the augmented Dickey-Fuller test to test whether the regression’s residuals have a unit root, and according to the result, there indeed exists the unit root, or we can’t reject the null hypothesis. Then we get the conclusion that the growth rate of NVDA and TSLA's stock price, oil price, SP500 are cointegrated. 

We also found an interesting thing: the growth rate of the stock has a noticeable lagged reflection to that of oil prices and SP500, since we can clearly see that the coefficients in the second-order, third order, fourth-order are much higher and statistically significant than first order, we can estimate the growth rate of stock price need some time to make any adjustment.


