## Stocks Trading Strategies & Financial Analysis

### Importing Libraries


```python
%load_ext lab_black
```


```python
import pandas as pd
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

sns.set_style("whitegrid")
```

### Setting the relative path


```python
def import_cvs_as_dataframe(stock):
    """This function takes the stock name of the dataset (as a string) and import it as a clean data frame."""

    file_path = os.path.join("../data", stock + ".csv")
    df = (
        pd.read_csv(file_path)
        .assign(Date=lambda stock: pd.to_datetime(stock["Date"]))
        .dropna()
        .set_index("Date")
    )
    return df
```

### Micro-level analysis and Fast & Slow signals trading strategy 


```python
def plot_close_price_on_one_plot(df1, df2, df3, header1, header2, header3):
    """This function takes three data frames with headers, and return a plot to visualize and compare the daily close price for each stock on the same plot"""
    plt.figure(figsize=(10, 6))
    dfx = [df1, df2, df3]

    headerx = [header1, header2, header3]

    for d, h in zip(dfx, headerx):
        d["Close"].loc["2019-01-02":"2022-02-25"].plot(label=h)

    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.title("Daily Closing Prices")
    plt.legend(loc="upper left")

    return plt.show()
```


```python
def plot_compares_open_and_close_price_on_one_plot(
    df1, df2, df3, header1, header2, header3
):
    """This function takes three data frames with headers, and return subplots to visualize and compare the daily close and open price for each stock individually"""

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.set_figheight(26)
    fig.set_figwidth(10)

    dfx = [df1, df2, df3]
    headerx = [header1, header2, header3]
    i = 0
    for d, h in zip(dfx, headerx):

        d["Open"].plot(ax=axes[i])
        d["Close"].plot(ax=axes[i]).legend(loc="lower right")
        axes[i].set_title(h + " Daily Open and Close Prices")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("USD")
        i = i + 1

    return plt.show()
```


```python
def plot_volume_traded(df1, df2, df3, header1, header2, header3):
    """This function takes three data frames with headers, and return subplots to visualize and compare the daily volume traded for each stock individually"""

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.set_figheight(26)
    fig.set_figwidth(10)

    dfx = [df1, df2, df3]
    headerx = [header1, header2, header3]
    i = 0
    for d, h in zip(dfx, headerx):

        d["Volume"].plot(ax=axes[i])
        axes[i].set_title(h + " Daily Volume Traded")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Shares")
        i = i + 1

    return plt.show()
```


```python
def plot_Log_Return(df1, df2, df3, header1, header2, header3):
    """This function takes three data frames with headers, and return histogram subplots to visualize and compare the daily Log return for each stock individually"""

    dfx = [df1, df2, df3]
    for d in dfx:
        d["LogReturn"] = np.log(d["Close"]).shift(-1) - np.log(d["Close"])
        d = d.dropna()

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.set_figheight(26)
    fig.set_figwidth(10)

    headerx = [header1, header2, header3]
    i = 0
    for d, h in zip(dfx, headerx):
        d["LogReturn"].hist(bins=50, ax=axes[i])
        axes[i].set_title(h + " Daily Return")
        axes[i].set_xlabel("Log Return")
        axes[i].set_ylabel("Frequency")
        i = i + 1

    return plt.show()
```

### Value at Risk (VaR) analysis


```python
def Calculating_value_at_risk(df1, df2, df3, header1, header2, header3):
    """This function takes three data frames with headers, and return the value at risk (VAR) for 5%, 95%, 25%, and 75% quantiles"""
    dfx = [df1, df2, df3]
    mu = []
    sigma = []
    for d in dfx:
        m = d["LogReturn"].mean()
        s = d["LogReturn"].std(ddof=1)
        mu.append(m)
        sigma.append(s)

    headerx = [header1, header2, header3]

    for m, s, h in zip(mu, sigma, headerx):
        print("VAR for " + h)
        print("5% quantile ", norm.ppf(0.05, m, s))
        print("95% quantile ", norm.ppf(0.95, m, s))
        print("25% quantile ", norm.ppf(0.25, m, s))
        print("75% quantile ", norm.ppf(0.75, m, s))
```


```python
def adding_moving_averages(df):
    """This function takes a data frame of the stock and returns a new data frame with 3 columns
    [Closing price, and 2 Moving Averages (fast & slow signals)] that is ready for plotting the stock signals."""

    df_ma = df["Close"].to_frame()
    df_ma["MA7"] = df_ma["Close"].rolling(7).mean()
    df_ma["MA40"] = df_ma["Close"].rolling(40).mean()
    df_ma = df_ma.dropna()

    return df_ma
```


```python
def plot_fast_slow_signals(df, header):
    """This function takes a data frame of a certain stock and a name of the plot title as a string,
    and returns a plot to visualize the signals movement of the stock."""

    df_ma = adding_moving_averages(df)
    plt.figure(figsize=(10, 8))
    df_ma["MA7"].loc["2021-01-01":"2021-12-31"].plot(label="Fast Signal")
    df_ma["MA40"].loc["2021-01-01":"2021-12-31"].plot(label="Slow Signal")
    df_ma["Close"].loc["2021-01-01":"2021-12-31"].plot(
        label="Closing Price", title=header
    )
    plt.legend()

    return plt.show()
```


```python
def trading_based_on_signals(df):
    """This function is for testing the trading strategy that is based on the fast & slow signals,
    it takes a stock data frame cleans it in the desired format, and then adds to it shares column (boolean)
    which buys one share of stock when the fast signal goes above the slow signal & sells it when the fast signal
    drops below the slow signal. Another two columns added to calculate the daily profit/loss while we have the stock (long position)
    and ignores the profit calculation when we don't have the stock (short position). Finally, one last culomn that
    calculates the cumulative wealth from the daily profit/loss; Then the function returns a new data frame for trading startegy testing."""

    df2 = adding_moving_averages(df)
    df2["Shares"] = [
        1 if (df2.loc[ei, "MA7"] > df2.loc[ei, "MA40"]) else 0 for ei in df2.index
    ]
    df2["Close1"] = df2["Close"].shift(-1)
    df2["Profit"] = [
        (df2.loc[ei, "Close1"] - df2.loc[ei, "Close"])
        if df2.loc[ei, "Shares"] == 1
        else 0
        for ei in df2.index
    ]
    df2["Wealth"] = df2["Profit"].loc["2021-01-01":"2022-02-25"].cumsum()

    return df2
```


```python
def plot_profit(df, header):

    performance_df = trading_based_on_signals(df)
    performance_df["Profit"].loc["2021-01-01":"2022-02-25"].plot(title=header)
    plt.axhline(y=0, color="red")

    return plt.show()
```


```python
def plot_wealth(df, header):

    performance_df = trading_based_on_signals(df)
    performance_df["Wealth"].loc["2021-01-01":"2022-02-25"].plot()
    plt.title(
        header
        + ": ${} in Net Profit".format(
            int(performance_df.loc[performance_df.index[-2], "Wealth"])
        )
    )

    return plt.show()
```


```python
xom = import_cvs_as_dataframe("XOM")
tsla = import_cvs_as_dataframe("TSLA")
nvda = import_cvs_as_dataframe("NVDA")
```


```python
plot_close_price_on_one_plot(tsla, nvda, xom, "Tesla", "NVIDIA", "Exxon Mobil")
plot_compares_open_and_close_price_on_one_plot(
    tsla, nvda, xom, "Tesla", "NVIDIA", "Exxon Mobil"
)
plot_volume_traded(tsla, nvda, xom, "Tesla", "NVIDIA", "Exxon Mobil")
plot_fast_slow_signals(tsla, "Tesla Stock Signals Movement")
plot_fast_slow_signals(nvda, "NVIDIA Stock Signals Movement")
plot_fast_slow_signals(xom, "Exxon Mobil Stock Signals Movement")
plot_profit(tsla, "Tesla Daily Profit/Loss")
plot_profit(nvda, "NVIDIA Daily Profit/Loss")
plot_profit(xom, "Exxon Mobil Daily Profit/Loss")
plot_wealth(tsla, "Tesla Stock")
plot_wealth(nvda, "NVIDIA Stock")
plot_wealth(xom, "Exxon Mobil Stock")
plot_Log_Return(tsla, nvda, xom, "Tesla", "NVIDIA", "Exxon Mobil")
Calculating_value_at_risk(tsla, nvda, xom, "Tesla", "NVIDIA", "Exxon Mobil")
```


    
![png](jubyter-notebook_files/jubyter-notebook_19_0.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_1.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_2.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_3.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_4.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_5.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_6.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_7.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_8.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_9.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_10.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_11.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_19_12.png)
    


    VAR for Tesla
    5% quantile  -0.06653962651049658
    95% quantile  0.07301154681305047
    25% quantile  -0.025376262387034748
    75% quantile  0.03184818268958867
    VAR for NVIDIA
    5% quantile  -0.04860526906430513
    95% quantile  0.053540237453345546
    25% quantile  -0.01847544229020068
    75% quantile  0.02341041067924111
    VAR for Exxon Mobil
    5% quantile  -0.037681040759660736
    95% quantile  0.037959626458253515
    25% quantile  -0.015369337797380932
    75% quantile  0.01564792349597372


## Macro-level analysis

### Data cleaning & merging


```python
oil_price = (
    pd.read_csv("../data/DCOILBRENTEU.csv")
    .rename(columns={"DATE": "Date"})
    .assign(Date=lambda oil_price: pd.to_datetime(oil_price["Date"]))
    .assign(
        DCOILBRENTEU=lambda oil_price: pd.to_numeric(
            oil_price["DCOILBRENTEU"], errors="coerce"
        )
    )
)

sp500 = (
    pd.read_csv("../data/SP500.csv")
    .rename(columns={"DATE": "Date"})
    .assign(Date=lambda oil_price: pd.to_datetime(oil_price["Date"]))
    .assign(SP500=lambda SP500: pd.to_numeric(SP500["SP500"], errors="coerce"))
)


def extract_stock_price(stock):
    """This functions takes stock price from dataset and import a cleaned dataframe"""
    file_path = os.path.join("../data", stock + ".csv")
    file_folder = pd.read_csv(file_path).assign(
        Date=lambda stock: pd.to_datetime(stock["Date"])
    )
    return file_folder


tsla = extract_stock_price("TSLA")
nvda = extract_stock_price("NVDA")


def merge_dataset(data1, data2):
    # merge two dataset based on Date
    new_data_frame = pd.merge(data1, data2, on=["Date"])
    return new_data_frame


oil_sp = merge_dataset(oil_price, sp500)
oil_sp_nvda = merge_dataset(oil_sp, nvda)
oil_sp_nvda_tsla = pd.merge(
    oil_sp_nvda, tsla, on=["Date"], suffixes=("_nvda", "_tsla")
).set_index(["Date"])
```

### Macro trend analysis


```python
# The following code will do some basic statsical analysis, before and after pandemic
oil_sp_nvda_tsla.loc["2020-02":"2022-02"].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DCOILBRENTEU</th>
      <th>SP500</th>
      <th>Open_nvda</th>
      <th>High_nvda</th>
      <th>Low_nvda</th>
      <th>Close_nvda</th>
      <th>Adj Close_nvda</th>
      <th>Volume_nvda</th>
      <th>Open_tsla</th>
      <th>High_tsla</th>
      <th>Low_tsla</th>
      <th>Close_tsla</th>
      <th>Adj Close_tsla</th>
      <th>Volume_tsla</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>514.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>5.220000e+02</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>522.000000</td>
      <td>5.220000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>58.523444</td>
      <td>3819.080268</td>
      <td>158.110953</td>
      <td>161.058362</td>
      <td>154.877481</td>
      <td>158.090704</td>
      <td>157.968776</td>
      <td>4.348446e+07</td>
      <td>581.853127</td>
      <td>595.254057</td>
      <td>567.189809</td>
      <td>581.875957</td>
      <td>581.875957</td>
      <td>4.786885e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.773545</td>
      <td>623.715047</td>
      <td>69.163560</td>
      <td>70.634270</td>
      <td>67.240387</td>
      <td>68.869142</td>
      <td>68.904937</td>
      <td>2.126702e+07</td>
      <td>297.182418</td>
      <td>303.276969</td>
      <td>289.979378</td>
      <td>296.748379</td>
      <td>296.748379</td>
      <td>3.614618e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.120000</td>
      <td>2237.400000</td>
      <td>50.025002</td>
      <td>52.485001</td>
      <td>45.169998</td>
      <td>49.099998</td>
      <td>48.997280</td>
      <td>9.788400e+06</td>
      <td>74.940002</td>
      <td>80.972000</td>
      <td>70.101997</td>
      <td>72.244003</td>
      <td>72.244003</td>
      <td>9.800600e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>42.580000</td>
      <td>3334.885000</td>
      <td>112.611252</td>
      <td>113.808749</td>
      <td>109.780623</td>
      <td>112.424997</td>
      <td>112.241015</td>
      <td>2.716580e+07</td>
      <td>320.399994</td>
      <td>330.058998</td>
      <td>297.700004</td>
      <td>311.631996</td>
      <td>311.631996</td>
      <td>2.334440e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>61.130000</td>
      <td>3871.015000</td>
      <td>137.336251</td>
      <td>138.772499</td>
      <td>134.477501</td>
      <td>136.493752</td>
      <td>136.331283</td>
      <td>3.782460e+07</td>
      <td>637.035004</td>
      <td>652.910004</td>
      <td>620.505005</td>
      <td>642.570007</td>
      <td>642.570007</td>
      <td>3.440770e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.582500</td>
      <td>4384.645000</td>
      <td>205.029995</td>
      <td>207.297501</td>
      <td>202.082504</td>
      <td>205.158749</td>
      <td>205.089641</td>
      <td>5.636700e+07</td>
      <td>778.917526</td>
      <td>795.402512</td>
      <td>768.527497</td>
      <td>781.122498</td>
      <td>781.122498</td>
      <td>6.321078e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>101.660000</td>
      <td>4796.560000</td>
      <td>335.170013</td>
      <td>346.470001</td>
      <td>320.359985</td>
      <td>333.760010</td>
      <td>333.662292</td>
      <td>1.463684e+08</td>
      <td>1234.410034</td>
      <td>1243.489990</td>
      <td>1217.000000</td>
      <td>1229.910034</td>
      <td>1229.910034</td>
      <td>3.046940e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
oil_sp_nvda_tsla.loc["2019-01":"2020-01"].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DCOILBRENTEU</th>
      <th>SP500</th>
      <th>Open_nvda</th>
      <th>High_nvda</th>
      <th>Low_nvda</th>
      <th>Close_nvda</th>
      <th>Adj Close_nvda</th>
      <th>Volume_nvda</th>
      <th>Open_tsla</th>
      <th>High_tsla</th>
      <th>Low_tsla</th>
      <th>Close_tsla</th>
      <th>Adj Close_tsla</th>
      <th>Volume_tsla</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>2.730000e+02</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>2.730000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64.268828</td>
      <td>2941.425348</td>
      <td>44.961740</td>
      <td>45.603516</td>
      <td>44.348562</td>
      <td>44.997408</td>
      <td>44.803496</td>
      <td>4.437294e+07</td>
      <td>58.452044</td>
      <td>59.511172</td>
      <td>57.508315</td>
      <td>58.631099</td>
      <td>58.631099</td>
      <td>4.973016e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.349552</td>
      <td>174.670059</td>
      <td>7.616731</td>
      <td>7.591335</td>
      <td>7.615774</td>
      <td>7.622473</td>
      <td>7.636614</td>
      <td>2.318681e+07</td>
      <td>17.025550</td>
      <td>17.504145</td>
      <td>16.804551</td>
      <td>17.315205</td>
      <td>17.315205</td>
      <td>2.833308e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>53.230000</td>
      <td>2447.890000</td>
      <td>32.660000</td>
      <td>33.790001</td>
      <td>31.922501</td>
      <td>31.997499</td>
      <td>31.787331</td>
      <td>1.388640e+07</td>
      <td>36.220001</td>
      <td>37.335999</td>
      <td>35.397999</td>
      <td>35.793999</td>
      <td>35.793999</td>
      <td>1.232800e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>61.040000</td>
      <td>2832.940000</td>
      <td>39.037498</td>
      <td>39.514999</td>
      <td>38.277500</td>
      <td>39.112499</td>
      <td>38.895641</td>
      <td>3.034880e+07</td>
      <td>46.402000</td>
      <td>47.200001</td>
      <td>45.750000</td>
      <td>46.801998</td>
      <td>46.801998</td>
      <td>3.081300e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>63.990000</td>
      <td>2926.460000</td>
      <td>43.305000</td>
      <td>44.322498</td>
      <td>42.884998</td>
      <td>43.500000</td>
      <td>43.351063</td>
      <td>3.943000e+07</td>
      <td>53.750000</td>
      <td>54.560001</td>
      <td>52.944000</td>
      <td>53.556000</td>
      <td>53.556000</td>
      <td>4.054500e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>67.050000</td>
      <td>3020.970000</td>
      <td>49.187500</td>
      <td>49.822498</td>
      <td>48.825001</td>
      <td>49.092499</td>
      <td>48.924416</td>
      <td>5.116160e+07</td>
      <td>64.344002</td>
      <td>66.000000</td>
      <td>63.223999</td>
      <td>65.542000</td>
      <td>65.542000</td>
      <td>5.900300e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>74.940000</td>
      <td>3329.620000</td>
      <td>64.375000</td>
      <td>64.875000</td>
      <td>62.250000</td>
      <td>63.215000</td>
      <td>63.045044</td>
      <td>2.511528e+08</td>
      <td>128.000000</td>
      <td>130.600006</td>
      <td>126.503998</td>
      <td>130.113998</td>
      <td>130.113998</td>
      <td>1.568450e+08</td>
    </tr>
  </tbody>
</table>
</div>



The range for the oil price, SP500 and the two stocks prices are big, we can estimate that during the pandemic,both of them exprienced a huge flactulation. Next, We use line charts to more intuitively see their trends before and after the epidemic. To simplify, we focus on close price and compare there trends before and after pandemic


```python
def line_plot_for_otherVar_pandemic(variable):
    oil_sp_nvda_tsla.loc["2020-02":"2022-02", variable].plot()
    oil_sp_nvda_tsla.loc["2019-01":"2020-01", variable].plot(title=variable).legend(
        ["After Pandemic", "Before Pandemic"]
    )


line_plot_for_otherVar_pandemic("DCOILBRENTEU")
```


    
![png](jubyter-notebook_files/jubyter-notebook_27_0.png)
    


We can see that crude oil prices fluctuated before the epidemic, and then fell rapidly. After the epidemic, until the end of May 2020, it began to rise continuously 


```python
line_plot_for_otherVar_pandemic("SP500")
```


    
![png](jubyter-notebook_files/jubyter-notebook_29_0.png)
    


We can see SP500 rise steadily before the pandemic, but then drop rapidly until May 2020, and then continue to rise. Now we can see that after pandemic, the stock prices of these two high-tech companies have soared. We will conduct in-depth research on the impact of the pandemic on the development of traditional industrial products and high-tech enterprises. 

Because the bases number of each indicator are different, we will analyze the growth rate of each indicator before and after the epidemic, and use a simple econometric model to analyze the impact of oil prices and SP500 on these two stocks.

### Regression analysis


```python
# we first construct a new dataset, only choose the close price for tsla and nvda
stock_price_anal = oil_sp_nvda_tsla[
    ["Close_nvda", "Close_tsla", "DCOILBRENTEU", "SP500"]
].dropna()
# mutate variables for calculating growth rates

stock_price_anal["growth_nvda"] = stock_price_anal["Close_nvda"].pct_change() * 100
stock_price_anal["growth_tsla"] = stock_price_anal["Close_tsla"].pct_change() * 100
stock_price_anal["growth_oil_price"] = (
    stock_price_anal["DCOILBRENTEU"].pct_change() * 100
)
stock_price_anal["growth_sp"] = stock_price_anal["SP500"].pct_change() * 100
```

Then we'll set the two company's stock price's grwoth rates as  dependent variable, SP500 and oil price's growth rates as dependent variable, analysing the effect of change in oil price on these two company's stock prices


```python
def acf_pacf(variable):
    """plot acf and pacf for stock price"""
    plot_acf(stock_price_anal[[variable]].dropna(), lags=20)
    plot_pacf(stock_price_anal[[variable]].dropna(), lags=20)
```


```python
acf_pacf("growth_nvda")
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/graphics/tsaplots.py:353: FutureWarning: The default method 'yw' can produce PACF values outside of the [-1,1] interval. After 0.13, the default will change tounadjusted Yule-Walker ('ywm'). You can use this method now by setting method='ywm'.
      FutureWarning,



    
![png](jubyter-notebook_files/jubyter-notebook_35_1.png)
    



    
![png](jubyter-notebook_files/jubyter-notebook_35_2.png)
    


We can see that the acf and pacf for two stock prices do not provide some reliable information, we use ARIMA model to analyse them and check the result


```python
model = sm.tsa.arima.ARIMA(stock_price_anal[["Close_tsla"]], order=(3, 0, 3))
arima_result = model.fit()
arima_result.summary()
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)





<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Close_tsla</td>    <th>  No. Observations:  </th>    <td>787</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(3, 0, 3)</td>  <th>  Log Likelihood     </th> <td>-3507.136</td>
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 22 Mar 2022</td> <th>  AIC                </th> <td>7030.272</td> 
</tr>
<tr>
  <th>Time:</th>                <td>14:40:33</td>     <th>  BIC                </th> <td>7067.618</td> 
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>7044.630</td> 
</tr>
<tr>
  <th></th>                      <td> - 787</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>  <td>  399.8272</td> <td>  406.312</td> <td>    0.984</td> <td> 0.325</td> <td> -396.529</td> <td> 1196.184</td>
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.9701</td> <td>    0.283</td> <td>    3.424</td> <td> 0.001</td> <td>    0.415</td> <td>    1.525</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>    0.5939</td> <td>    0.185</td> <td>    3.213</td> <td> 0.001</td> <td>    0.232</td> <td>    0.956</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.5648</td> <td>    0.251</td> <td>   -2.253</td> <td> 0.024</td> <td>   -1.056</td> <td>   -0.074</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.0011</td> <td>    0.275</td> <td>   -0.004</td> <td> 0.997</td> <td>   -0.541</td> <td>    0.539</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>   -0.6031</td> <td>    0.253</td> <td>   -2.383</td> <td> 0.017</td> <td>   -1.099</td> <td>   -0.107</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>    0.0876</td> <td>    0.024</td> <td>    3.584</td> <td> 0.000</td> <td>    0.040</td> <td>    0.136</td>
</tr>
<tr>
  <th>sigma2</th> <td>  431.7637</td> <td>    9.541</td> <td>   45.252</td> <td> 0.000</td> <td>  413.063</td> <td>  450.464</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>      <td>0.00</td>  <th>  Jarque-Bera (JB):  </th> <td>3193.19</td>
</tr>
<tr>
  <th>Prob(Q):</th>                 <td>1.00</td>  <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>196.40</td> <th>  Skew:              </th>  <td>0.19</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>     <td>0.00</td>  <th>  Kurtosis:          </th>  <td>12.86</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



We can see that Tesla stock price is autocorrelated and moving average in its lagged variable.


```python
model = sm.tsa.arima.ARIMA(stock_price_anal[["Close_nvda"]], order=(3, 0, 3))
arima_result = model.fit()
arima_result.summary()
```

    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    /opt/conda/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)





<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Close_nvda</td>    <th>  No. Observations:  </th>    <td>787</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(3, 0, 3)</td>  <th>  Log Likelihood     </th> <td>-2303.593</td>
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 22 Mar 2022</td> <th>  AIC                </th> <td>4623.186</td> 
</tr>
<tr>
  <th>Time:</th>                <td>14:40:34</td>     <th>  BIC                </th> <td>4660.532</td> 
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>4637.544</td> 
</tr>
<tr>
  <th></th>                      <td> - 787</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>  <td>  120.6304</td> <td>   84.797</td> <td>    1.423</td> <td> 0.155</td> <td>  -45.569</td> <td>  286.830</td>
</tr>
<tr>
  <th>ar.L1</th>  <td>    1.5013</td> <td>    0.136</td> <td>   11.004</td> <td> 0.000</td> <td>    1.234</td> <td>    1.769</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -0.1735</td> <td>    0.262</td> <td>   -0.663</td> <td> 0.508</td> <td>   -0.687</td> <td>    0.340</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.3281</td> <td>    0.141</td> <td>   -2.323</td> <td> 0.020</td> <td>   -0.605</td> <td>   -0.051</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.5746</td> <td>    0.133</td> <td>   -4.306</td> <td> 0.000</td> <td>   -0.836</td> <td>   -0.313</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>   -0.3925</td> <td>    0.148</td> <td>   -2.646</td> <td> 0.008</td> <td>   -0.683</td> <td>   -0.102</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>    0.1571</td> <td>    0.025</td> <td>    6.365</td> <td> 0.000</td> <td>    0.109</td> <td>    0.205</td>
</tr>
<tr>
  <th>sigma2</th> <td>   20.2579</td> <td>    0.516</td> <td>   39.275</td> <td> 0.000</td> <td>   19.247</td> <td>   21.269</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.00</td>  <th>  Jarque-Bera (JB):  </th> <td>2092.51</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.97</td>  <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>41.52</td> <th>  Skew:              </th>  <td>0.60</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>  <td>10.90</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



we can see that nvda stock price is autocorrelated and moving average in its lagged variable

Since their first-order lag autoregressive coefficients are close to 1, it is likely that they have unit roots, then I regress the two stock prices for SP500 and oil price, since stock prices might not be affected by oil price and SP500 immediately, I estimate these two variables could have same lag effects on stock price,in this case, I assume the lag effect of oil price and SP 500 about three backward periods to stock prices.


```python
# next, we'll create lagged variable for oil price and SP500, since we need to test the lasting effect of these macro variables
# We use growth rates for stock prices as dependent variables, and the growth rate of oil price and sp500 as independent variables.
# After the regression, we test the unit root of the residual, try to test the cointegration problems
y = stock_price_anal[["growth_nvda"]]
# create lagged variable for oil price growth rate
stock_price_anal[["lag1_growth_oil_price"]] = stock_price_anal[
    ["growth_oil_price"]
].shift(1)
stock_price_anal[["lag2_growth_oil_price"]] = stock_price_anal[
    ["growth_oil_price"]
].shift(2)
stock_price_anal[["lag3_growth_oil_price"]] = stock_price_anal[
    ["growth_oil_price"]
].shift(3)
stock_price_anal[["lag4_growth_oil_price"]] = stock_price_anal[
    ["growth_oil_price"]
].shift(4)
stock_price_anal[["lag5_growth_oil_price"]] = stock_price_anal[
    ["growth_oil_price"]
].shift(5)


# create lagged variable for SP500 growth rate
stock_price_anal[["lag1_growth_sp"]] = stock_price_anal[["growth_sp"]].shift(1)
stock_price_anal[["lag2_growth_sp"]] = stock_price_anal[["growth_sp"]].shift(2)
stock_price_anal[["lag3_growth_sp"]] = stock_price_anal[["growth_sp"]].shift(3)
stock_price_anal[["lag4_growth_sp"]] = stock_price_anal[["growth_sp"]].shift(4)
stock_price_anal[["lag5_growth_sp"]] = stock_price_anal[["growth_sp"]].shift(5)

# create independent variable
X = stock_price_anal[
    [
        "lag1_growth_oil_price",
        "lag2_growth_oil_price",
        "lag3_growth_oil_price",
        "lag4_growth_oil_price",
        "lag5_growth_oil_price",
        "lag1_growth_sp",
        "lag2_growth_sp",
        "lag3_growth_sp",
        "lag4_growth_sp",
        "lag5_growth_sp",
    ]
]
nvda_reg = sm.OLS(y, X, missing="drop").fit()
nvda_reg_resid = sm.OLS(y, X, missing="drop").fit().resid
```


```python
# Then use augmented Dickey-Fuller test to test whether the resid has unit root

adfuller(nvda_reg_resid)
```




    (-27.13210367211541,
     0.0,
     0,
     780,
     {'1%': -3.4387614757350087,
      '5%': -2.865252556432172,
      '10%': -2.5687469247205788},
     3790.419539691179)




We can see that test-statistic is -27.19, p- value is 0, we can reject null hypothesis that resid has unit root, so resid is not I(1) 

On the other hand, we can conclude that the growth rate of nvda's stock price, oil price, SP500 are cointegrated.




```python
nvda_reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>growth_nvda</td>   <th>  R-squared (uncentered):</th>      <td>   0.085</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.073</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   7.147</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 22 Mar 2022</td> <th>  Prob (F-statistic):</th>          <td>7.89e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:40:34</td>     <th>  Log-Likelihood:    </th>          <td> -1962.9</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   781</td>      <th>  AIC:               </th>          <td>   3946.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   771</td>      <th>  BIC:               </th>          <td>   3992.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>              <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>lag1_growth_oil_price</th> <td>    0.0139</td> <td>    0.027</td> <td>    0.517</td> <td> 0.605</td> <td>   -0.039</td> <td>    0.067</td>
</tr>
<tr>
  <th>lag2_growth_oil_price</th> <td>    0.0692</td> <td>    0.027</td> <td>    2.571</td> <td> 0.010</td> <td>    0.016</td> <td>    0.122</td>
</tr>
<tr>
  <th>lag3_growth_oil_price</th> <td>    0.0793</td> <td>    0.027</td> <td>    2.903</td> <td> 0.004</td> <td>    0.026</td> <td>    0.133</td>
</tr>
<tr>
  <th>lag4_growth_oil_price</th> <td>   -0.0253</td> <td>    0.027</td> <td>   -0.925</td> <td> 0.355</td> <td>   -0.079</td> <td>    0.028</td>
</tr>
<tr>
  <th>lag5_growth_oil_price</th> <td>   -0.0069</td> <td>    0.027</td> <td>   -0.251</td> <td> 0.802</td> <td>   -0.061</td> <td>    0.047</td>
</tr>
<tr>
  <th>lag1_growth_sp</th>        <td>   -0.5360</td> <td>    0.088</td> <td>   -6.113</td> <td> 0.000</td> <td>   -0.708</td> <td>   -0.364</td>
</tr>
<tr>
  <th>lag2_growth_sp</th>        <td>    0.0969</td> <td>    0.090</td> <td>    1.076</td> <td> 0.282</td> <td>   -0.080</td> <td>    0.274</td>
</tr>
<tr>
  <th>lag3_growth_sp</th>        <td>    0.0058</td> <td>    0.090</td> <td>    0.065</td> <td> 0.948</td> <td>   -0.170</td> <td>    0.182</td>
</tr>
<tr>
  <th>lag4_growth_sp</th>        <td>    0.0042</td> <td>    0.086</td> <td>    0.049</td> <td> 0.961</td> <td>   -0.165</td> <td>    0.174</td>
</tr>
<tr>
  <th>lag5_growth_sp</th>        <td>    0.0726</td> <td>    0.085</td> <td>    0.858</td> <td> 0.391</td> <td>   -0.094</td> <td>    0.239</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>47.411</td> <th>  Durbin-Watson:     </th> <td>   1.925</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 170.187</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.119</td> <th>  Prob(JB):          </th> <td>1.11e-37</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.274</td> <th>  Cond. No.          </th> <td>    4.41</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



According to the result, we can see that nvda stock price growth rate has a minimal relationship with lagged oil price and SP 500 growth rate the interesting thing is, the relationship with dependent variables are almost positive, sometimes the coeffcient of lagged variable is negative, but it is not very statistically signiciant. 


```python
# then do the same thing with the growth rate of tsla, check the unit root of residual
y = stock_price_anal[["growth_tsla"]]
X = stock_price_anal[
    [
        "lag1_growth_oil_price",
        "lag2_growth_oil_price",
        "lag3_growth_oil_price",
        "lag4_growth_oil_price",
        "lag5_growth_oil_price",
        "lag1_growth_sp",
        "lag2_growth_sp",
        "lag3_growth_sp",
        "lag4_growth_sp",
        "lag5_growth_sp",
    ]
]
tsla_reg = sm.OLS(y, X, missing="drop").fit()
tsla_reg_resid = sm.OLS(y, X, missing="drop").fit().resid
adfuller(tsla_reg_resid)
```




    (-27.46647485426182,
     0.0,
     0,
     780,
     {'1%': -3.4387614757350087,
      '5%': -2.865252556432172,
      '10%': -2.5687469247205788},
     4353.546024132106)



we can also see that p-value is 0, which means we can reject null hypothesis that resid has unit root, so resid is not I(1),
so the growth rate of tsla's stock price, oil price, SP500 are cointegrated, 


```python
tsla_reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>growth_tsla</td>   <th>  R-squared (uncentered):</th>      <td>   0.033</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.020</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   2.600</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 22 Mar 2022</td> <th>  Prob (F-statistic):</th>           <td>0.00416</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:40:34</td>     <th>  Log-Likelihood:    </th>          <td> -2237.5</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   781</td>      <th>  AIC:               </th>          <td>   4495.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   771</td>      <th>  BIC:               </th>          <td>   4542.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>              <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>lag1_growth_oil_price</th> <td>   -0.0054</td> <td>    0.038</td> <td>   -0.142</td> <td> 0.887</td> <td>   -0.081</td> <td>    0.070</td>
</tr>
<tr>
  <th>lag2_growth_oil_price</th> <td>    0.0834</td> <td>    0.038</td> <td>    2.180</td> <td> 0.030</td> <td>    0.008</td> <td>    0.158</td>
</tr>
<tr>
  <th>lag3_growth_oil_price</th> <td>    0.1085</td> <td>    0.039</td> <td>    2.793</td> <td> 0.005</td> <td>    0.032</td> <td>    0.185</td>
</tr>
<tr>
  <th>lag4_growth_oil_price</th> <td>   -0.0671</td> <td>    0.039</td> <td>   -1.723</td> <td> 0.085</td> <td>   -0.143</td> <td>    0.009</td>
</tr>
<tr>
  <th>lag5_growth_oil_price</th> <td>    0.0371</td> <td>    0.039</td> <td>    0.954</td> <td> 0.341</td> <td>   -0.039</td> <td>    0.113</td>
</tr>
<tr>
  <th>lag1_growth_sp</th>        <td>   -0.0285</td> <td>    0.125</td> <td>   -0.229</td> <td> 0.819</td> <td>   -0.273</td> <td>    0.216</td>
</tr>
<tr>
  <th>lag2_growth_sp</th>        <td>    0.1576</td> <td>    0.128</td> <td>    1.231</td> <td> 0.219</td> <td>   -0.094</td> <td>    0.409</td>
</tr>
<tr>
  <th>lag3_growth_sp</th>        <td>   -0.0503</td> <td>    0.127</td> <td>   -0.395</td> <td> 0.693</td> <td>   -0.300</td> <td>    0.200</td>
</tr>
<tr>
  <th>lag4_growth_sp</th>        <td>    0.2339</td> <td>    0.123</td> <td>    1.905</td> <td> 0.057</td> <td>   -0.007</td> <td>    0.475</td>
</tr>
<tr>
  <th>lag5_growth_sp</th>        <td>   -0.0196</td> <td>    0.120</td> <td>   -0.163</td> <td> 0.871</td> <td>   -0.256</td> <td>    0.216</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>85.638</td> <th>  Durbin-Watson:     </th> <td>   1.955</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 499.204</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.275</td> <th>  Prob(JB):          </th> <td>3.97e-109</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.878</td> <th>  Cond. No.          </th> <td>    4.41</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



According to the result, we can see that tsla stock price growth rate has a minimal relationship with lagged oil price and SP 500 growth rate the interesting thing is, the relationship with dependent variables are almost positive, sometimes the coeffcient of lagged variable is negative, but it is not very statistically signiciant. 

And we can find a interesting thing, which is the growth rate of the stock has a noticeable lagged reflection to that of oil prices and SP500, since we can clearly see that the coefficients in second order, third order, fourth order are much higher and statistically siginiciant than first order, we can estimate the growth rate of stock price need some time to make any adjustment.


```python

```
