#!/usr/bin/env python3

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as mpl
import seaborn as sb
from scipy.stats import norm

def ImportData(ticker):
    data = yf.download(ticker, '2022-01-01', '2024-05-31')
    return data


def main():
    data = ImportData('AAL')
    print(data.head(5))
    data.iloc[:, 3].plot(figsize=(10,5))
    print(data.iloc[:, 3].head(5))
    mpl.show()

    log_return = np.log(1 + data.iloc[:, 1].pct_change())
    
    sb.displot(log_return.iloc[1:])
    mpl.xlabel("Daily Return")
    mpl.ylabel("Frequency")
    mpl.show()

    u = log_return.mean()
    var = log_return.var()
    drift = u - (0.5*var)

    stdev = log_return.std()
    days = 14
    trials = 1000
    Z = norm.ppf(np.random.rand(days, trials))
    daily_returns = np.exp(drift + stdev * Z)

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1, 3]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]
    

    # Calculate the 95% confidence interval
    lower_bound = np.percentile(price_paths, 2.5)
    upper_bound = np.percentile(price_paths, 97.5)
    # Calculate the 99% confidence interval
    lower_bound_1 = np.percentile(price_paths, 0.1)
    upper_bound_1 = np.percentile(price_paths, 99.9)
    print("95% interval: ", lower_bound, upper_bound)
    print("99% interval: ", lower_bound_1, upper_bound_1)

    # Plotting the price as a probability distribution
    sb.displot(price_paths, bins=30, color='blue', legend=False)
    mpl.axvline(x=lower_bound, color='g', linestyle='--')  # Add a vertical line at the lower bound
    mpl.axvline(x=upper_bound, color='g', linestyle='--')  # Add a vertical line at the upper bound
    mpl.axvline(x=lower_bound_1, color='r', linestyle='--')  # Add a vertical line at the lower bound
    mpl.axvline(x=upper_bound_1, color='r', linestyle='--')  # Add a vertical line at the upper bound
    mpl.xlabel('Price')
    mpl.ylabel('Probability')
    mpl.title('Price Distribution')
    mpl.show()

    # # Download the actual future data
    # data_future = yf.download('AAL', '2024-05-10', '2024-05-28')
    # data_future = data_future.iloc[:, 3]

    # # Plot the simulated price paths and the actual future prices on the same plot
    # mpl.figure(figsize=(10,5))
    # mpl.plot(price_paths, label='Simulated')
    # mpl.plot(data_future.values, 'r', label='Actual', linewidth=3)
    # mpl.xlabel('Days')
    # mpl.ylabel('Price')
    # mpl.title('Price Paths')
    # mpl.legend()
    # mpl.show()
    

if __name__ == '__main__':
    main()

