#!/usr/bin/env python3

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as mpl
import seaborn as sb
from scipy.stats import norm


class IronCondor:
    def __init__(self, sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p, max_profit, max_loss_p, max_loss_c, break_even_p, break_even_c, rr):
        self.sp_s = sp_s
        self.sc_s = sc_s
        self.lp_s = lp_s
        self.lc_s = lc_s
        self.sp_p = sp_p
        self.sc_p = sc_p
        self.lp_p = lp_p
        self.lc_p = lc_p
        self.max_profit = max_profit
        self.max_loss_p = max_loss_p
        self.max_loss_c = max_loss_c
        self.break_even_p = break_even_p
        self.break_even_c = break_even_c
        self.rr = rr
      
def calculateRR(max_profit, max_loss_p, max_loss_c):
    return max_profit / min(max_loss_p, max_loss_c)

# Note that this is set up for an 80% interval, needs to be modified for different intervals
def calculateExpectedValue(rr):
    return abs(rr)*0.8 - 1*0.2


def calculateStrategy(sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p):
    max_profit = (sp_p + sc_p - lp_p - lc_p) * 100.0
    max_loss_p = max_profit - (sp_s - lp_s) * 100.0
    max_loss_c = max_profit - (lc_s - sc_s) * 100.0
    break_even_p = sp_s - (sp_p + lp_p)
    break_even_c = sc_s + (sc_p + lc_p)

    return max_profit, max_loss_p, max_loss_c, break_even_p, break_even_c

def importData(ticker):
    data = yf.download(ticker, '2022-01-01', '2024-05-10')
    return data

def ironCondorModel(price_paths):
    # Define Iron Condor strikes based on percentiles and round to the nearest integer
    sp_s = round(np.percentile(price_paths, 10))
    sc_s = round(np.percentile(price_paths, 90))
    lp_s = round(np.percentile(price_paths, 2.5)) 
    lc_s = round(np.percentile(price_paths, 97.5))
    
    # Retrieve the option chain for SPY
    try:
        ticker = yf.Ticker("SPY")
        opt = ticker.option_chain("2024-06-21")  # replace with your desired expiration date
    except:
        ticker = yf.Ticker("SPY")
        opt = ticker.option_chain("2024-06-21")  # replace with your desired expiration date

    ##############################################################################################
    # Retrieve the premium for each option contract
    try: 
        sc_p = opt.calls.loc[opt.calls['strike'] == sc_s, 'lastPrice'].values[0]
        sp_p = opt.puts.loc[opt.puts['strike'] == sp_s, 'lastPrice'].values[0]
        lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
    except:
        sc_p = opt.calls.loc[opt.calls['strike'] == sc_s, 'lastPrice'].values[0]
        sp_p = opt.puts.loc[opt.puts['strike'] == sp_s, 'lastPrice'].values[0]
        lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
        
    print("Strike prices: ", lp_s, sp_s, sc_s, lc_s)
    print("Premiums: ", lp_p, sp_p, sc_p, lc_p)

    # Calculate statistics for the Iron Condor
    max_profit, max_loss_p, max_loss_c, break_even_p, break_even_c = calculateStrategy(sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p)
    rr = calculateRR(max_profit, max_loss_p, max_loss_c)
    
    # Print the statistics for the Iron Condor
    print("Max profit: ", max_profit)
    print("Max loss - put: ", max_loss_p)
    print("Max loss - call: ", max_loss_c)
    print("Break-even points: ", break_even_p, break_even_c)
    print("Risk-reward ratio: ", rr)
    print("Expected value: ", calculateExpectedValue(rr)*abs(min(max_loss_p, max_loss_c)))
    print("Expected value (as a ratio): ", calculateExpectedValue(rr))
    print("\n")

    ################################################################################################

    # Retrieve all possible strike prices less than the short put strike
    lp_s_possible = opt.puts[opt.puts['strike'] < sp_s]['strike'].values
    # Retrieve all possible strike prices greater than the short call strike
    lc_s_possible = opt.calls[opt.calls['strike'] > sc_s]['strike'].values
    
    best_rr = float('inf')
    iron_condor = None
    # Find the optimal outer contracts for the Iron Condor
    for lp_s in lp_s_possible:
        # Retrieve the premium for each option contract
        try: 
            lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        except:
            lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        for lc_s in lc_s_possible:
            # Retrieve the premium for each option contract
            try: 
                lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
            except:
                lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
            # Calculate the statistics for the Iron Condor
            max_profit, max_loss_p, max_loss_c, break_even_p, break_even_c = calculateStrategy(sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p)
            # Calculate the best risk-reward ratio
            rr = calculateRR(max_profit, max_loss_p, max_loss_c)
            if rr < best_rr:
                best_rr = rr
                iron_condor = IronCondor(sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p, max_profit, max_loss_p, max_loss_c, break_even_p, break_even_c, rr)
    
    # Print the statistics for the optimal Iron Condor
    print("Best Iron Condor:")
    print("Strike prices: ", iron_condor.lp_s, iron_condor.sp_s, iron_condor.sc_s, iron_condor.lc_s)
    print("Premiums: ", iron_condor.lp_p, iron_condor.sp_p, iron_condor.sc_p, iron_condor.lc_p)
    print("Max profit: ", iron_condor.max_profit)
    print("Max loss - put: ", iron_condor.max_loss_p)
    print("Max loss - call: ", iron_condor.max_loss_c)
    print("Break-even points: ", iron_condor.break_even_p, iron_condor.break_even_c)
    print("Risk-reward ratio: ", iron_condor.rr)
    print("Expected value: ", calculateExpectedValue(iron_condor.rr)*abs(min(iron_condor.max_loss_p, iron_condor.max_loss_c)))
    print("Expected value (as a ratio): ", calculateExpectedValue(iron_condor.rr))
    print("\n")
    ################################################################################################
    

    return

def main():
    data = importData('SPY')
    #print(data.head(5))
    data.iloc[:, 3].plot(figsize=(10,5))
    #print(data.iloc[:, 3].head(5))
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
    trials = 20000
    Z = norm.ppf(np.random.rand(days, trials))
    daily_returns = np.exp(drift + stdev * Z)

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1, 3]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]
    

    # Calculate the 95% confidence interval
    lower_bound = np.percentile(price_paths, 10)
    upper_bound = np.percentile(price_paths, 90)
    # Calculate the 99% confidence interval
    lower_bound_1 = np.percentile(price_paths, 2.5)
    upper_bound_1 = np.percentile(price_paths, 97.5)
    print("80% interval: ", lower_bound, upper_bound)
    print("95% interval: ", lower_bound_1, upper_bound_1)

    # # Plotting the price as a probability distribution
    # sb.displot(price_paths, bins=30, color='blue', legend=False)
    # mpl.axvline(x=lower_bound, color='g', linestyle='--')  # Add a vertical line at the lower bound
    # mpl.axvline(x=upper_bound, color='g', linestyle='--')  # Add a vertical line at the upper bound
    # mpl.axvline(x=lower_bound_1, color='r', linestyle='--')  # Add a vertical line at the lower bound
    # mpl.axvline(x=upper_bound_1, color='r', linestyle='--')  # Add a vertical line at the upper bound
    # mpl.xlabel('Price')
    # mpl.ylabel('Probability')
    # mpl.title('Price Distribution')
    # mpl.show()

    # # Download the actual future data
    # data_future = yf.download('SPY', '2024-05-10', '2024-06-01')
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

    # Run the Iron Condor model
    ironCondorModel(price_paths)
    

if __name__ == '__main__':
    main()