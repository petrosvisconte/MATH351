## Author: Pierre Visconti
### MATH351 - Mathematical Modeling - Walla Walla University

**Disclaimer:** I am simply posting my work as a record for myself and as an academic resource for others. Do not blindly copy and attempt to submit as your own. 

### Final Project: ###
[Link to Presentation PDF](final_project/math351_presentation.pdf)

Geometric Brownian Motion (GBM) is a continuous-time stochastic process in which the logarithm of a randomly varying value follows a Brownian motion with drift.

Using Geometric Brownian Motion and historical volatily patterns, a stochastic simulation of possible price paths a given asset can take over a given time interval is performed. A Log-Normal distribution is compiled from the final prices at the end of the time interval which is used to model an optimal options strategy. Since the resulting distribution gives information on the probability of the asset being in a certain price interval over a given time period, an iron condor options strategy makes the most sense for this model. The optimal iron condor strategy for a given asset is determined by calculating the expected value of the trade based on potential profit/loss, and proability of winning the trade which comes from the compiled ending price distribution. This analysis is performed on a variety of assets that are deemed to be optimal for iron condor trades to determine the market-wide optimal iron condor trade. 
