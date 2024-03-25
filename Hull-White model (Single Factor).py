#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In financial mathematics, the Hull–White model is a model of future interest rates.
# In its most generic formulation, it belongs to the class of no-arbitrage models 
# that can fit today's term structure of interest rates.
# The "one factor means" refers to the single stochastic factor that drives the dynamics of short-term interest rate
# The stochastic process is normally modeled as a Wiener process of Brownian motion
# The single factor captures the uncertainty and volatility of interest rate movements over time


# In[63]:


pip install QuantLib


# In[64]:


# imports the necessary libraries, sets up parameters for the Hull-White model, 
# such as volatility (sigma), mean reversion (a), number of time steps (timestep), 
# total length of the simulation (length), the initial forward rate (forward_rate), 
# day-count convention (day_count), and the evaluation date (todays_date)


# In[65]:


import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
sigma = 0.1
a = 0.1
timestep = 360
length = 30 # in years
forward_rate = 0.05
day_count = ql.Thirty360(ql.Thirty360.ISDA)
todays_date = ql.Date(1, 3, 2024)


# In[66]:


# Set up the Quant Lib with the evaluation date. Everything starts with “evaluation date” which means the date you want to value an instrument. 
ql.Settings.instance().setEvaluationDate(todays_date)


# In[67]:


# creates a flat forward curve (spot_curve) and sets up the Hull-White process 
spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)


# In[68]:


# (hw_process) using the specified parameters
# ql.HullWhiteProcess(riskFreeTS, a, sigma)
# The 3 parameters of the model are: 
# 1) mean reverting level (assumed to be the Spot Curve)
# 2) Rate of mean reversion (a)
# 3) Volatility (sigma)
hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)


# In[69]:


# Random paths generation
rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)
def generate_paths(num_paths, timestep):
    arr = np.zeros((num_paths, timestep+1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr
num_paths = 5
time, paths = generate_paths(num_paths, timestep)
for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
plt.title("Hull-White Short Rate Simulation")
plt.show()


# In[70]:


# generates and visualizes the variance of short rates over time. 
# It compares the simulated variance (in red) with the analytical solution 
# for the variance under the Hull-White model (in blue).

num_paths = 1000
time, paths = generate_paths(num_paths, timestep)
vol = [np.var(paths[:, i]) for i in range(timestep+1)]
plt.plot(time, vol, "r-.", lw=3, alpha=0.6)
plt.plot(time,sigma*sigma/(2*a)*(1.0-np.exp(-2.0*a*np.array(time))), "b-", lw=2, alpha=0.5)
plt.title("Variance of Short Rates")
plt.show()

def alpha(forward, sigma, a, t):
    return forward + 0.5* np.power(sigma/a*(1.0 - np.exp(-a*t)), 2)

avg = [np.mean(paths[:, i]) for i in range(timestep+1)]
plt.plot(time, avg, "r-.", lw=3, alpha=0.6)
plt.plot(time,alpha(forward_rate, sigma, a, time), "b-", lw=2, alpha=0.6)
plt.title("Mean of Short Rates")
plt.show()


# In[ ]:




