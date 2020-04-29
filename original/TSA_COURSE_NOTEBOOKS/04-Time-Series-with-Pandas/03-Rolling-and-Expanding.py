#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Rolling and Expanding
# 
# A common process with time series is to create data based off of a rolling mean. The idea is to divide the data into "windows" of time, and then calculate an aggregate function for each window. In this way we obtain a <em>simple moving average</em>. Let's show how to do this easily with pandas!

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import the data:
df = pd.read_csv('../Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[3]:


df.head()


# In[4]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# Now let's add in a rolling mean! This rolling method provides row entries, where every entry is then representative of the window. 

# In[5]:


# 7 day rolling mean
df.rolling(window=7).mean().head(15)


# In[6]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)
df.rolling(window=30).mean()['Close'].plot();


# The easiest way to add a legend is to make the rolling value a new column, then pandas does it automatically!

# In[7]:


df['Close: 30 Day Mean'] = df['Close'].rolling(window=30).mean()
df[['Close','Close: 30 Day Mean']].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# ## Expanding
# 
# Instead of calculating values for a rolling window of dates, what if you wanted to take into account everything from the start of the time series up to each point in time? For example, instead of considering the average over the last 7 days, we would consider all prior data in our expanding set of averages.

# In[10]:


# df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)

# Optional: specify a minimum number of periods to start from
df['Close'].expanding(min_periods=30).mean().plot(figsize=(12,5));


# That's it! It doesn't help much to visualize an expanding operation against the daily data, since all it really gives us is a picture of the "stability" or "volatility" of a stock. However, if you do want to see it, simply uncomment the first plot line above and rerun the cell.
# 
# Next up, we'll take a deep dive into visualizing time series data!
