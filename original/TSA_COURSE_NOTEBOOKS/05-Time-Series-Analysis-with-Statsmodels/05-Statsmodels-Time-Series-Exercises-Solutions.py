#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Statsmodels Time Series Excercises - Solutions
# For this set of exercises we're using data from the Federal Reserve Economic Database (FRED) concerning the Industrial Production Index for Electricity and Gas Utilities from January 1970 to December 1989.
# 
# Data source: https://fred.stlouisfed.org/series/IPG2211A2N
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>IMPORTANT NOTE!</strong> Make sure you don't run the cells directly above the example output shown, <br>otherwise you will end up writing over the example output!</div>

# In[2]:


# RUN THIS CELL
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../Data/EnergyProduction.csv',index_col=0,parse_dates=True)
df.head()


# ### 1. Assign a frequency of 'MS' to the DatetimeIndex.

# In[ ]:


# CODE HERE


# In[3]:


# DON'T WRITE HERE
df.index.freq = 'MS'
df.index


# ### 2. Plot the dataset.

# In[4]:


# CODE HERE


# In[5]:


# DON'T WRITE HERE
df.plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# ### 3. Add a column that shows a 12-month Simple Moving Average (SMA).<br>&nbsp;&nbsp;&nbsp;&nbsp;Plot the result.

# In[ ]:





# In[6]:


# DON'T WRITE HERE
df['SMA12'] = df['EnergyIndex'].rolling(window=12).mean()
df.plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# ### 4. Add a column that shows an Exponentially Weighted Moving Average (EWMA) with a span of 12 using the statsmodels <tt>SimpleExpSmoothing</tt> function. Plot the result.

# In[7]:


# DON'T FORGET TO PERFORM THE IMPORT!
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

df['SES12']=SimpleExpSmoothing(df['EnergyIndex']).fit(smoothing_level=2/(12+1),optimized=False).fittedvalues.shift(-1)
df.plot(figsize=(12,6)).autoscale(axis='x',tight=True);



# In[5]:


# DON'T WRITE HERE


# ### 5. Add a column to the DataFrame that shows a Holt-Winters fitted model using Triple Exponential Smoothing with multiplicative models. Plot the result.

# In[ ]:


# DON'T FORGET TO PERFORM THE IMPORT!



# In[6]:


# DON'T WRITE HERE
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df['TESmul12'] = ExponentialSmoothing(df['EnergyIndex'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues.shift(-1)
df.plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# ### OPTIONAL: Plot the same  as above, but for only the first two years.

# In[ ]:





# In[7]:


# DON'T WRITE HERE
df.iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# ### BONUS QUESTION: There is a visible decline in the Industrial Production Index around 1982-1983.<br>Why do you think this might be?

# In[8]:


# The United States suffered a significant economic recession at that time.


# ## Great job!
