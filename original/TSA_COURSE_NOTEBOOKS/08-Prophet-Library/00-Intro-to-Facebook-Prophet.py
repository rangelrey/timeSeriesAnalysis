#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Quick Guide to Facebook's Prophet Basics
# ---
# ---
# 
# ## IMPORTANT NOTE ONE:
# 
# **You should really read the papaer for Prophet! It is relatively straightforward and has a lot of insight on their techniques on how Prophet works internally!**
# 
# Link to paper: https://peerj.com/preprints/3190.pdf
# ---
# ---
# 
# ## IMPORTANT NOTE TWO:
# 
# -----
# ------
# 
# * **NOTE: Link to installation instructions:** 
#     * https://facebook.github.io/prophet/docs/installation.html#python 
#     * SCROLL DOWN UNTIL YOU SEE THE ANACONDA OPTION AT THE BOTTOM OF THE PAGE.
#     * YOU MAY NEED TO INSTALL BOTH **conda install gcc** and **conda install -c conda-forge fbprophet**
#     * PLEASE READ THROUGH THE DOCS AND STACKOERFLOW CAREFULLY BEFORE POSTING INSTALLATION ISSUES TO THE QA FORUMS.
# 
# -----
# ----

# ## Load Libraries

# In[1]:


import pandas as pd
from fbprophet import Prophet


# ## Load Data
# 
# The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

# In[2]:


df = pd.read_csv('../Data/BeerWineLiquor.csv')


# In[3]:


df.head()


# ### Format the Data

# In[4]:


df.columns = ['ds','y']


# In[5]:


df['ds'] = pd.to_datetime(df['ds'])


# ## Create and Fit Model

# In[6]:


# This is fitting on all the data (no train test split in this example)
m = Prophet()
m.fit(df)


# ## Forecasting
# 
# ### Step 1: Create "future" placeholder dataframe
# 
# **NOTE: Prophet by default is for daily data. You need to pass a frequency for sub-daily or monthly data. Info: https://facebook.github.io/prophet/docs/non-daily_data.html**

# In[7]:


future = m.make_future_dataframe(periods=24,freq = 'MS')


# In[8]:


df.tail()


# In[9]:


future.tail()


# In[10]:


len(df)


# In[11]:


len(future)


# 
# ### Step 2: Predict and fill in the Future

# In[12]:


forecast = m.predict(future)


# In[13]:


forecast.head()


# In[14]:


forecast.tail()


# In[15]:


forecast.columns


# In[16]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)


# ### Plotting Forecast
# 
# We can use Prophet's own built in plotting tools

# In[17]:


m.plot(forecast);


# In[18]:


import matplotlib.pyplot as plt
m.plot(forecast)
plt.xlim('2014-01-01','2022-01-01')


# In[19]:


forecast.plot(x='ds',y='yhat')


# In[20]:


m.plot_components(forecast);


# ## Great Job!

# In[ ]:




