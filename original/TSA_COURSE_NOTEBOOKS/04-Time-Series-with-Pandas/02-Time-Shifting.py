#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Time Shifting
# 
# Sometimes you may need to shift all your data up or down along the time series index. In fact, a lot of pandas built-in methods do this under the hood. This isn't something we'll do often in the course, but it's definitely good to know about this anyways!

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../Data/starbucks.csv',index_col='Date',parse_dates=True)


# In[4]:


df.head()


# Let's check the last 5 rows of our dataframe

# In[4]:


df.tail()


# ## .shift() forward
# This method shifts the entire date index a given number of rows, without regard for time periods (months & years).<br>It returns a modified copy of the original DataFrame.
# 
# In other words, it moves down all the rows down or up.

# In[5]:


# We move down all the rows
.shift(1).head()


# In[6]:


# NOTE: You will lose that last piece of data that no longer has an index!
df.shift(1).tail()


# ## .shift() backwards

# In[7]:


df.shift(-1).head()


# In[8]:


df.shift(-1).tail()


# ## Shifting based on Time Series Frequency Code
# 
# We can choose to shift <em>index values</em> up or down without realigning the data by passing in a <strong>freq</strong> argument.<br>
# This method shifts dates to the next period based on a frequency code. Common codes are 'M' for month-end and 'A' for year-end. <br>Refer to the <em>Time Series Offset Aliases</em> table from the Time Resampling lecture for a full list of values, or click <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.<br>

# In[5]:


df.head()


# In[9]:


# Shift everything to the end of the month
df.shift(periods=1, freq='M').head()


# For more info on time shifting visit http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html<br>
# Up next we'll look at rolling and expanding!
