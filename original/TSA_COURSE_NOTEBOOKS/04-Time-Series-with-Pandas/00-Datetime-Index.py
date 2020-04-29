#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Introduction to Time Series with Pandas
# 
# Most of our data will have a datatime index, so let's learn how to deal with this sort of data with pandas!

# ## Python Datetime Review
# In the course introduction section we discussed Python datetime objects.

# In[2]:


from datetime import datetime


# In[2]:


# To illustrate the order of arguments
my_year = 2017
my_month = 1
my_day = 2
my_hour = 13
my_minute = 30
my_second = 15


# In[3]:


# January 2nd, 2017
my_date = datetime(my_year,my_month,my_day)


# In[4]:


# Defaults to 0:00
my_date 


# In[5]:


# January 2nd, 2017 at 13:30:15
my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second)


# In[6]:


my_date_time


# You can grab any part of the datetime object you want

# In[7]:


my_date.day


# In[8]:


my_date_time.hour


# ## NumPy Datetime Arrays
# We mentioned that NumPy handles dates more efficiently than Python's datetime format.<br>
# The NumPy data type is called <em>datetime64</em> to distinguish it from Python's datetime.
# 
# In this section we'll show how to set up datetime arrays in NumPy. These will become useful later on in the course.<br>
# For more info on NumPy visit https://docs.scipy.org/doc/numpy-1.15.4/reference/arrays.datetime.html

# In[8]:


import numpy as np


# In[10]:


# CREATE AN ARRAY FROM THREE DATES
np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64')


# <div class="alert alert-info"><strong>NOTE:</strong> We see the dtype listed as <tt>'datetime64[D]'</tt>. This tells us that NumPy applied a day-level date precision.<br>
#     If we want we can pass in a different measurement, such as <TT>[h]</TT> for hour or <TT>[Y]</TT> for year.</div>

# In[11]:


np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[h]')


# In[12]:


np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[Y]')


# ## NumPy Date Ranges
# Just as <tt>np.arange(start,stop,step)</tt> can be used to produce an array of evenly-spaced integers, we can pass a <tt>dtype</tt> argument to obtain an array of dates. Remember that the stop date is <em>exclusive</em>.
# 
# Note that we indicate the unit of time in the dtype [D], [Y]...

# In[13]:


# AN ARRAY OF DATES FROM 6/1/18 TO 6/22/18 SPACED ONE WEEK APART
np.arange('2018-06-01', '2018-06-23', 7, dtype='datetime64[D]')


# By omitting the step value we can obtain every value based on the precision.

# In[14]:


# AN ARRAY OF DATES FOR EVERY YEAR FROM 1968 TO 1975
np.arange('1968', '1976', dtype='datetime64[Y]')


# ## Pandas Datetime Index
# 
# We'll usually deal with time series as a datetime index when working with pandas dataframes. Fortunately pandas has a lot of functions and methods to work with time series!<br>
# For more on the pandas DatetimeIndex visit https://pandas.pydata.org/pandas-docs/stable/timeseries.html

# In[4]:


import pandas as pd


# The simplest way to build a DatetimeIndex is with the <tt><strong>pd.date_range()</strong></tt> method:

# In[5]:


# THE WEEK OF JULY 8TH, 2018
idx = pd.date_range('7/8/2018', periods=7, freq='D')
idx


# <div class="alert alert-info"><strong>DatetimeIndex Frequencies:</strong> When we used <tt>pd.date_range()</tt> above, we had to pass in a frequency parameter <tt>'D'</tt>. This created a series of 7 dates spaced one day apart. We'll cover this topic in depth in upcoming lectures, but for now, a list of time series offset aliases like <tt>'D'</tt> can be found <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.</div>

# Another way is to convert incoming text with the <tt><strong>pd.to_datetime()</strong></tt> method:

# In[6]:


idx = pd.to_datetime(['Jan 01, 2018','1/2/18','03-Jan-2018',None])
idx


# A third way is to pass a list or an array of datetime objects into the <tt><strong>pd.DatetimeIndex()</strong></tt> method:

# In[9]:


# Create a NumPy datetime array
some_dates = np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[D]')
some_dates


# In[11]:


pd.to_datetime(['2/1/2018','3/1/2018'],format='%d/%m/%Y')


# You can transform from any type of date notation that you need

# In[12]:


pd.to_datetime(['2_1_2018','3_1_2018'],format='%d_%m_%Y')


# In[19]:


# Convert to an index
idx = pd.DatetimeIndex(some_dates)
idx


# Notice that even though the dates came into pandas with a day-level precision, pandas assigns a nanosecond-level precision with the expectation that we might want this later on.
# 
# To set an existing column as the index, use <tt>.set_index()</tt><br>
# ><tt>df.set_index('Date',inplace=True)</tt>

# ## Pandas Datetime Analysis

# In[20]:


# Create some random data
data = np.random.randn(3,2)
cols = ['A','B']
print(data)


# In[21]:


# Create a DataFrame with our random data, our date index, and our columns
df = pd.DataFrame(data,idx,cols)
df


# Now we can perform a typical analysis of our DataFrame

# In[22]:


df.index


# In[23]:


# Latest Date Value
df.index.max()


# In[24]:


# Latest Date Index Location
df.index.argmax()


# In[25]:


# Earliest Date Value
df.index.min()


# In[26]:


# Earliest Date Index Location
df.index.argmin()


# <div class="alert alert-info"><strong>NOTE:</strong> Normally we would find index locations by running <tt>.idxmin()</tt> or <tt>.idxmax()</tt> on <tt>df['column']</tt> since <tt>.argmin()</tt> and <tt>.argmax()</tt> have been deprecated. However, we still use <tt>.argmin()</tt> and <tt>.argmax()</tt> on the index itself.</div>

# ## Great, let's move on!
