#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Time Resampling
# 
# Let's learn how to sample time series data! This will be useful later on in the course!

# In[3]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import the data
# For this exercise we'll look at Starbucks stock data from 2015 to 2018 which includes daily closing prices and trading volumes.

# In[4]:


# Index_col indicates that the index will be the column called 'Date'
# parse_dates, transforms the strings into datetime format

df = pd.read_csv('../Data/starbucks.csv', index_col='Date', parse_dates=True)


# Note: the above code is a faster way of doing the following:
# <pre>df = pd.read_csv('../Data/starbucks.csv')
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date',inplace=True)</pre>

# In[5]:


df.head()


# ## resample()
# 
# A common operation with time series data is resampling based on the time series index. Let's see how to use the resample() method. [[reference](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)]

# In[6]:


# Our index
df.index


# When calling `.resample()` you first need to pass in a **rule** parameter, then you need to call some sort of aggregation function.
# 
# The **rule** parameter describes the frequency with which to apply the aggregation function (daily, monthly, yearly, etc.)<br>
# It is passed in using an "offset alias" - refer to the table below. [[reference](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)]
# 
# The aggregation function is needed because, due to resampling, we need some sort of mathematical rule to join the rows (mean, sum, count, etc.)

# <table style="display: inline-block">
#     <caption style="text-align: center"><strong>TIME SERIES OFFSET ALIASES</strong></caption>
# <tr><th>ALIAS</th><th>DESCRIPTION</th></tr>
# <tr><td>B</td><td>business day frequency</td></tr>
# <tr><td>C</td><td>custom business day frequency (experimental)</td></tr>
# <tr><td>D</td><td>calendar day frequency</td></tr>
# <tr><td>W</td><td>weekly frequency</td></tr>
# <tr><td>M</td><td>month end frequency</td></tr>
# <tr><td>SM</td><td>semi-month end frequency (15th and end of month)</td></tr>
# <tr><td>BM</td><td>business month end frequency</td></tr>
# <tr><td>CBM</td><td>custom business month end frequency</td></tr>
# <tr><td>MS</td><td>month start frequency</td></tr>
# <tr><td>SMS</td><td>semi-month start frequency (1st and 15th)</td></tr>
# <tr><td>BMS</td><td>business month start frequency</td></tr>
# <tr><td>CBMS</td><td>custom business month start frequency</td></tr>
# <tr><td>Q</td><td>quarter end frequency</td></tr>
# <tr><td></td><td><font color=white>intentionally left blank</font></td></tr></table>
# 
# <table style="display: inline-block; margin-left: 40px">
# <caption style="text-align: center"></caption>
# <tr><th>ALIAS</th><th>DESCRIPTION</th></tr>
# <tr><td>BQ</td><td>business quarter endfrequency</td></tr>
# <tr><td>QS</td><td>quarter start frequency</td></tr>
# <tr><td>BQS</td><td>business quarter start frequency</td></tr>
# <tr><td>A</td><td>year end frequency</td></tr>
# <tr><td>BA</td><td>business year end frequency</td></tr>
# <tr><td>AS</td><td>year start frequency</td></tr>
# <tr><td>BAS</td><td>business year start frequency</td></tr>
# <tr><td>BH</td><td>business hour frequency</td></tr>
# <tr><td>H</td><td>hourly frequency</td></tr>
# <tr><td>T, min</td><td>minutely frequency</td></tr>
# <tr><td>S</td><td>secondly frequency</td></tr>
# <tr><td>L, ms</td><td>milliseconds</td></tr>
# <tr><td>U, us</td><td>microseconds</td></tr>
# <tr><td>N</td><td>nanoseconds</td></tr></table>

# Let's resample our dataframe, by using rule "A", which is year and frecuency and aggregate it with the mean

# In[7]:


# Yearly Means
df.resample(rule='A').mean()


# Resampling rule 'A' takes all of the data points in a given year, applies the aggregation function (in this case we calculate the mean), and reports the result as the last day of that year.

# ### Custom Resampling Functions
# 
# We're not limited to pandas built-in summary functions (min/max/mean etc.). We can define our own function:

# In[8]:


def first_day(entry):
    """
    Returns the first instance of the period, regardless of sampling rate.
    """
    if len(entry):  # handles the case of missing data
        return entry[0]


# In[7]:


df.resample(rule='A').apply(first_day)


# ### Plotting

# Let's plot the average closing price per year

# In[8]:


df['Close'].resample('A').mean().plot.bar(title='Yearly Mean Closing Price for Starbucks');


# Pandas treats each sample as its own trace, and by default assigns different colors to each one. If you want, you can pass a <strong>color</strong> argument to assign your own color collection, or to set a uniform color. For example, <tt>color='#1f77b4'</tt> sets a uniform "steel blue" color.
# 
# Also, the above code can be broken into two lines for improved readability.

# In[9]:


title = 'Yearly Mean Closing Price for Starbucks'
df['Close'].resample('A').mean().plot.bar(title=title,color=['#1f77b4']);


# In[10]:


title = 'Monthly Max Closing Price for Starbucks'
df['Close'].resample('M').max().plot.bar(figsize=(16,6), title=title,color='#1f77b4');


# That is it! Up next we'll learn about time shifts!
