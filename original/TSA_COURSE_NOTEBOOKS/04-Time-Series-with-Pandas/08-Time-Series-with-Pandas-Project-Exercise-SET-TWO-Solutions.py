#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Time Series with Pandas Project Exercise
# 
# For this exercise, answer the questions below given the dataset: https://fred.stlouisfed.org/series/UMTMVS
# 
# This dataset is the Value of Manufacturers' Shipments for All Manufacturing Industries.

# **Import any necessary libraries.**

# In[42]:


# CODE HERE


# In[43]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read in the data UMTMVS.csv file from the Data folder**

# In[44]:


# CODE HERE


# In[45]:


df = pd.read_csv('../Data/UMTMVS.csv')


# **Check the head of the data**

# In[46]:


# CODE HERE


# In[47]:


df.head()


# **Set the DATE column as the index.**

# In[48]:


# CODE HERE


# In[49]:


df = df.set_index('DATE')


# In[50]:


df.head()


# **Check the data type of the index.**

# In[51]:


# CODE HERE


# In[52]:


df.index


# **Convert the index to be a datetime index. Note, there are many, many correct ways to do this!**

# In[53]:


# CODE HERE


# In[54]:


df.index = pd.to_datetime(df.index)


# In[55]:


df.index


# **Plot out the data, choose a reasonable figure size**

# In[56]:


# CODE HERE


# In[69]:


df.plot(figsize=(14,8))


# **What was the percent increase in value from Jan 2009 to Jan 2019?**

# In[71]:


#CODE HERE


# In[76]:


100 * (df.loc['2019-01-01'] - df.loc['2009-01-01']) / df.loc['2009-01-01']


# **What was the percent decrease from Jan 2008 to Jan 2009?**

# In[ ]:


#CODE HERE


# In[77]:


100 * (df.loc['2009-01-01'] - df.loc['2008-01-01']) / df.loc['2008-01-01']


# **What is the month with the least value after 2005?** [HINT](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.idxmin.html)

# In[59]:


#CODE HERE


# In[61]:


df.loc['2005-01-01':].idxmin()


# **What 6 months have the highest value?**

# In[68]:


# CODE HERE


# In[80]:


df.sort_values(by='UMTMVS',ascending=False).head(5)


# **How many millions of dollars in value was lost in 2008? (Another way of posing this question is what was the value difference between Jan 2008 and Jan 2009)**

# In[17]:


# CODE HERE


# In[18]:


df.loc['2008-01-01'] - df.loc['2009-01-01']


# **Create a bar plot showing the average value in millions of dollars per year**

# In[19]:


# CODE HERE


# In[20]:


df.resample('Y').mean().plot.bar(figsize=(15,8))


# **What year had the biggest increase in mean value from the previous year's mean value? (Lots of ways to get this answer!)**
# 
# [HINT for a useful method](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.idxmax.html)

# In[21]:


# CODE HERE


# In[22]:


yearly_data = df.resample('Y').mean()
yearly_data_shift = yearly_data.shift(1)


# In[23]:


yearly_data.head()


# In[24]:


change = yearly_data - yearly_data_shift 


# In[25]:


change['UMTMVS'].idxmax()


# **Plot out the yearly rolling mean on top of the original data. Recall that this is monthly data and there are 12 months in a year!**

# In[26]:


# CODE HERE


# In[78]:


df['Yearly Mean'] = df['UMTMVS'].rolling(window=12).mean()
df[['UMTMVS','Yearly Mean']].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# **BONUS QUESTION (HARD).**
# 
# **Some month in 2008 the value peaked for that year. How many months did it take to surpass that 2008 peak? (Since it crashed immediately after this peak) There are many ways to get this answer. NOTE: I get 70 months as my answer, you may get 69 or 68, depending on whether or not you count the start and end months. Refer to the video solutions for full explanation on this.**

# In[91]:


#CODE HERE


# In[97]:


df = pd.read_csv('../Data/UMTMVS.csv',index_col='DATE',parse_dates=True)


# In[98]:


df.head()


# In[99]:


df2008 = df.loc['2008-01-01':'2009-01-01']


# In[100]:


df2008.idxmax()


# In[101]:


df2008.max()


# In[105]:


df_post_peak = df.loc['2008-06-01':]


# In[106]:


df_post_peak[df_post_peak>=510081].dropna()


# In[108]:


len(df.loc['2008-06-01':'2014-03-01'])


# # GREAT JOB!
