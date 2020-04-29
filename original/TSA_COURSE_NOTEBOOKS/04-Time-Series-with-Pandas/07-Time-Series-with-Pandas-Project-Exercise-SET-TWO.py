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

# In[1]:


# CODE HERE
import pandas as pd
import numpy as np
import matplotlib


# In[43]:





# **Read in the data UMTMVS.csv file from the Data folder**

# In[44]:


# CODE HERE


# In[6]:


df = pd.read_csv("../Data/UMTMVS.csv")


# **Check the head of the data**

# In[46]:


# CODE HERE


# In[47]:





# **Set the DATE column as the index.**

# In[48]:


# CODE HERE


# In[12]:


df["DATE"] = pd.to_datetime(df["DATE"])


# In[14]:


df.set_index('DATE', inplace=True)


# In[15]:





# **Check the data type of the index.**

# In[51]:


# CODE HERE


# In[52]:





# **Convert the index to be a datetime index. Note, there are many, many correct ways to do this!**

# In[53]:


# CODE HERE


# In[54]:





# In[55]:





# **Plot out the data, choose a reasonable figure size**

# In[56]:


# CODE HERE


# In[19]:


df["UMTMVS"].plot(figsize=(12,5))


# **What was the percent increase in value from Jan 2009 to Jan 2019?**

# In[23]:


#CODE HERE

df.loc['2019-01-01'] / df.loc['2009-01-01'] -1


# In[76]:





# **What was the percent decrease from Jan 2008 to Jan 2009?**

# In[ ]:


#CODE HERE


# In[77]:





# **What is the month with the least value after 2005?** [HINT](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.idxmin.html)

# In[37]:


#CODE HERE

df.loc['2005-01-01':].idxmin()


# In[61]:





# **What 6 months have the highest value?**

# In[39]:


# CODE HERE
df.sort_values("UMTMVS",ascending=False)


# In[80]:





# **How many millions of dollars in value was lost in 2008? (Another way of posing this question is what was the value difference between Jan 2008 and Jan 2009)**

# In[17]:


# CODE HERE


# In[18]:





# **Create a bar plot showing the average value in millions of dollars per year**

# In[20]:





# **What year had the biggest increase in mean value from the previous year's mean value? (Lots of ways to get this answer!)**
# 
# [HINT for a useful method](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.idxmax.html)

# In[21]:


# CODE HERE


# In[45]:


df.resample('Y').mean()


# In[23]:





# In[24]:





# In[25]:





# **Plot out the yearly rolling mean on top of the original data. Recall that this is monthly data and there are 12 months in a year!**

# In[26]:


# CODE HERE


# In[49]:


df["yearly mean"] = df["UMTMVS"].rolling(window=12).mean()


# In[51]:


df["yearly mean"].plot()


# **BONUS QUESTION (HARD).**
# 
# **Some month in 2008 the value peaked for that year. How many months did it take to surpass that 2008 peak? (Since it crashed immediately after this peak) There are many ways to get this answer. NOTE: I get 70 months as my answer, you may get 69 or 68, depending on whether or not you count the start and end months. Refer to the video solutions for full explanation on this.**

# In[91]:


#CODE HERE


# In[108]:





# # GREAT JOB!
