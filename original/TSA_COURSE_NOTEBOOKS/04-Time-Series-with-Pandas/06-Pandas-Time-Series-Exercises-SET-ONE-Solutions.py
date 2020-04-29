#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Pandas Time Series Exercise Set #1 - Solution
# 
# For this set of exercises we'll use a dataset containing monthly milk production values in pounds per cow from January 1962 to December 1975.
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>IMPORTANT NOTE!</strong> Make sure you don't run the cells directly above the example output shown, <br>otherwise you will end up writing over the example output!</div>

# In[16]:


# RUN THIS CELL
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../Data/monthly_milk_production.csv', encoding='utf8')
title = "Monthly milk production: pounds per cow. Jan '62 - Dec '75"

print(len(df))
print(df.head())


# So <tt>df</tt> has 168 records and 2 columns.

# ### 1. What is the current data type of the Date column?
# HINT: We show how to list column dtypes in the first set of DataFrame lectures.

# In[ ]:


# CODE HERE


# In[17]:


# DON'T WRITE HERE
df.dtypes


# ### 2. Change the Date column to a datetime format

# In[ ]:





# In[18]:


# DON'T WRITE HERE
df['Date']=pd.to_datetime(df['Date'])
df.dtypes


# ### 3. Set the Date column to be the new index

# In[ ]:





# In[19]:


# DON'T WRITE HERE
df.set_index('Date',inplace=True)
df.head()


# ### 4. Plot the DataFrame with a simple line plot. What do you notice about the plot?

# In[ ]:





# In[20]:


# DON'T WRITE HERE
df.plot();

# THE PLOT SHOWS CONSISTENT SEASONALITY, AS WELL AS AN UPWARD TREND


# ### 5. Add a column called 'Month' that takes the month value from the index
# HINT: You have to call <tt>df.index</tt> as <tt>df['Date']</tt> won't work.
# 
# <strong>BONUS: See if you can obtain the <em>name</em> of the month instead of a number!</strong>

# In[ ]:





# In[28]:


# DON'T WRITE HERE
df['Month']=df.index.month
df.head()


# In[22]:


# BONUS SOLUTION:
df['Month']=df.index.strftime('%B')
df.head()


# ### 6. Create a BoxPlot that groups by the Month field

# In[ ]:





# In[29]:


# DON'T WRITE HERE
df.boxplot(by='Month',figsize=(12,5));


# # Great Job!
