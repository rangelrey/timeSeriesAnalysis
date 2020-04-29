#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Operations
# 
# There are lots of operations with pandas that will be really useful to you, but don't fall into any distinct category. Let's show them here in this lecture:

# In[52]:


import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()


# ### Info on Unique Values

# In[53]:


df['col2'].unique()


# In[54]:


df['col2'].nunique()


# In[55]:


df['col2'].value_counts()


# ### Selecting Data

# In[56]:


#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]


# In[57]:


newdf


# ### Applying Functions

# In[58]:


def times2(x):
    return x*2


# In[59]:


df['col1'].apply(times2)


# In[60]:


df['col3'].apply(len)


# In[61]:


df['col1'].sum()


# ### Permanently Removing a Column

# In[62]:


del df['col1']


# In[63]:


df


# ### Get column and index names:

# In[64]:


df.columns


# In[65]:


df.index


# ### Sorting and Ordering a DataFrame:

# In[66]:


df


# In[67]:


df.sort_values(by='col2') #inplace=False by default


# # Great Job!
