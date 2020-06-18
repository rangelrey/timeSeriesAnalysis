#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from fbprophet import Prophet


# ### Load Data
# 
# The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.
# 

# In[ ]:




