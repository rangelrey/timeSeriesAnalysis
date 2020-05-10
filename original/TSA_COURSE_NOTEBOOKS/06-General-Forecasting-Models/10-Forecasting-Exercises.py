#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Forecasting Exercises
# This exercise walks through a SARIMA prediction and forecast similar to the one done on the Mauna Loa CO₂ dataset.<br>
# This time we're using a seasonal time series of California Hospitality Industry Employees.
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>IMPORTANT NOTE!</strong> Make sure you don't run the cells directly above the example output shown, <br>otherwise you will end up writing over the example output!</div>

# In[1]:


# RUN THIS CELL
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load datasets
df = pd.read_csv('../Data/HospitalityEmployees.csv',index_col='Date',parse_dates=True)
df.index.freq = 'MS'
print(len(df))
print(df.head())


# So <tt>df</tt> has 348 records and one column. The data represents the number of employees in thousands of persons as monthly averages from January, 1990 to December 2018.

# ### 1. Plot the source data
# Create a line chart of the dataset. Optional: add a title and y-axis label.

# In[ ]:


## CODE HERE





# In[2]:


# DON'T WRITE HERE


# ### 2. Run an ETS Decomposition
# Use an 'additive' model.

# In[ ]:





# In[3]:


# DON'T WRITE HERE


# ### 3. Run <tt>pmdarima.auto_arima</tt> to obtain recommended orders
# This may take awhile as there are a lot of combinations to evaluate.

# In[ ]:





# In[4]:


# DON'T WRITE HERE


# You should see a recommended ARIMA Order of (0,1,0) combined with a seasonal order of (2,0,0,12).
# ### 4. Split the data into train/test sets
# Set one year (12 records) for testing. There is more than one way to do this!

# In[ ]:





# In[5]:


# DON'T WRITE HERE


# ### 5. Fit a SARIMA(0,1,0)(2,0,0,12) model to the training set

# In[ ]:





# In[6]:


# DON'T WRITE HERE


# ### 6. Obtain predicted values

# In[ ]:





# In[7]:


# DON'T WRITE HERE


# ### 7. Plot predictions against known values
# Optional: add a title and y-axis label.

# In[ ]:





# In[8]:


# DON'T WRITE HERE


# ### 8. Evaluate the Model using MSE and RMSE
# You can run both from the same cell if you want.

# In[ ]:





# In[9]:


# DON'T WRITE HERE


# ### 9. Retrain the model on the full data and forecast one year into the future

# In[ ]:





# In[10]:


# DON'T WRITE HERE


# ### 10. Plot the forecasted values alongside the original data
# Optional: add a title and y-axis label.

# In[ ]:





# In[11]:


# DON'T WRITE HERE


# ## Great job!
