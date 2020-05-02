#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Introduction to Forecasting
# In the previous section we fit various smoothing models to existing data. The purpose behind this is to predict what happens next.<br>
# What's our best guess for next month's value? For the next six months?
# 
# In this section we'll look to extend our models into the future. First we'll divide known data into training and testing sets, and evaluate the performance of a trained model on known test data.
# 
# * Goals
#   * Compare a Holt-Winters forecasted model to known data
#   * Understand <em>stationarity</em>, <em>differencing</em> and <em>lagging</em>
#   * Introduce ARIMA and describe next steps

# ### <font color=blue>Simple Exponential Smoothing / Simple Moving Average</font>
# This is the simplest to forecast. $\hat{y}$ is equal to the most recent value in the dataset, and the forecast plot is simply a horizontal line extending from the most recent value.
# ### <font color=blue>Double Exponential Smoothing / Holt's Method</font>
# This model takes trend into account. Here the forecast plot is still a straight line extending from the most recent value, but it has slope.
# ### <font color=blue>Triple Exponential Smoothing / Holt-Winters Method</font>
# This model has (so far) the "best" looking forecast plot, as it takes seasonality into account. When we expect regular fluctuations in the future, this model attempts to map the seasonal behavior.

# ## Forecasting with the Holt-Winters Method
# For this example we'll use the same airline_passengers dataset, and we'll split the data into 108 training records and 36 testing records. Then we'll evaluate the performance of the model.

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()


# In[4]:


df.tail()


# In[6]:


df.info()


# ## Train Test Split

# In[9]:


train_data = df.iloc[:109] # Goes up to but not including 109
test_data = df.iloc[108:]


# ## Fitting the Model

# In[10]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# ## Evaluating Model against Test Set

# In[13]:


# YOU CAN SAFELY IGNORE WARNINGS HERE!
# THIS WILL NOT EFFECT YOUR FORECAST, ITS JUST SOMETHING STATSMODELS NEEDS TO UPDATE UPON NEXT RELEASE.
test_predictions = fitted_model.forecast(36).rename('HW Forecast')


# In[15]:


test_predictions


# In[24]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))


# In[32]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')


# In[34]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION',xlim=['1958-01-01','1961-01-01'])


# ## Evaluation Metrics

# In[40]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[41]:


mean_absolute_error(test_data,test_predictions)


# In[42]:


mean_squared_error(test_data,test_predictions)


# In[37]:


np.sqrt(mean_squared_error(test_data,test_predictions))


# In[45]:


test_data.describe()


# ## Forecasting into Future

# In[46]:


final_model = ExponentialSmoothing(df['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# In[47]:


forecast_predictions = final_model.forecast(36)


# In[50]:


df['Thousands of Passengers'].plot(figsize=(12,8))
forecast_predictions.plot()


# ## Stationarity
# Time series data is said to be <em>stationary</em> if it does <em>not</em> exhibit trends or seasonality. That is, fluctuations in the data are entirely due to outside forces and noise. The file <tt>samples.csv</tt> contains made-up datasets that illustrate stationary and non-stationary data.

# In[4]:


df2 = pd.read_csv('../Data/samples.csv',index_col=0,parse_dates=True)
df2.head()


# In[5]:


df2['a'].plot(ylim=[0,100],title="STATIONARY DATA").autoscale(axis='x',tight=True);


# In[6]:


df2['b'].plot(ylim=[0,100],title="NON-STATIONARY DATA").autoscale(axis='x',tight=True);


# In an upcoming section we'll learn how to test for stationarity.

# ## Differencing
# Non-stationary data can be made to look stationary through <em>differencing</em>. A simple differencing method calculates the difference between consecutive points.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.diff.html'>statespace.tools.diff</a></strong><font color=black>(series[, k_diff, …])</font>&nbsp;&nbsp;Difference a series simply and/or seasonally along the zero-th axis.</tt></div>

# In[7]:


from statsmodels.tsa.statespace.tools import diff
df2['d1'] = diff(df2['b'],k_diff=1)

df2['d1'].plot(title="FIRST DIFFERENCE DATA").autoscale(axis='x',tight=True);


# <div class="alert alert-info"><strong>NOTE:</strong> the same operation can be accomplished using pandas alone:<br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>df2['d1'] = df2['b'] - df2['b'].shift(1)</tt></font></div>

# ## Lagging
# Also known as "backshifting", lagging notation reflects the value of $y$ at a prior point in time. This is a useful technique for performing <em>regressions</em> as we'll see in upcoming sections.
# 
# \begin{split}L{y_t} = y_{t-1} & \text{      one lag shifts the data back one period}\\
# L^{2}{y_t} = y_{t-2} & \text{      two lags shift the data back two periods} \end{split}
# <br><br>
# <table>
# <tr><td>$y_t$</td><td>6</td><td>8</td><td>3</td><td>4</td><td>9</td><td>2</td><td>5</td></tr>
# <tr><td>$y_{t-1}$</td><td>8</td><td>3</td><td>4</td><td>9</td><td>2</td><td>5</td></tr>
# <tr><td>$y_{t-2}$</td><td>3</td><td>4</td><td>9</td><td>2</td><td>5</td></tr>
# </table>
# 

# # Introduction to ARIMA Models
# We'll investigate a variety of different forecasting models in upcoming sections, but they all stem from ARIMA.
# 
# <strong>ARIMA</strong>, or <em>Autoregressive Integrated Moving Average</em> is actually a combination of 3 models:
# * <strong>AR(p)</strong> Autoregression - a regression model that utilizes the dependent relationship between a current observation and observations over a previous period
# * <strong>I(d)</strong> Integration - uses differencing of observations (subtracting an observation from an observation at the previous time step) in order to make the time series stationary
# * <strong>MA(q)</strong> Moving Average - a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# 
# <strong>Moving Averages</strong> we've already seen with EWMA and the Holt-Winters Method.<br>
# <strong>Integration</strong> will apply differencing to make a time series stationary, which ARIMA requires.<br>
# <strong>Autoregression</strong> is explained in detail in the next section. Here we're going to correlate a current time series with a lagged version of the same series.<br>
# Once we understand the components, we'll investigate how to best choose the $p$, $d$ and $q$ values required by the model.

# ### Great, let's get started!
