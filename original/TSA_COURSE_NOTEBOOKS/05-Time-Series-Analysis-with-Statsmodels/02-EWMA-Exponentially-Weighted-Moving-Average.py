#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # MA
# ## Moving Averages
# In this section we'll compare <em>Simple Moving Averages</em> to <em>Exponentially Weighted Moving Averages</em> in terms of complexity and performance.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong><a href='https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html'>pandas.DataFrame.rolling</a></strong><font color=black>(window)</font>&nbsp;&nbsp;
# Provides rolling window calculations<br>
# <strong><a href='https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html'>pandas.DataFrame.ewm</a></strong><font color=black>(span)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# Provides exponential weighted functions</tt></div></div>
# 
# ### Perform standard imports and load the dataset
# For these examples we'll use the International Airline Passengers dataset, which gives monthly totals in thousands from January 1949 to December 1960.

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


airline = pd.read_csv('../Data/airline_passengers.csv',index_col='Month',parse_dates=True)


# In[3]:


airline.dropna(inplace=True)


# In[4]:


airline.head()


# ___
# # SMA
# ## Simple Moving Average
# 
# We've already shown how to create a <a href='https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average'>simple moving average</a> by applying a <tt>mean</tt> function to a rolling window.
# 
# For a quick review:

# In[5]:


airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()


# In[6]:


airline.head(15)


# In[7]:


airline.plot();


# ___
# # EWMA
# ## Exponentially Weighted Moving Average 
# 
# We just showed how to calculate the SMA based on some window. However, basic SMA has some weaknesses:
# * Smaller windows will lead to more noise, rather than signal
# * It will always lag by the size of the window
# * It will never reach to full peak or valley of the data due to the averaging.
# * Does not really inform you about possible future behavior, all it really does is describe trends in your data.
# * Extreme historical values can skew your SMA significantly
# 
# To help fix some of these issues, we can use an <a href='https://en.wikipedia.org/wiki/Exponential_smoothing'>EWMA (Exponentially weighted moving average)</a>.

# EWMA will allow us to reduce the lag effect from SMA and it will put more weight on values that occured more recently (by applying more weight to the more recent values, thus the name). The amount of weight applied to the most recent values will depend on the actual parameters used in the EWMA and the number of periods given a window size.
# [Full details on Mathematics behind this can be found here](http://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows).
# Here is the shorter version of the explanation behind EWMA.
# 
# The formula for EWMA is:
# ### $y_t =   \frac{\sum\limits_{i=0}^t w_i x_{t-i}}{\sum\limits_{i=0}^t w_i}$

# Where $x_t$ is the input value, $w_i$ is the applied weight (Note how it can change from $i=0$ to $t$), and $y_t$ is the output.
# 
# Now the question is, how to we define the weight term $w_i$?
# 
# This depends on the <tt>adjust</tt> parameter you provide to the <tt>.ewm()</tt> method.
# 
# When <tt>adjust=True</tt> (default) is used, weighted averages are calculated using weights equal to $w_i = (1 - \alpha)^i$
# 
# which gives
# 
# ### $y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
# + (1 - \alpha)^t x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
# + (1 - \alpha)^t}$

# When <tt>adjust=False</tt> is specified, moving averages are calculated as:
# 
# ### $\begin{split}y_0 &= x_0 \\
# y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,\end{split}$
# 
# which is equivalent to using weights:
# 
#  \begin{split}w_i = \begin{cases}
#     \alpha (1 - \alpha)^i & \text{if } i < t \\
#     (1 - \alpha)^i        & \text{if } i = t.
# \end{cases}\end{split}

# When <tt>adjust=True</tt> we have $y_0=x_0$ and from the last representation above we have 
# $y_t=\alpha x_t+(1−α)y_{t−1}$, therefore there is an assumption that $x_0$ is not an ordinary value but rather an exponentially weighted moment of the infinite series up to that point.
# 
# For the smoothing factor $\alpha$ one must have $0<\alpha≤1$, and while it is possible to pass <em>alpha</em> directly, it’s often easier to think about either the <em>span</em>, <em>center of mass</em> (com) or <em>half-life</em> of an EW moment:

# \begin{split}\alpha =
#  \begin{cases}
#      \frac{2}{s + 1},               & \text{for span}\ s \geq 1\\
#      \frac{1}{1 + c},               & \text{for center of mass}\ c \geq 0\\
#      1 - \exp^{\frac{\log 0.5}{h}}, & \text{for half-life}\ h > 0
#  \end{cases}\end{split}

# * <strong>Span</strong> corresponds to what is commonly called an “N-day EW moving average”.
# * <strong>Center of mass</strong> has a more physical interpretation and can be thought of in terms of span: $c=(s−1)/2$
# * <strong>Half-life</strong> is the period of time for the exponential weight to reduce to one half.
# * <strong>Alpha</strong> specifies the smoothing factor directly.
# 
# We have to pass precisely one of the above into the <tt>.ewm()</tt> function. For our data we'll use <tt>span=12</tt>.

# In[8]:


airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()


# In[9]:


airline[['Thousands of Passengers','EWMA12']].plot();


# ## Comparing SMA to EWMA

# In[10]:


airline[['Thousands of Passengers','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);


# ## Simple Exponential Smoothing
# The above example employed <em>Simple Exponential Smoothing</em> with one smoothing factor <strong>α</strong>. Unfortunately, this technique does a poor job of forecasting when there is a trend in the data as seen above. In the next section we'll look at <em>Double</em> and <em>Triple Exponential Smoothing</em> with the Holt-Winters Methods.
