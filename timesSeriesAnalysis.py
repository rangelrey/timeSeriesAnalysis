#!/usr/bin/env python
# coding: utf-8

# # Numpy Reminders: 
# Let's have a look at some of the most basic numpy concepts that we will use in the more advanced sections

# In[2]:


import numpy as np
import datetime as datetime
from matplotlib import dates


# ## Numpy Creating data
# In order to test our functions, being able to create "random" data is the key

# In[3]:


#Create a range based on your input. From 0 to 10, not including 10 and step size parameter 2
np.arange(0,10,2)


# In[4]:


#Return evenly space 5 floats from 0 to 10, including 10
np.linspace(0,10,5)


# In[5]:


#return 2 radom floats from a uniform distribution (all numbers have the same probability)
np.random.rand(2)


# In[6]:


#return 2 radom floats from a normal distribution with mean = 0 and std = 1
np.random.randn(2)


# In[7]:


#return 2 radom floats from a normal distribution with mean = 3 and std = 1
np.random.normal(3,1,2)


# In[8]:


#Generate the same random numbers by setting a seed
#The number in the seed is irrelevant
#It will only work if we are on the same cell
np.random.seed(1)
print(np.random.rand(1))

np.random.seed(2)
print(np.random.rand(1))

np.random.seed(1)
print(np.random.rand(1))


# In[9]:


#Creating a matrix from an array
#First create an array
arr = np.arange(4)

#Then reshape it
matrix = arr.reshape(2,2)


# In[10]:


#return the min and max values of an array/matrix

print(arr.max())
print(matrix.min())


# ## Numpy Indexing and Selection
# 

# In[11]:


#Remember if you want to work with array copies use:
arr_copy = arr.copy()

#Otherwise you will be always editing the original array as well,
#since the new object created is pointing to the original object


# In[12]:


#Create a matrix
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix)
#Return the item in the column 1 and row 1
matrix[1][1]


# In[13]:


#Return (from the matrix) until the 2 row (not included)
matrix[:2]


# In[14]:


#Until row 2 and from column 1
matrix[:2,1:]


# In[15]:


#Return a filtered array whose values are lower than 2
print("Original array: ")
print(arr)
print("Result: ") 
print(arr[arr<2])


# ## Numpy Operations
# Skipping the basic ones

# In[16]:


#Sum of all the values of the columns
print(matrix)
matrix.sum(axis=0)


# In[17]:


#Sum of all the values of the rows
print(matrix)
matrix.sum(axis=1)


# # Pandas Reminders

# In[18]:


import pandas as pd


# In[19]:


#Create a Matrix, which will be used for the dataframe creation
rand_mat = np.random.rand(5,4)


# In[20]:


#Create dataframe
df = pd.DataFrame(data=rand_mat, index = 'A B C D E'.split(), columns = "R P U I".split())


# In[21]:


#Drop row
df.drop("A")


# In[22]:


#Drop column
df.drop("R",axis=1)


# In[23]:


#Return series of the row A
df.loc["A"]


# In[24]:


#Return series of the row number 2
df.iloc[2]


# In[25]:


#Filtering by value. Filter all rows that are smaller than 0.3 in column I
df[df["I"]>0.3]


# In[26]:


#Return a unique array of the column R
df["R"].unique()


# In[27]:


#Return the number of unique items of the array of the column R
df["R"].nunique()


# In[28]:


#Apply a function to a column

df["R"].apply(lambda a: a+1)


# ## Pandas Viz Reminders

# In[29]:


#Display plots directly in jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


#Import data
df1 = pd.read_csv("./original/TSA_COURSE_NOTEBOOKS/03-Pandas-Visualization/df1.csv",index_col=0)
df2 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/03-Pandas-Visualization/df2.csv')


# ### Plot Types
# 
# There are several plot types built into pandas; most of them are statistical by nature:
# 
# <pre>
# df.plot.hist()     histogram
# df.plot.bar()      bar chart
# df.plot.barh()     horizontal bar chart
# df.plot.line()     line chart
# df.plot.area()     area chart
# df.plot.scatter()  scatter plot
# df.plot.box()      box plot
# df.plot.kde()      kde plot
# df.plot.hexbin()   hexagonal bin plot
# df.plot.pie()      pie chart</pre>
# 
# You can also call specific plots by passing their name as an argument, as with `df.plot(kind='area')`.
# 
# Let's start going through them! First we'll look at the data:
# 

# ### Histograms
# This is one of the most commonly used plots. Histograms describe the distribution of continuous data by dividing the data into "bins" of equal width, and plotting the number of values that fall into each bin. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.hist.html'>[reference]</a>

# In[31]:


df1['A'].plot.hist();


# We can add settings to do things like bring the x- and y-axis values to the edge of the graph, and insert lines between vertical bins:

# In[32]:


df1['A'].plot.hist(edgecolor='k').autoscale(enable=True, axis='both', tight=True)


# You can use any [matplotlib color spec](https://matplotlib.org/api/colors_api.html) for **edgecolor**, such as `'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'`, or the string representation of a float value for shades of grey, such as `'0.5'`
# 
# For **autoscale** the axis can be set to `'x'`, `'y'` or `'both'`
# 
# We can also change the number of bins (the range of values over which frequencies are calculated) from the default value of 10:

# In[33]:


df1['A'].plot.hist(bins=40, edgecolor='k').autoscale(enable=True, axis='both', tight=True)


# You can also access an histogram like this:

# In[34]:


df1['A'].hist();


# For more on using <tt>df.hist()</tt> visit https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html

# ## Barplots
# Barplots are similar to histograms, except that they deal with discrete data, and often reflect multiple variables. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.bar.html'>[reference]</a>

# In[35]:


df2.plot.bar();


# In[36]:


df2.plot.bar(stacked=True);


# In[37]:


# USE .barh() TO DISPLAY A HORIZONTAL BAR PLOT
df2.plot.barh();


# ## Line Plots
# Line plots are used to compare two or more variables. By default the x-axis values are taken from the index. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.line.html'>[reference]</a>
# 
# Line plots happen to be the default pandas plot. They are accessible through <tt>df.plot()</tt> as well as <tt>df.plot.line()</tt>

# In[38]:


df2.plot.line(y='a',figsize=(12,3),lw=2);


# In[39]:


# Use lw to change the size of the line

df2.plot.line(y=['a','b','c'],figsize=(12,3),lw=3);


# ## Area Plots
# Area plots represent cumulatively stacked line plots where the space between lines is emphasized with colors. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.area.html'>[reference]</a>

# In[40]:


df2.plot.area();


# It often helps to mute the colors by passing an <strong>alpha</strong> transparency value between 0 and 1.

# In[41]:


df2.plot.area(alpha=0.4);


# To produce a blended area plot, pass a <strong>stacked=False</strong> argument:

# In[42]:


df2.plot.area(stacked=False, alpha=0.4);


# ## Scatter Plots
# Scatter plots are a useful tool to quickly compare two variables, and to look for possible trends. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.scatter.html'>[reference]</a>

# In[43]:


df1.plot.scatter(x='A',y='B');


# ### Scatter plots with colormaps
# You can use <strong>c</strong> to color each marker based off another column value. Use `cmap` to indicate which colormap to use.<br>
# For all the available colormaps, check out: http://matplotlib.org/users/colormaps.html

# In[44]:


df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm');


# ### Scatter plots with sized markers
# Alternatively you can use <strong>s</strong> to indicate marker size based off another column. The <strong>s</strong> parameter needs to be an array, not just the name of a column:

# In[45]:


df1.plot.scatter(x='A',y='B',s=df1['C']*50);


# The warning above appeared because some `df1['C']` values are negative. We can fix this finding the minimum value, writing a function that adds to each value, and applying our function to the data with <strong>.apply(func)</strong>.
# 
# Also, these data points have a lot of overlap. We can address this issue by passing in an <strong>alpha</strong> blending value between 0 and 1 to make markers more transparent.

# In[46]:


def add_three(val):
    return val+3

df1.plot.scatter(x='A',y='B',s=df1['C'].apply(add_three)*45, alpha=0.2);


# ## BoxPlots
# Box plots, aka "box and whisker diagrams", describe the distribution of data by dividing data into <em>quartiles</em> about the mean.<br>
# Look <a href='https://en.wikipedia.org/wiki/Box_plot'>here</a> for a description of boxplots. <a href='https://pandas.pydata.org/pandas-docs/stable/visualization.html#box-plots'>[reference]</a>

# In[47]:


df2.boxplot();


# ### Boxplots with Groupby
# To draw boxplots based on groups, first pass in a list of columns you want plotted (including the groupby column), then pass <strong>by='columname'</strong> into <tt>.boxplot()</tt>. Here we'll group records by the <strong>'e'</strong> column, and draw boxplots for the <strong>'b'</strong> column.

# In[48]:


df2[['b','e']].boxplot(by='e', grid=False);


# In the next section on Customizing Plots we'll show how to change the title and axis labels.

# ## Kernel Density Estimation (KDE) Plot
# In order to see the underlying distribution, which is similar to an histogram.
# These plots are accessible either through <tt>df.plot.kde()</tt> or <tt>df.plot.density()</tt> <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.kde.html'>[reference]</a>

# In[49]:


df2['a'].plot.kde();


# In[50]:


df2.plot.density();


# ## Hexagonal Bin Plot
# 
# Useful for Bivariate Data, alternative to scatterplot. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.hexbin.html'>[reference]</a>

# In[51]:


# FIRST CREATE A DATAFRAME OF RANDOM VALUES
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])

# MAKE A HEXAGONAL BIN PLOT
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges');


# # HTML Input
# Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects:

# In[52]:


df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')


# # Customizing Pandas Plots
# In this section we'll show how to control the position and appearance of axis labels and legends.<br>
# For more info on the following topics visit https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# ## Colors, Widths and Linestyles
# The pandas <tt>.plot()</tt> method takes optional arguments that allow you to control linestyles, colors, widths and more.

# In[ ]:


# START WITH A SIMPLE LINE PLOT
df2['c'].plot(figsize=(8,3));


# In[ ]:


df2['c'].plot.line(ls='-.', c='r', lw='4', figsize=(8,3));


# For more on linestyles, click <a href='https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle'>here</a>.

# ## Adding Titles and Axis Labels

# In[ ]:


# START WITH A SIMPLE MULTILINE PLOT
df2.plot(figsize=(8,3));


# ### Object-oriented plotting
# 
# When we call <tt>df.plot()</tt>, pandas returns a <tt>matplotlib.axes.AxesSubplot</tt> object. We can set labels
# on that object so long as we do it in the same jupyter cell. Setting an autoscale is done the same way.

# In[ ]:


title='Custom Pandas Plot'
ylabel='Y-axis data'
xlabel='X-axis data'

ax = df2.plot(figsize=(8,3),title=title)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.autoscale(axis='x',tight=True);


# ## Plot Legend Placement
# Pandas tries to optimize placement of the legend to reduce overlap on the plot itself. However, we can control the placement ourselves, and even place the legend outside of the plot. We do this through the <tt>.legend()</tt> method. <a href='https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html'>[reference]</a>

# For starters we can pass a location code. <tt>.legend(loc=1)</tt> places the legend in the upper-right corner of the plot.<br>Alternatively we can pass a location string: <tt>.legend(loc='upper right')</tt> does the same thing.
# 
# <table style="display: inline-block">
# <tr><th>LOCATION CODE</th><th>LOCATION STRING</th></tr>
# <tr><td>0</td><td>'best'</td></tr>
# <tr><td>1</td><td>'upper right'</td></tr>
# <tr><td>2</td><td>'upper left'</td></tr>
# <tr><td>3</td><td>'lower left'</td></tr>
# <tr><td>4</td><td>'lower right'</td></tr>
# <tr><td>5</td><td>'right'</td></tr>
# <tr><td>6</td><td>'center left'</td></tr>
# <tr><td>7</td><td>'center right'</td></tr>
# <tr><td>8</td><td>'lower center'</td></tr>
# <tr><td>9</td><td>'upper center'</td></tr>
# <tr><td>10</td><td>'center'</td></tr>
# </table>

# In[ ]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=1);


# We can pass a second argument, <tt>bbox_to_anchor</tt> that treats the value passed in through <tt>loc</tt> as an anchor point, and positions the legend along the x and y axes based on a two-value tuple.

# In[ ]:


# FIRST, PLACE THE LEGEND IN THE LOWER-LEFT
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3);


# In[ ]:


# NEXT, MOVE THE LEGEND A LITTLE TO THE RIGHT AND UP
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(0.1,0.1));


# ### Placing the Legend Outside the Plot
# In the above plot we passed <tt>(0.1,0.1)</tt> as our two-item tuple. This places the legend slightly to the right and slightly upward.<br>To place the legend outside the plot on the right-hand side, pass a value greater than or equal to 1 as the first item in the tuple.

# In[ ]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(1.0,0.1));


# ## Pandas Datetime Index
# 
# We'll usually deal with time series as a datetime index when working with pandas dataframes. Fortunately pandas has a lot of functions and methods to work with time series!<br>
# For more on the pandas DatetimeIndex visit https://pandas.pydata.org/pandas-docs/stable/timeseries.html

# Ways to build a DatetimeIndex:

# In[ ]:


# THE WEEK OF JULY 8TH, 2018
idx = pd.date_range('7/8/2018', periods=7, freq='D')
idx


# In[ ]:


idx = pd.to_datetime(['Jan 01, 2018','1/2/18','03-Jan-2018',None])
idx


# In[ ]:


# Create a NumPy datetime array
some_dates = np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[D]')
some_dates


# In[ ]:


pd.to_datetime(['2/1/2018','3/1/2018'],format='%d/%m/%Y')


# # Time Resampling
# 

# Note: the above code is a faster way of doing the following:
# <pre>df = pd.read_csv('../Data/starbucks.csv')
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date',inplace=True)</pre>

# ## resample()
# 
# A common operation with time series data is resampling based on the time series index. Let's see how to use the resample() method. [[reference](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)]

# In[ ]:


# Index_col indicates that the index will be the column called 'Date'
# parse_dates, transforms the strings into datetime format

df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[ ]:


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

# In[ ]:


# Yearly Means
df.resample(rule='A').mean()


# Resampling rule 'A' takes all of the data points in a given year, applies the aggregation function (in this case we calculate the mean), and reports the result as the last day of that year.

# In[ ]:


title = 'Monthly Max Closing Price for Starbucks'
df['Close'].resample('M').max().plot.bar(figsize=(16,6), title=title,color='#1f77b4');


# # Time Shifting
# 
# Sometimes you may need to shift all your data up or down along the time series index. In fact, a lot of pandas built-in methods do this under the hood. This isn't something we'll do often in the course, but it's definitely good to know about this anyways!

# In[ ]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv',index_col='Date',parse_dates=True)


# ## .shift() forward
# This method shifts the entire date index a given number of rows, without regard for time periods (months & years).<br>It returns a modified copy of the original DataFrame.
# 
# In other words, it moves down all the rows down or up.

# In[ ]:


# We move down all the rows
df.shift(1).head()


# In[ ]:


# NOTE: You will lose that last piece of data that no longer has an index!
df.shift(1).tail()


# ## Shifting based on Time Series Frequency Code
# 
# We can choose to shift <em>index values</em> up or down without realigning the data by passing in a <strong>freq</strong> argument.<br>
# This method shifts dates to the next period based on a frequency code. Common codes are 'M' for month-end and 'A' for year-end. <br>Refer to the <em>Time Series Offset Aliases</em> table from the Time Resampling lecture for a full list of values, or click <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.<br>

# In[ ]:


# Shift everything to the end of the month
df.shift(periods=1, freq='M').head()


# For more info on time shifting visit http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html<br>

# # Rolling and Expanding
# 
# A common process with time series is to create data based off of a rolling mean. The idea is to divide the data into "windows" of time, and then calculate an aggregate function for each window. In this way we obtain a <em>simple moving average</em>. Let's show how to do this easily with pandas!

# In[ ]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[ ]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# Now let's add in a rolling mean! This rolling method provides row entries, where every entry is then representative of the window. 

# In[ ]:


# 7 day rolling mean
df.rolling(window=7).mean().head(15)


# In[ ]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)
df.rolling(window=30).mean()['Close'].plot();


# ## Expanding
# 
# Instead of calculating values for a rolling window of dates, what if you wanted to take into account everything from the start of the time series up to each point in time? For example, instead of considering the average over the last 7 days, we would consider all prior data in our expanding set of averages.

# In[ ]:



# Optional: specify a minimum number of periods to start from
df['Close'].expanding(min_periods=30).mean().plot(figsize=(12,5));


# # Visualizing Time Series Data

# In[ ]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv',index_col='Date',parse_dates=True)


# First we'll create a line plot that puts both <tt>'Close'</tt> and <tt>'Volume'</tt> on the same graph.<br>Remember that we can use <tt>df.plot()</tt> in place of <tt>df.plot.line()</tt>

# In[ ]:


df.index = pd.to_datetime(df.index)


# In[ ]:


df.plot();


# ## Adding a title and axis labels

# In[ ]:


title='Starbucks Closing Stock Prices'
ylabel='Closing Price (USD)'
xlabel='Closing Date'

ax = df['Close'].plot(figsize=(12,6),title=title)
ax.autoscale(axis='x',tight=True) 
ax.set(xlabel=xlabel, ylabel=ylabel);


# 
# 
# 
# Thanks to the date index, we can make a selection like the following:
# 
# 

# In[53]:


df['Close']['2017-01-01':'2017-03-01']


# ## X Limits
# There are two ways we can set a specific span of time as an x-axis limit. We can plot a slice of the dataset, or we can pass x-limit values as an argument into <tt>df.plot()</tt>.
# 
# The advantage of using a slice is that pandas automatically adjusts the y-limits accordingly.
# 
# The advantage of passing in arguments is that pandas automatically tightens the x-axis. Plus, if we're also setting y-limits this can improve readability.

# ### Choosing X Limits by Slice:

# In[54]:


# Dates are separated by a colon:
df['Close']['2017-01-01':'2017-03-01'].plot(figsize=(12,4)).autoscale(axis='x',tight=True);


# ### Choosing X Limits by Argument:

# In[55]:


# Dates are separated by a comma:
#Let's say we want to display the plot only from the 1st of january until the 1t of march
df['Close'].plot(figsize=(12,4),xlim=['2017-01-01','2017-03-01']);


# Now let's focus on the y-axis limits to get a better sense of the shape of the data.<br>First we'll find out what upper and lower limits to use.

# In[56]:


# FIND THE MINIMUM VALUE IN THE RANGE:
df.loc['2017-01-01':'2017-03-01']['Close'].min()


# ## Title and axis labels
# Let's add a title and axis labels to our subplot.
# <div class="alert alert-info"><strong>REMEMBER:</strong> <tt><font color=black>ax.autoscale(axis='both',tight=True)</font></tt> is unnecessary if axis limits have been passed into <tt>.plot()</tt>.<br>
# If we were to add it, autoscale would revert the axis limits to the full dataset.</div>

# In[57]:


title='Starbucks Closing Stock Prices'
ylabel='Closing Price (USD)'
xlabel='Closing Date'

ax = df['Close'].plot(xlim=['2017-01-04','2017-03-01'],ylim=[51,57],figsize=(12,4),title=title)
ax.set(xlabel=xlabel, ylabel=ylabel);


# We can pass arguments into <tt>.plot()</tt> to change the linestyle and color. Refer to the Customizing Plots lecture from the previous section for more options.

# In[58]:


df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],ls='--',c='r');


# ## X Ticks
# In this section we'll look at how to change the format and appearance of dates along the x-axis. To do this, we'll borrow a tool from <tt>matplotlib</tt> called <tt>dates</tt>.

# ### Set the spacing
# The x-axis values can be divided into major and minor axes. For now, we'll work only with the major axis and learn how to set the spacing with <tt>.set_major_locator()</tt>.
# As you can see in the graph below, 
# the X axis is not beautifully distributed

# In[59]:


# CREATE OUR AXIS OBJECT
ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57])


# With set_major_locator we can solve this problem

# In[60]:



# CREATE OUR AXIS OBJECT
ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57])

# REMOVE PANDAS DEFAULT "Date" LABEL
ax.set(xlabel='')

# SET THE TICK LOCATOR AND FORMATTER FOR THE MAJOR AXIS
# byweekday = 0 means Monday

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))


# Notice that dates are spaced one week apart. The dates themselves correspond with <tt>byweekday=0</tt>, or Mondays.<br>
# For a full list of locator options available from <tt>matplotlib.dates</tt> visit <a href='https://matplotlib.org/api/dates_api.html#date-tickers'>https://matplotlib.org/api/dates_api.html#date-tickers</a>

# ### Date Formatting
# Formatting follows the Python datetime <strong><a href='http://strftime.org/'>strftime</a></strong> codes.<br>
# The following examples are based on <tt>datetime.datetime(2001, 2, 3, 16, 5, 6)</tt>:
# <br><br>
# 
# <table style="display: inline-block">  
# <tr><th>CODE</th><th>MEANING</th><th>EXAMPLE</th><tr>
# <tr><td>%Y</td><td>Year with century as a decimal number.</td><td>2001</td></tr>
# <tr><td>%y</td><td>Year without century as a zero-padded decimal number.</td><td>01</td></tr>
# <tr><td>%m</td><td>Month as a zero-padded decimal number.</td><td>02</td></tr>
# <tr><td>%B</td><td>Month as locale’s full name.</td><td>February</td></tr>
# <tr><td>%b</td><td>Month as locale’s abbreviated name.</td><td>Feb</td></tr>
# <tr><td>%d</td><td>Day of the month as a zero-padded decimal number.</td><td>03</td></tr>  
# <tr><td>%A</td><td>Weekday as locale’s full name.</td><td>Saturday</td></tr>
# <tr><td>%a</td><td>Weekday as locale’s abbreviated name.</td><td>Sat</td></tr>
# <tr><td>%H</td><td>Hour (24-hour clock) as a zero-padded decimal number.</td><td>16</td></tr>
# <tr><td>%I</td><td>Hour (12-hour clock) as a zero-padded decimal number.</td><td>04</td></tr>
# <tr><td>%p</td><td>Locale’s equivalent of either AM or PM.</td><td>PM</td></tr>
# <tr><td>%M</td><td>Minute as a zero-padded decimal number.</td><td>05</td></tr>
# <tr><td>%S</td><td>Second as a zero-padded decimal number.</td><td>06</td></tr>
# </table>
# <table style="display: inline-block">
# <tr><th>CODE</th><th>MEANING</th><th>EXAMPLE</th><tr>
# <tr><td>%#m</td><td>Month as a decimal number. (Windows)</td><td>2</td></tr>
# <tr><td>%-m</td><td>Month as a decimal number. (Mac/Linux)</td><td>2</td></tr>
# <tr><td>%#x</td><td>Long date</td><td>Saturday, February 03, 2001</td></tr>
# <tr><td>%#c</td><td>Long date and time</td><td>Saturday, February 03, 2001 16:05:06</td></tr>
# </table>  
#     

# In[61]:


# USE THIS SPACE TO EXPERIMENT WITH DIFFERENT FORMATS
from datetime import datetime
datetime(2001, 2, 3, 16, 5, 6).strftime("%A, %B %d, %Y  %I:%M:%S %p")


# We use the set_major_formatter to format the display of the date in the plot

# In[62]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))


# ## Major vs. Minor Axis Values
# All of the tick marks we've used so far have belonged to the major axis. We can assign another level called the <em>minor axis</em>, perhaps to separate month names from days of the month.

# In[63]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],rot=0,title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))


# ## Adding Gridlines
# We can add x and y axis gridlines that extend into the plot from each major tick mark.

# In[64]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],rot=0,title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))

ax.yaxis.grid(True)
ax.xaxis.grid(True)


# # TIME SERIES
# ## Statsmodels

# <div class="alert alert-info"><h3>For Further Reading:</h3>
# <strong>
# <a href='http://www.statsmodels.org/stable/tsa.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Time Series Analysis</font></div>
# 
# Let's walk through a very simple example of using statsmodels!

# ### Perform standard imports and load the dataset
# For these exercises we'll be using a statsmodels built-in macroeconomics dataset:
# 
# <pre><strong>US Macroeconomic Data for 1959Q1 - 2009Q3</strong>
# Number of Observations - 203
# Number of Variables - 14
# Variable name definitions:
#     year      - 1959q1 - 2009q3
#     quarter   - 1-4
#     realgdp   - Real gross domestic product (Bil. of chained 2005 US$,
#                 seasonally adjusted annual rate)
#     realcons  - Real personal consumption expenditures (Bil. of chained
#                 2005 US$, seasonally adjusted annual rate)
#     realinv   - Real gross private domestic investment (Bil. of chained
#                 2005 US$, seasonally adjusted annual rate)
#     realgovt  - Real federal consumption expenditures & gross investment
#                 (Bil. of chained 2005 US$, seasonally adjusted annual rate)
#     realdpi   - Real private disposable income (Bil. of chained 2005
#                 US$, seasonally adjusted annual rate)
#     cpi       - End of the quarter consumer price index for all urban
#                 consumers: all items (1982-84 = 100, seasonally adjusted).
#     m1        - End of the quarter M1 nominal money stock (Seasonally
#                 adjusted)
#     tbilrate  - Quarterly monthly average of the monthly 3-month
#                 treasury bill: secondary market rate
#     unemp     - Seasonally adjusted unemployment rate (%)
#     pop       - End of the quarter total population: all ages incl. armed
#                 forces over seas
#     infl      - Inflation rate (ln(cpi_{t}/cpi_{t-1}) * 400)
#     realint   - Real interest rate (tbilrate - infl)</pre>
#     
# <div class="alert alert-info"><strong>NOTE:</strong> Although we've provided a .csv file in the Data folder, you can also build this DataFrame with the following code:<br>
# <tt>&nbsp;&nbsp;&nbsp;&nbsp;import pandas as pd<br>
# &nbsp;&nbsp;&nbsp;&nbsp;import statsmodels.api as sm<br>
# &nbsp;&nbsp;&nbsp;&nbsp;df = sm.datasets.macrodata.load_pandas().data<br>
# &nbsp;&nbsp;&nbsp;&nbsp;df.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))<br>
# &nbsp;&nbsp;&nbsp;&nbsp;print(sm.datasets.macrodata.NOTE)</tt></div>

# In[65]:


from statsmodels.tsa.filters.hp_filter import hpfilter
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/macrodata.csv',index_col=0,parse_dates=True)
df.head()


# In[66]:


ax = df['realgdp'].plot()
ax.autoscale(axis='x',tight=True)
ax.set(ylabel='REAL GDP');


# ## Using Statsmodels to get the trend
# <div class="alert alert-info"><h3>Related Function:</h3>
# <tt><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html'><strong>statsmodels.tsa.filters.hp_filter.hpfilter</strong></a><font color=black>(X, lamb=1600)</font>&nbsp;&nbsp;Hodrick-Prescott filter</div>
#     
# The <a href='https://en.wikipedia.org/wiki/Hodrick%E2%80%93Prescott_filter'>Hodrick-Prescott filter</a> separates a time-series  $y_t$ into a trend component $\tau_t$ and a cyclical component $c_t$
# 
# $y_t = \tau_t + c_t$
# 
# The components are determined by minimizing the following quadratic loss function, where $\lambda$ is a smoothing parameter:
# 
# $\min_{\\{ \tau_{t}\\} }\sum_{t=1}^{T}c_{t}^{2}+\lambda\sum_{t=1}^{T}\left[\left(\tau_{t}-\tau_{t-1}\right)-\left(\tau_{t-1}-\tau_{t-2}\right)\right]^{2}$
# 
# 
# The $\lambda$ value above handles variations in the growth rate of the trend component.<br>When analyzing quarterly data, the default lambda value of 1600 is recommended. Use 6.25 for annual data, and 129,600 for monthly data.

# The following function extracts the cycle and trend component and returns it

# ## Hodrick-Prescott filter 

# In[67]:


# Tuple unpacking
gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
#Both are a pandas.core.series.Series


# Let's plot the results

# In[68]:


gdp_cycle.plot();


# In[69]:


gdp_trend.plot();


# In[70]:


df['trend'] = gdp_trend
df[['trend','realgdp']].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# ## ETS
# 
# ## Error/Trend/Seasonality Models
# As we begin working with <em>endogenous</em> data ("endog" for short) and start to develop forecasting models, it helps to identify and isolate factors working within the system that influence behavior. Here the name "endogenous" considers internal factors, while "exogenous" would relate to external forces. These fall under the category of <em>state space models</em>, and include <em>decomposition</em> (described below), and <em>exponential smoothing</em> (described in an upcoming section).
# 
# The <a href='https://en.wikipedia.org/wiki/Decomposition_of_time_series'>decomposition</a> of a time series attempts to isolate individual components such as <em>error</em>, <em>trend</em>, and <em>seasonality</em> (ETS). We've already seen a simplistic example of this in the <strong>Introduction to Statsmodels</strong> section with the Hodrick-Prescott filter. There we separated data into a trendline and a cyclical feature that mapped observed data back to the trend.
# 
# <div class="alert alert-info"><h3>Related Function:</h3>
# <tt><strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html'>statsmodels.tsa.seasonal.seasonal_decompose</a></strong><font color=black>(x, model)</font>&nbsp;&nbsp;
# Seasonal decomposition using moving averages</tt>
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://otexts.com/fpp2/ets.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Innovations state space models for exponential smoothing</font><br>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Decomposition_of_time_series'>Wikipedia</a></strong>&nbsp;&nbsp;<font color=black>Decomposition of time series</font></div>
# 
# ## Seasonal Decomposition
# Statsmodels provides a <em>seasonal decomposition</em> tool we can use to separate out the different components. This lets us see quickly and visually what each component contributes to the overall behavior.
# 
# 
# We apply an <strong>additive</strong> model when it seems that the trend is more linear and the seasonality and trend components seem to be constant over time (e.g. every year we add 10,000 passengers).<br>
# A <strong>multiplicative</strong> model is more appropriate when we are increasing (or decreasing) at a non-linear rate (e.g. each year we double the amount of passengers).
# 
# For these examples we'll use the International Airline Passengers dataset, which gives monthly totals in thousands from January 1949 to December 1960.

# In[71]:


from statsmodels.tsa.seasonal import seasonal_decompose

airline = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)

airline.dropna(inplace=True)

airline.plot();


# The graph above shows clearle that a additive model fits better. The number of passangers grows with time

# Using the seasonal_decompose from statsmodels we will decompose the time series into its components:

# In[72]:


result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')  # model='mul' also works
result.plot();


# In[73]:


#press on tab to check the different components
result.seasonal.plot();


# # SMA
# ## Simple Moving Average
# 
# We've already shown how to create a <a href='https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average'>simple moving average</a> by applying a <tt>mean</tt> function to a rolling window.
# 
# For a quick review

# In[75]:


airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()


# In[79]:


airline.plot();


# # EWMA-Exponentially-Weighted-Moving-Average

# 
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
# 
# Where $x_t$ is the input value, $w_i$ is the applied weight (Note how it can change from $i=0$ to $t$), and $y_t$ is the output.
# 
# Now the question is, how do we define the weight term $w_i$?
# 
# This depends on the <tt>adjust</tt> parameter you provide to the <tt>.ewm()</tt> method.
# 
# 
# When <tt>adjust=True</tt> (default) is used, weighted averages are calculated using weights equal to $w_i = (1 - \alpha)^i$
# This is called <b> Simple Exponential Smoothing </b> 
# which gives:
# 
# ### $y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
# + (1 - \alpha)^t x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
# + (1 - \alpha)^t}$
# 
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
# 
# When <tt>adjust=True</tt> we have $y_0=x_0$ and from the last representation above we have 
# $y_t=\alpha x_t+(1−α)y_{t−1}$, therefore there is an assumption that $x_0$ is not an ordinary value but rather an exponentially weighted moment of the infinite series up to that point.
# 
# The use of adjust=True or adjust=False, has to do whether a series is finite or infinite. In fact if you take an infinite series and pass it in the adjust=True formula, you will end up having the adjust=False formula.
# 
# For the smoothing factor $\alpha$ one must have $0<\alpha≤1$, and while it is possible to pass <em>alpha</em> directly, it’s often easier to think about either the <em>span</em>, <em>center of mass</em> (com) or <em>half-life</em> of an EW moment:
# 
# \begin{split}\alpha =
#  \begin{cases}
#      \frac{2}{s + 1},               & \text{for span}\ s \geq 1\\
#      \frac{1}{1 + c},               & \text{for center of mass}\ c \geq 0\\
#      1 - \exp^{\frac{\log 0.5}{h}}, & \text{for half-life}\ h > 0
#  \end{cases}\end{split}
#  
# <em> Those are the ways alpha can be calculated, but you can also just input it manually </em> 
#  
# * <strong>Span</strong> corresponds to what is commonly called an “N-day EW moving average”. It's the window.
# * <strong>Center of mass</strong> has a more physical interpretation and can be thought of in terms of span: $c=(s−1)/2$
# * <strong>Half-life</strong> is the period of time for the exponential weight to reduce to one half.
# * <strong>Alpha</strong> specifies the smoothing factor directly.
# 
# We have to pass precisely one of the above into the <tt>.ewm()</tt> function. For our data we'll use <tt>span=12</tt>.
# <em> </em>
# 
# Disadvantages of EWMA: 
# 
#     - Does not take into account trend and seasonality. 

# In[91]:


# With span = 12, we are using al alpha = 2/(12+1). Look at the above formula.
# Therefore when we pass the parameter span in the formula, we are setting alpha
airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()
airline[['Thousands of Passengers','EWMA12']].plot();


# In[82]:


airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()
airline[['Thousands of Passengers','EWMA12']].plot();


# ## SMA vs EWMA

# In[83]:


airline[['Thousands of Passengers','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);


# # Holt-Winters Methods
# In the previous section on <strong>Exponentially Weighted Moving Averages</strong> (EWMA) we applied <em>Simple Exponential Smoothing</em> using just one smoothing factor $\alpha$ (alpha). This failed to account for other contributing factors like trend and seasonality.
# 
# In this section we'll look at <em>Double</em> and <em>Triple Exponential Smoothing</em> with the <a href='https://otexts.com/fpp2/holt-winters.html'>Holt-Winters Methods</a>. 
# 
# In <strong>Double Exponential Smoothing</strong> (double is because we have 2 parameters) (aka Holt's Method) we introduce a new smoothing factor $\beta$ (beta) that addresses <b> trend </b>:
# 
# \begin{split}l_t &= (1 - \alpha) l_{t-1} + \alpha x_t, & \text{    level}\\
# b_t &= (1-\beta)b_{t-1} + \beta(l_t-l_{t-1}) & \text{    trend}\\
# y_t &= l_t + b_t & \text{    fitted model}\\
# \hat y_{t+h} &= l_t + hb_t & \text{    forecasting model (} h = \text{# periods into the future)}\end{split}
# 
# Because we haven't yet considered seasonal fluctuations, the forecasting model is simply a straight sloped line extending from the most recent data point. We'll see an example of this in upcoming lectures.
# 
# With <strong>Triple Exponential Smoothing</strong> (aka the Holt-Winters Method) we introduce a smoothing factor $\gamma$ (gamma) that addresses seasonality:
# 
# \begin{split}l_t &= (1 - \alpha) l_{t-1} + \alpha x_t, & \text{    level}\\
# b_t &= (1-\beta)b_{t-1} + \beta(l_t-l_{t-1}) & \text{    trend}\\
# c_t &= (1-\gamma)c_{t-L} + \gamma(x_t-l_{t-1}-b_{t-1}) & \text{    seasonal}\\
# y_t &= (l_t + b_t) c_t & \text{    fitted model}\\
# \hat y_{t+m} &= (l_t + mb_t)c_{t-L+1+(m-1)modL} & \text{    forecasting model (} m = \text{# periods into the future)}\end{split}
# 
# Here $L$ represents the number of divisions per cycle. In our case looking at monthly data that displays a repeating pattern each year, we would use $L=12$.
# 
# In general, higher values for $\alpha$, $\beta$ and $\gamma$ (values closer to 1), place more emphasis on recent data.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html'>statsmodels.tsa.holtwinters.SimpleExpSmoothing</a></strong><font color=black>(endog)</font>&nbsp;&nbsp;&nbsp;&nbsp;
# Simple Exponential Smoothing<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html'>statsmodels.tsa.holtwinters.ExponentialSmoothing</a></strong><font color=black>(endog)</font>&nbsp;&nbsp;
#     Holt-Winters Exponential Smoothing</tt>
#     
# <h3>For Further Reading:</h3>
# <tt>
# <strong>
# <a href='https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm'>NIST/SEMATECH e-Handbook of Statistical Methods</a></strong>&nbsp;&nbsp;<font color=black>What is Exponential Smoothing?</font></tt></div>

# # Simple Exponential Smoothing - Simple Moving Average

# We worked on this above. Just as a reminder. A variation of the statmodels Holt-Winters function provides Simple Exponential Smoothing. We'll show that it performs the same calculation of the weighted moving average as the pandas <tt>.ewm()</tt> method:<br>
# $\begin{split}y_0 &= x_0 \\
# y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,\end{split}$

# In[113]:


#We will keep using the arilines dataframe
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)

df.dropna(inplace=True)


# Note that our DatetimeIndex does not have a frequency. In order to build a Holt-Winters smoothing model, statsmodels needs to know the frequency of the data (whether it's daily, monthly etc.). Since observations occur at the start of each month, we'll use MS.<br>A full list of time series offset aliases can be found <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.

# In[114]:


df.index


# In[115]:


df.index.freq = 'MS'
df.index


# In[116]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span = 12
alpha = 2/(span+1)
#as we did in the previous cells
df['EWMA12'] = df['Thousands of Passengers'].ewm(alpha=alpha,adjust=False).mean()


# Now we are going to implement the SIMPLE EXP SMOOTHING model, but first let's understand what is going on with the code:

# In[117]:


model = SimpleExpSmoothing(df['Thousands of Passengers'])


# In[118]:


# With Tab you can check the various attributes of the object "model", like "fit"
# As paremeters in the fit method we have the smoothing_level which is alpha
fitted_model =  model.fit(smoothing_level=alpha,optimized=False)

# This returns a HoltWintersResultsWrapper

fitted_model.fittedvalues


# <div class="alert alert-danger"><strong>NOTE:</strong> For some reason, when <tt>optimized=False</tt> is passed into <tt>.fit()</tt>, the statsmodels <tt>SimpleExpSmoothing</tt> function shifts fitted values down one row. We fix this by adding <tt>.shift(-1)</tt> after <tt>.fittedvalues</tt></div>

# In[119]:


# We will simply solve the problem with the shift function

fitted_model.fittedvalues.shift(-1)


# In[120]:


# we add the predictions to our dataframe

df['SES12'] = fitted_model.fittedvalues.shift(-1)


# In[121]:


df.head()


# Note that we could have simply put all the code in the cells into one line of code:

# In[122]:


#df['SES12']=SimpleExpSmoothing(df['Thousands of Passengers']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)


# In[123]:


df


# In[128]:


# EWMA12 and SES12 are the same in this particular case
df.plot(figsize=(14,6))


# Now that we have the Simple Exponential Smoothing, let's try the Double Exponential Smoothing, before we plot the results.

# ___
# ## Double Exponential Smoothing - Holt's Method
# Where Simple Exponential Smoothing employs just one smoothing factor $\alpha$ (alpha), Double Exponential Smoothing adds a second smoothing factor $\beta$ (beta) that addresses trends in the data. Like the alpha factor, values for the beta factor fall between zero and one ($0<\beta≤1$). The benefit here is that the model can anticipate future increases or decreases where the level model would only work from recent calculations.
# 
# We can also address different types of change (growth/decay) in the trend. If a time series displays a straight-line sloped trend, you would use an <strong>additive</strong> adjustment. If the time series displays an exponential (curved) trend, you would use a <strong>multiplicative</strong> adjustment.
# 
# As we move toward forecasting, it's worth noting that both additive and multiplicative adjustments may become exaggerated over time, and require <em>damping</em> that reduces the size of the trend over future periods until it reaches a flat line.

# In[131]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

#It's hard to say wethet the passengers data has a multiplicative or additive trend, we will assume it is additive


df['DESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='add').fit().fittedvalues.shift(-1)
df.head()


# The DESadd12 is almost the same as the number of passangers (the blue line is just behind the red)

# In[133]:


df.plot(figsize=(12,5))


# In[134]:


# Let's do some Zoom
df.iloc[:24].plot(figsize=(12,5))


# Double exponential is clearly better than Simple

# What about Tripple ?

# ___
# ## Triple Exponential Smoothing - Holt-Winters Method
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data. 
# 
# 

# In[135]:


df['TESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df.head()


# In[136]:


df['TESmul12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
df.head()


# In[137]:


df[['Thousands of Passengers','TESadd12','TESmul12']].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# In[138]:


# Let's chekc for the first 2 years (24 months)
df[['Thousands of Passengers','TESadd12','TESmul12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# # Forecasting
# 
# In the previous section we fit various smoothing models to existing data. So the purpose was not to forecast, but predict.
# The purpose now is to predict what happens next.<br>
# What's our best guess for next month's value? For the next six months?
# 
# In this section we'll look to extend our models into the future. First we'll divide known data into training and testing sets, and evaluate the performance of a trained model on known test data.
# 
# * Goals
#   * Compare a Holt-Winters forecasted model to known data
#   * Understand <em>stationarity</em>, <em>differencing</em> and <em>lagging</em>
#   * Introduce ARIMA and describe next steps

# The <b> Forecasting Procedure </b> looks like:
#   * Modle selection
#   * Splitting data into train and test sets
#   * Fit model on training set
#   * Evaluate model on test set
#   * Re-fit model on entire data set
#   * Forecast for future data

# ## Forecasting with the Holt-Winters Method
# For this example we'll use the same airline_passengers dataset, and we'll split the data into 108 training records and 36 testing records. Then we'll evaluate the performance of the model.

# In[140]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()


# ### Train Test Split

# In[143]:


train_data = df.iloc[:109] # Goes up to but not including 109
test_data = df.iloc[108:]


# ### Fitting the Model

# In[144]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# ### Evaluating model against test

# In[147]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8));


# In[148]:


test_predictions = fitted_model.forecast(36).rename('HW Forecast')


# In[151]:


test_predictions.plot(legend=True, label="Prediction")


# In[152]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8));
test_predictions.plot(legend=True, label="Prediction")


# The prediction (green line) seems to be quite accurate compared to the test data.
# Remember, the test data is real.
# 
# Recall and accuracy are not appropiate evaluation metrics for forecasting techniques. We need metrics designed for continuous values

# ## Evaluation Metrics
# 
# Let's analyse how can we use the most baic regression evaluation metrics
#     * Mean Absolute Error
#     * Mean Squared Error
#     * Root Mean Square Error

# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# ### Mean Absolute Error

# ![image.png](attachment:image.png)
# being y the true value and y hat the predicted value
# 
# Disadvantages:
# It does not take into account if a few predicted points are actually very far away from real points. 
# Exmaple: imagine spending for a mktg campaign is very high in december (Xmas) and the rest of the months is similar, then the MAE might not show us the effect of that month, since it is a mean.
# 
# Therefore we have the MSE as a solution

# ### Mean Squared Error

# ![image.png](attachment:image.png)
# 
# Since it is squared, those predicted points that are very far away from the real ones will have more importance in the calculation. You punish the model by having large errors.
# 
# Of course this means we cannot interpret directly the result (since it is squared). So we have the RMSE for this

# ### Root Mean Square Error

# ![image.png](attachment:image.png)
# 
# Squaring it we get the units back in its original form (like std with the variance).
# The intepretation dependends then on the data.
# A RMSE of 20€ of the price of a house is a very good, but for candy it's not.

# In[ ]:




