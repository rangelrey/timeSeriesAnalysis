#!/usr/bin/env python
# coding: utf-8

# # Numpy Reminders: 
# Let's have a look at some of the most basic numpy concepts that we will use in the more advanced sections

# In[102]:


import numpy as np
import datetime as datetime
from matplotlib import dates


# ## Numpy Creating data
# In order to test our functions, being able to create "random" data is the key

# In[5]:


#Create a range based on your input. From 0 to 10, not including 10 and step size parameter 2
np.arange(0,10,2)


# In[10]:


#Return evenly space 5 floats from 0 to 10, including 10
np.linspace(0,10,5)


# In[23]:


#return 2 radom floats from a uniform distribution (all numbers have the same probability)
np.random.rand(2)


# In[25]:


#return 2 radom floats from a normal distribution with mean = 0 and std = 1
np.random.randn(2)


# In[28]:


#return 2 radom floats from a normal distribution with mean = 3 and std = 1
np.random.normal(3,1,2)


# In[37]:


#Generate the same random numbers by setting a seed
#The number in the seed is irrelevant
#It will only work if we are on the same cell
np.random.seed(1)
print(np.random.rand(1))

np.random.seed(2)
print(np.random.rand(1))

np.random.seed(1)
print(np.random.rand(1))


# In[47]:


#Creating a matrix from an array
#First create an array
arr = np.arange(4)

#Then reshape it
matrix = arr.reshape(2,2)


# In[50]:


#return the min and max values of an array/matrix

print(arr.max())
print(matrix.min())


# ## Numpy Indexing and Selection
# 

# In[52]:


#Remember if you want to work with array copies use:
arr_copy = arr.copy()

#Otherwise you will be always editing the original array as well,
#since the new object created is pointing to the original object


# In[59]:


#Create a matrix
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix)
#Return the item in the column 1 and row 1
matrix[1][1]


# In[63]:


#Return (from the matrix) until the 2 row (not included)
matrix[:2]


# In[64]:


#Until row 2 and from column 1
matrix[:2,1:]


# In[74]:


#Return a filtered array whose values are lower than 2
print("Original array: ")
print(arr)
print("Result: ") 
print(arr[arr<2])


# ## Numpy Operations
# Skipping the basic ones

# In[81]:


#Sum of all the values of the columns
print(matrix)
matrix.sum(axis=0)


# In[82]:


#Sum of all the values of the rows
print(matrix)
matrix.sum(axis=1)


# # Pandas Reminders

# In[85]:


import pandas as pd


# In[87]:


#Create a Matrix, which will be used for the dataframe creation
rand_mat = np.random.rand(5,4)


# In[91]:


#Create dataframe
df = pd.DataFrame(data=rand_mat, index = 'A B C D E'.split(), columns = "R P U I".split())


# In[ ]:


#Drop row
df.drop("A")


# In[95]:


#Drop column
df.drop("R",axis=1)


# In[102]:


#Return series of the row A
df.loc["A"]


# In[103]:


#Return series of the row number 2
df.iloc[2]


# In[107]:


#Filtering by value. Filter all rows that are smaller than 0.3 in column I
df[df["I"]>0.3]


# In[109]:


#Return a unique array of the column R
df["R"].unique()


# In[110]:


#Return the number of unique items of the array of the column R
df["R"].nunique()


# In[112]:


#Apply a function to a column

df["R"].apply(lambda a: a+1)


# ## Pandas Viz Reminders

# In[115]:


#Display plots directly in jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


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

# In[10]:


df1['A'].plot.hist();


# We can add settings to do things like bring the x- and y-axis values to the edge of the graph, and insert lines between vertical bins:

# In[11]:


df1['A'].plot.hist(edgecolor='k').autoscale(enable=True, axis='both', tight=True)


# You can use any [matplotlib color spec](https://matplotlib.org/api/colors_api.html) for **edgecolor**, such as `'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'`, or the string representation of a float value for shades of grey, such as `'0.5'`
# 
# For **autoscale** the axis can be set to `'x'`, `'y'` or `'both'`
# 
# We can also change the number of bins (the range of values over which frequencies are calculated) from the default value of 10:

# In[12]:


df1['A'].plot.hist(bins=40, edgecolor='k').autoscale(enable=True, axis='both', tight=True)


# You can also access an histogram like this:

# In[14]:


df1['A'].hist();


# For more on using <tt>df.hist()</tt> visit https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html

# ## Barplots
# Barplots are similar to histograms, except that they deal with discrete data, and often reflect multiple variables. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.bar.html'>[reference]</a>

# In[15]:


df2.plot.bar();


# In[16]:


df2.plot.bar(stacked=True);


# In[17]:


# USE .barh() TO DISPLAY A HORIZONTAL BAR PLOT
df2.plot.barh();


# ## Line Plots
# Line plots are used to compare two or more variables. By default the x-axis values are taken from the index. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.line.html'>[reference]</a>
# 
# Line plots happen to be the default pandas plot. They are accessible through <tt>df.plot()</tt> as well as <tt>df.plot.line()</tt>

# In[18]:


df2.plot.line(y='a',figsize=(12,3),lw=2);


# In[19]:


# Use lw to change the size of the line

df2.plot.line(y=['a','b','c'],figsize=(12,3),lw=3);


# ## Area Plots
# Area plots represent cumulatively stacked line plots where the space between lines is emphasized with colors. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.area.html'>[reference]</a>

# In[20]:


df2.plot.area();


# It often helps to mute the colors by passing an <strong>alpha</strong> transparency value between 0 and 1.

# In[21]:


df2.plot.area(alpha=0.4);


# To produce a blended area plot, pass a <strong>stacked=False</strong> argument:

# In[23]:


df2.plot.area(stacked=False, alpha=0.4);


# ## Scatter Plots
# Scatter plots are a useful tool to quickly compare two variables, and to look for possible trends. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.scatter.html'>[reference]</a>

# In[24]:


df1.plot.scatter(x='A',y='B');


# ### Scatter plots with colormaps
# You can use <strong>c</strong> to color each marker based off another column value. Use `cmap` to indicate which colormap to use.<br>
# For all the available colormaps, check out: http://matplotlib.org/users/colormaps.html

# In[25]:


df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm');


# ### Scatter plots with sized markers
# Alternatively you can use <strong>s</strong> to indicate marker size based off another column. The <strong>s</strong> parameter needs to be an array, not just the name of a column:

# In[26]:


df1.plot.scatter(x='A',y='B',s=df1['C']*50);


# The warning above appeared because some `df1['C']` values are negative. We can fix this finding the minimum value, writing a function that adds to each value, and applying our function to the data with <strong>.apply(func)</strong>.
# 
# Also, these data points have a lot of overlap. We can address this issue by passing in an <strong>alpha</strong> blending value between 0 and 1 to make markers more transparent.

# In[27]:


def add_three(val):
    return val+3

df1.plot.scatter(x='A',y='B',s=df1['C'].apply(add_three)*45, alpha=0.2);


# ## BoxPlots
# Box plots, aka "box and whisker diagrams", describe the distribution of data by dividing data into <em>quartiles</em> about the mean.<br>
# Look <a href='https://en.wikipedia.org/wiki/Box_plot'>here</a> for a description of boxplots. <a href='https://pandas.pydata.org/pandas-docs/stable/visualization.html#box-plots'>[reference]</a>

# In[28]:


df2.boxplot();


# ### Boxplots with Groupby
# To draw boxplots based on groups, first pass in a list of columns you want plotted (including the groupby column), then pass <strong>by='columname'</strong> into <tt>.boxplot()</tt>. Here we'll group records by the <strong>'e'</strong> column, and draw boxplots for the <strong>'b'</strong> column.

# In[29]:


df2[['b','e']].boxplot(by='e', grid=False);


# In the next section on Customizing Plots we'll show how to change the title and axis labels.

# ## Kernel Density Estimation (KDE) Plot
# In order to see the underlying distribution, which is similar to an histogram.
# These plots are accessible either through <tt>df.plot.kde()</tt> or <tt>df.plot.density()</tt> <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.kde.html'>[reference]</a>

# In[33]:


df2['a'].plot.kde();


# In[34]:


df2.plot.density();


# ## Hexagonal Bin Plot
# 
# Useful for Bivariate Data, alternative to scatterplot. <a href='https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.hexbin.html'>[reference]</a>

# In[32]:


# FIRST CREATE A DATAFRAME OF RANDOM VALUES
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])

# MAKE A HEXAGONAL BIN PLOT
df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges');


# # HTML Input
# Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects:

# In[4]:


df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')


# # Customizing Pandas Plots
# In this section we'll show how to control the position and appearance of axis labels and legends.<br>
# For more info on the following topics visit https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# ## Colors, Widths and Linestyles
# The pandas <tt>.plot()</tt> method takes optional arguments that allow you to control linestyles, colors, widths and more.

# In[35]:


# START WITH A SIMPLE LINE PLOT
df2['c'].plot(figsize=(8,3));


# In[36]:


df2['c'].plot.line(ls='-.', c='r', lw='4', figsize=(8,3));


# For more on linestyles, click <a href='https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle'>here</a>.

# ## Adding Titles and Axis Labels

# In[37]:


# START WITH A SIMPLE MULTILINE PLOT
df2.plot(figsize=(8,3));


# ### Object-oriented plotting
# 
# When we call <tt>df.plot()</tt>, pandas returns a <tt>matplotlib.axes.AxesSubplot</tt> object. We can set labels
# on that object so long as we do it in the same jupyter cell. Setting an autoscale is done the same way.

# In[38]:


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

# In[40]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=1);


# We can pass a second argument, <tt>bbox_to_anchor</tt> that treats the value passed in through <tt>loc</tt> as an anchor point, and positions the legend along the x and y axes based on a two-value tuple.

# In[42]:


# FIRST, PLACE THE LEGEND IN THE LOWER-LEFT
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3);


# In[43]:


# NEXT, MOVE THE LEGEND A LITTLE TO THE RIGHT AND UP
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(0.1,0.1));


# ### Placing the Legend Outside the Plot
# In the above plot we passed <tt>(0.1,0.1)</tt> as our two-item tuple. This places the legend slightly to the right and slightly upward.<br>To place the legend outside the plot on the right-hand side, pass a value greater than or equal to 1 as the first item in the tuple.

# In[44]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(1.0,0.1));


# ## Pandas Datetime Index
# 
# We'll usually deal with time series as a datetime index when working with pandas dataframes. Fortunately pandas has a lot of functions and methods to work with time series!<br>
# For more on the pandas DatetimeIndex visit https://pandas.pydata.org/pandas-docs/stable/timeseries.html

# Ways to build a DatetimeIndex:

# In[45]:


# THE WEEK OF JULY 8TH, 2018
idx = pd.date_range('7/8/2018', periods=7, freq='D')
idx


# In[46]:


idx = pd.to_datetime(['Jan 01, 2018','1/2/18','03-Jan-2018',None])
idx


# In[47]:


# Create a NumPy datetime array
some_dates = np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[D]')
some_dates


# In[48]:


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

# In[55]:


# Index_col indicates that the index will be the column called 'Date'
# parse_dates, transforms the strings into datetime format

df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[56]:


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

# In[57]:


# Yearly Means
df.resample(rule='A').mean()


# Resampling rule 'A' takes all of the data points in a given year, applies the aggregation function (in this case we calculate the mean), and reports the result as the last day of that year.

# In[58]:


title = 'Monthly Max Closing Price for Starbucks'
df['Close'].resample('M').max().plot.bar(figsize=(16,6), title=title,color='#1f77b4');


# # Time Shifting
# 
# Sometimes you may need to shift all your data up or down along the time series index. In fact, a lot of pandas built-in methods do this under the hood. This isn't something we'll do often in the course, but it's definitely good to know about this anyways!

# In[60]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv',index_col='Date',parse_dates=True)


# ## .shift() forward
# This method shifts the entire date index a given number of rows, without regard for time periods (months & years).<br>It returns a modified copy of the original DataFrame.
# 
# In other words, it moves down all the rows down or up.

# In[63]:


# We move down all the rows
df.shift(1).head()


# In[64]:


# NOTE: You will lose that last piece of data that no longer has an index!
df.shift(1).tail()


# ## Shifting based on Time Series Frequency Code
# 
# We can choose to shift <em>index values</em> up or down without realigning the data by passing in a <strong>freq</strong> argument.<br>
# This method shifts dates to the next period based on a frequency code. Common codes are 'M' for month-end and 'A' for year-end. <br>Refer to the <em>Time Series Offset Aliases</em> table from the Time Resampling lecture for a full list of values, or click <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.<br>

# In[65]:


# Shift everything to the end of the month
df.shift(periods=1, freq='M').head()


# For more info on time shifting visit http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html<br>

# # Rolling and Expanding
# 
# A common process with time series is to create data based off of a rolling mean. The idea is to divide the data into "windows" of time, and then calculate an aggregate function for each window. In this way we obtain a <em>simple moving average</em>. Let's show how to do this easily with pandas!

# In[67]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[68]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# Now let's add in a rolling mean! This rolling method provides row entries, where every entry is then representative of the window. 

# In[69]:


# 7 day rolling mean
df.rolling(window=7).mean().head(15)


# In[70]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)
df.rolling(window=30).mean()['Close'].plot();


# ## Expanding
# 
# Instead of calculating values for a rolling window of dates, what if you wanted to take into account everything from the start of the time series up to each point in time? For example, instead of considering the average over the last 7 days, we would consider all prior data in our expanding set of averages.

# In[71]:



# Optional: specify a minimum number of periods to start from
df['Close'].expanding(min_periods=30).mean().plot(figsize=(12,5));


# # Visualizing Time Series Data

# In[97]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv',index_col='Date',parse_dates=True)


# First we'll create a line plot that puts both <tt>'Close'</tt> and <tt>'Volume'</tt> on the same graph.<br>Remember that we can use <tt>df.plot()</tt> in place of <tt>df.plot.line()</tt>

# In[98]:


df.index = pd.to_datetime(df.index)


# In[99]:


df.plot();


# ## Adding a title and axis labels

# In[88]:


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

# In[89]:


df['Close']['2017-01-01':'2017-03-01']


# ## X Limits
# There are two ways we can set a specific span of time as an x-axis limit. We can plot a slice of the dataset, or we can pass x-limit values as an argument into <tt>df.plot()</tt>.
# 
# The advantage of using a slice is that pandas automatically adjusts the y-limits accordingly.
# 
# The advantage of passing in arguments is that pandas automatically tightens the x-axis. Plus, if we're also setting y-limits this can improve readability.

# ### Choosing X Limits by Slice:

# In[90]:


# Dates are separated by a colon:
df['Close']['2017-01-01':'2017-03-01'].plot(figsize=(12,4)).autoscale(axis='x',tight=True);


# ### Choosing X Limits by Argument:

# In[91]:


# Dates are separated by a comma:
#Let's say we want to display the plot only from the 1st of january until the 1t of march
df['Close'].plot(figsize=(12,4),xlim=['2017-01-01','2017-03-01']);


# Now let's focus on the y-axis limits to get a better sense of the shape of the data.<br>First we'll find out what upper and lower limits to use.

# In[92]:


# FIND THE MINIMUM VALUE IN THE RANGE:
df.loc['2017-01-01':'2017-03-01']['Close'].min()


# ## Title and axis labels
# Let's add a title and axis labels to our subplot.
# <div class="alert alert-info"><strong>REMEMBER:</strong> <tt><font color=black>ax.autoscale(axis='both',tight=True)</font></tt> is unnecessary if axis limits have been passed into <tt>.plot()</tt>.<br>
# If we were to add it, autoscale would revert the axis limits to the full dataset.</div>

# In[93]:


title='Starbucks Closing Stock Prices'
ylabel='Closing Price (USD)'
xlabel='Closing Date'

ax = df['Close'].plot(xlim=['2017-01-04','2017-03-01'],ylim=[51,57],figsize=(12,4),title=title)
ax.set(xlabel=xlabel, ylabel=ylabel);


# We can pass arguments into <tt>.plot()</tt> to change the linestyle and color. Refer to the Customizing Plots lecture from the previous section for more options.

# In[94]:


df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],ls='--',c='r');


# ## X Ticks
# In this section we'll look at how to change the format and appearance of dates along the x-axis. To do this, we'll borrow a tool from <tt>matplotlib</tt> called <tt>dates</tt>.

# ### Set the spacing
# The x-axis values can be divided into major and minor axes. For now, we'll work only with the major axis and learn how to set the spacing with <tt>.set_major_locator()</tt>.
# As you can see in the graph below, 
# the X axis is not beautifully distributed

# In[95]:


# CREATE OUR AXIS OBJECT
ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57])


# With set_major_locator we can solve this problem

# In[101]:



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

# In[103]:


# USE THIS SPACE TO EXPERIMENT WITH DIFFERENT FORMATS
from datetime import datetime
datetime(2001, 2, 3, 16, 5, 6).strftime("%A, %B %d, %Y  %I:%M:%S %p")


# We use the set_major_formatter to format the display of the date in the plot

# In[104]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))


# ## Major vs. Minor Axis Values
# All of the tick marks we've used so far have belonged to the major axis. We can assign another level called the <em>minor axis</em>, perhaps to separate month names from days of the month.

# In[105]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],rot=0,title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))


# ## Adding Gridlines
# We can add x and y axis gridlines that extend into the plot from each major tick mark.

# In[106]:


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

# In[117]:


from statsmodels.tsa.filters.hp_filter import hpfilter
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/macrodata.csv',index_col=0,parse_dates=True)
df.head()


# In[116]:


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

# In[118]:


# Tuple unpacking
gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
#Both are a pandas.core.series.Series


# Let's plot the results

# In[124]:


gdp_cycle.plot();


# In[122]:


gdp_trend.plot();


# In[126]:


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

# In[128]:


from statsmodels.tsa.seasonal import seasonal_decompose

airline = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)

airline.dropna(inplace=True)

airline.plot();


# The graph above shows clearle that a additive model fits better. The number of passangers grows with time

# Using the seasonal_decompose from statsmodels we will decompose the time series into its components:

# In[130]:


result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')  # model='mul' also works
result.plot();


# In[132]:


#press on tab to check the different components
result.seasonal.plot();


# In[ ]:




