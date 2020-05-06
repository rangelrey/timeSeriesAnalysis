#!/usr/bin/env python
# coding: utf-8

# # Numpy Reminders: 
# Let's have a look at some of the most basic numpy concepts that we will use in the more advanced sections

# In[1]:


import numpy as np
import datetime as datetime
from matplotlib import dates
import pandas as pd


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


# # Customizing Pandas Plots
# In this section we'll show how to control the position and appearance of axis labels and legends.<br>
# For more info on the following topics visit https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# ## Colors, Widths and Linestyles
# The pandas <tt>.plot()</tt> method takes optional arguments that allow you to control linestyles, colors, widths and more.

# In[52]:


# START WITH A SIMPLE LINE PLOT
df2['c'].plot(figsize=(8,3));


# In[53]:


df2['c'].plot.line(ls='-.', c='r', lw='4', figsize=(8,3));


# For more on linestyles, click <a href='https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle'>here</a>.

# ## Adding Titles and Axis Labels

# In[54]:


# START WITH A SIMPLE MULTILINE PLOT
df2.plot(figsize=(8,3));


# ### Object-oriented plotting
# 
# When we call <tt>df.plot()</tt>, pandas returns a <tt>matplotlib.axes.AxesSubplot</tt> object. We can set labels
# on that object so long as we do it in the same jupyter cell. Setting an autoscale is done the same way.

# In[55]:


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

# In[56]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=1);


# We can pass a second argument, <tt>bbox_to_anchor</tt> that treats the value passed in through <tt>loc</tt> as an anchor point, and positions the legend along the x and y axes based on a two-value tuple.

# In[57]:


# FIRST, PLACE THE LEGEND IN THE LOWER-LEFT
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3);


# In[58]:


# NEXT, MOVE THE LEGEND A LITTLE TO THE RIGHT AND UP
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(0.1,0.1));


# ### Placing the Legend Outside the Plot
# In the above plot we passed <tt>(0.1,0.1)</tt> as our two-item tuple. This places the legend slightly to the right and slightly upward.<br>To place the legend outside the plot on the right-hand side, pass a value greater than or equal to 1 as the first item in the tuple.

# In[59]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(1.0,0.1));


# ## Pandas Datetime Index
# 
# We'll usually deal with time series as a datetime index when working with pandas dataframes. Fortunately pandas has a lot of functions and methods to work with time series!<br>
# For more on the pandas DatetimeIndex visit https://pandas.pydata.org/pandas-docs/stable/timeseries.html

# Ways to build a DatetimeIndex:

# In[60]:


# THE WEEK OF JULY 8TH, 2018
idx = pd.date_range('7/8/2018', periods=7, freq='D')
idx


# In[61]:


idx = pd.to_datetime(['Jan 01, 2018','1/2/18','03-Jan-2018',None])
idx


# In[62]:


# Create a NumPy datetime array
some_dates = np.array(['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[D]')
some_dates


# In[63]:


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

# In[64]:


# Index_col indicates that the index will be the column called 'Date'
# parse_dates, transforms the strings into datetime format

df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[65]:


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

# In[66]:


# Yearly Means
df.resample(rule='A').mean()


# Resampling rule 'A' takes all of the data points in a given year, applies the aggregation function (in this case we calculate the mean), and reports the result as the last day of that year.

# In[67]:


title = 'Monthly Max Closing Price for Starbucks'
df['Close'].resample('M').max().plot.bar(figsize=(16,6), title=title,color='#1f77b4');


# # Time Shifting
# 
# Sometimes you may need to shift all your data up or down along the time series index. In fact, a lot of pandas built-in methods do this under the hood. This isn't something we'll do often in the course, but it's definitely good to know about this anyways!

# In[68]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv',index_col='Date',parse_dates=True)


# ## .shift() forward
# This method shifts the entire date index a given number of rows, without regard for time periods (months & years).<br>It returns a modified copy of the original DataFrame.
# 
# In other words, it moves down all the rows down or up.

# In[69]:


# We move down all the rows
df.shift(1).head()


# In[70]:


# NOTE: You will lose that last piece of data that no longer has an index!
df.shift(1).tail()


# ## Shifting based on Time Series Frequency Code
# 
# We can choose to shift <em>index values</em> up or down without realigning the data by passing in a <strong>freq</strong> argument.<br>
# This method shifts dates to the next period based on a frequency code. Common codes are 'M' for month-end and 'A' for year-end. <br>Refer to the <em>Time Series Offset Aliases</em> table from the Time Resampling lecture for a full list of values, or click <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.<br>

# In[71]:


# Shift everything to the end of the month
df.shift(periods=1, freq='M').head()


# For more info on time shifting visit http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html<br>

# # Rolling and Expanding
# 
# A common process with time series is to create data based off of a rolling mean. The idea is to divide the data into "windows" of time, and then calculate an aggregate function for each window. In this way we obtain a <em>simple moving average</em>. Let's show how to do this easily with pandas!

# In[72]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv', index_col='Date', parse_dates=True)


# In[73]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True);


# Now let's add in a rolling mean! This rolling method provides row entries, where every entry is then representative of the window. 

# In[74]:


# 7 day rolling mean
df.rolling(window=7).mean().head(15)


# In[75]:


df['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True)
df.rolling(window=30).mean()['Close'].plot();


# ## Expanding
# 
# Instead of calculating values for a rolling window of dates, what if you wanted to take into account everything from the start of the time series up to each point in time? For example, instead of considering the average over the last 7 days, we would consider all prior data in our expanding set of averages.

# In[76]:



# Optional: specify a minimum number of periods to start from
df['Close'].expanding(min_periods=30).mean().plot(figsize=(12,5));


# # Visualizing Time Series Data

# In[77]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/starbucks.csv',index_col='Date',parse_dates=True)


# First we'll create a line plot that puts both <tt>'Close'</tt> and <tt>'Volume'</tt> on the same graph.<br>Remember that we can use <tt>df.plot()</tt> in place of <tt>df.plot.line()</tt>

# In[78]:


df.index = pd.to_datetime(df.index)


# In[79]:


df.plot();


# ## Adding a title and axis labels

# In[80]:


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

# In[81]:


df['Close']['2017-01-01':'2017-03-01']


# ## X Limits
# There are two ways we can set a specific span of time as an x-axis limit. We can plot a slice of the dataset, or we can pass x-limit values as an argument into <tt>df.plot()</tt>.
# 
# The advantage of using a slice is that pandas automatically adjusts the y-limits accordingly.
# 
# The advantage of passing in arguments is that pandas automatically tightens the x-axis. Plus, if we're also setting y-limits this can improve readability.

# ### Choosing X Limits by Slice:

# In[82]:


# Dates are separated by a colon:
df['Close']['2017-01-01':'2017-03-01'].plot(figsize=(12,4)).autoscale(axis='x',tight=True);


# ### Choosing X Limits by Argument:

# In[83]:


# Dates are separated by a comma:
#Let's say we want to display the plot only from the 1st of january until the 1t of march
df['Close'].plot(figsize=(12,4),xlim=['2017-01-01','2017-03-01']);


# Now let's focus on the y-axis limits to get a better sense of the shape of the data.<br>First we'll find out what upper and lower limits to use.

# In[84]:


# FIND THE MINIMUM VALUE IN THE RANGE:
df.loc['2017-01-01':'2017-03-01']['Close'].min()


# ## Title and axis labels
# Let's add a title and axis labels to our subplot.
# <div class="alert alert-info"><strong>REMEMBER:</strong> <tt><font color=black>ax.autoscale(axis='both',tight=True)</font></tt> is unnecessary if axis limits have been passed into <tt>.plot()</tt>.<br>
# If we were to add it, autoscale would revert the axis limits to the full dataset.</div>

# In[85]:


title='Starbucks Closing Stock Prices'
ylabel='Closing Price (USD)'
xlabel='Closing Date'

ax = df['Close'].plot(xlim=['2017-01-04','2017-03-01'],ylim=[51,57],figsize=(12,4),title=title)
ax.set(xlabel=xlabel, ylabel=ylabel);


# We can pass arguments into <tt>.plot()</tt> to change the linestyle and color. Refer to the Customizing Plots lecture from the previous section for more options.

# In[86]:


df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],ls='--',c='r');


# ## X Ticks
# In this section we'll look at how to change the format and appearance of dates along the x-axis. To do this, we'll borrow a tool from <tt>matplotlib</tt> called <tt>dates</tt>.

# ### Set the spacing
# The x-axis values can be divided into major and minor axes. For now, we'll work only with the major axis and learn how to set the spacing with <tt>.set_major_locator()</tt>.
# As you can see in the graph below, 
# the X axis is not beautifully distributed

# In[87]:


# CREATE OUR AXIS OBJECT
ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57])


# With set_major_locator we can solve this problem

# In[88]:



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

# In[89]:


# USE THIS SPACE TO EXPERIMENT WITH DIFFERENT FORMATS
from datetime import datetime
datetime(2001, 2, 3, 16, 5, 6).strftime("%A, %B %d, %Y  %I:%M:%S %p")


# We use the set_major_formatter to format the display of the date in the plot

# In[90]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))


# ## Major vs. Minor Axis Values
# All of the tick marks we've used so far have belonged to the major axis. We can assign another level called the <em>minor axis</em>, perhaps to separate month names from days of the month.

# In[91]:


ax = df['Close'].plot(xlim=['2017-01-01','2017-03-01'],ylim=[51,57],rot=0,title='2017 Starbucks Closing Stock Prices')
ax.set(xlabel='')

ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))

ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))


# ## Adding Gridlines
# We can add x and y axis gridlines that extend into the plot from each major tick mark.

# In[92]:


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

# In[93]:


from statsmodels.tsa.filters.hp_filter import hpfilter
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/macrodata.csv',index_col=0,parse_dates=True)
df.head()


# In[94]:


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

# In[95]:


# Tuple unpacking
gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
#Both are a pandas.core.series.Series


# Let's plot the results

# In[96]:


gdp_cycle.plot();


# In[97]:


gdp_trend.plot();


# In[98]:


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

# In[99]:


from statsmodels.tsa.seasonal import seasonal_decompose

airline = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)

airline.dropna(inplace=True)

airline.plot();


# The graph above shows clearle that a additive model fits better. The number of passangers grows with time

# Using the seasonal_decompose from statsmodels we will decompose the time series into its components:

# In[100]:


result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')  # model='mul' also works
result.plot();


# In[101]:


#press on tab to check the different components
result.seasonal.plot();


# # SMA
# ## Simple Moving Average
# 
# We've already shown how to create a <a href='https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average'>simple moving average</a> by applying a <tt>mean</tt> function to a rolling window.
# 
# For a quick review

# In[102]:


airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()


# In[103]:


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

# In[104]:


# With span = 12, we are using al alpha = 2/(12+1). Look at the above formula.
# Therefore when we pass the parameter span in the formula, we are setting alpha
airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()
airline[['Thousands of Passengers','EWMA12']].plot();


# In[105]:


airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()
airline[['Thousands of Passengers','EWMA12']].plot();


# ## SMA vs EWMA

# In[106]:


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

# In[107]:


#We will keep using the arilines dataframe
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)

df.dropna(inplace=True)


# Note that our DatetimeIndex does not have a frequency. In order to build a Holt-Winters smoothing model, statsmodels needs to know the frequency of the data (whether it's daily, monthly etc.). Since observations occur at the start of each month, we'll use MS.<br>A full list of time series offset aliases can be found <a href='http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'>here</a>.

# In[108]:


df.index


# In[109]:


df.index.freq = 'MS'
df.index


# In[110]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span = 12
alpha = 2/(span+1)
#as we did in the previous cells
df['EWMA12'] = df['Thousands of Passengers'].ewm(alpha=alpha,adjust=False).mean()


# Now we are going to implement the SIMPLE EXP SMOOTHING model, but first let's understand what is going on with the code:

# In[111]:


model = SimpleExpSmoothing(df['Thousands of Passengers'])


# In[112]:


# With Tab you can check the various attributes of the object "model", like "fit"
# As paremeters in the fit method we have the smoothing_level which is alpha
fitted_model =  model.fit(smoothing_level=alpha,optimized=False)

# This returns a HoltWintersResultsWrapper

fitted_model.fittedvalues


# <div class="alert alert-danger"><strong>NOTE:</strong> For some reason, when <tt>optimized=False</tt> is passed into <tt>.fit()</tt>, the statsmodels <tt>SimpleExpSmoothing</tt> function shifts fitted values down one row. We fix this by adding <tt>.shift(-1)</tt> after <tt>.fittedvalues</tt></div>

# In[113]:


# We will simply solve the problem with the shift function

fitted_model.fittedvalues.shift(-1)


# In[114]:


# we add the predictions to our dataframe

df['SES12'] = fitted_model.fittedvalues.shift(-1)


# In[115]:


df.head()


# Note that we could have simply put all the code in the cells into one line of code:

# In[116]:


#df['SES12']=SimpleExpSmoothing(df['Thousands of Passengers']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)


# In[117]:


df


# In[118]:


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

# In[119]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

#It's hard to say wethet the passengers data has a multiplicative or additive trend, we will assume it is additive


df['DESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='add').fit().fittedvalues.shift(-1)
df.head()


# The DESadd12 is almost the same as the number of passangers (the blue line is just behind the red)

# In[120]:


df.plot(figsize=(12,5))


# In[121]:


# Let's do some Zoom
df.iloc[:24].plot(figsize=(12,5))


# Double exponential is clearly better than Simple

# What about Tripple ?

# ___
# ## Triple Exponential Smoothing - Holt-Winters Method
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data. 
# 
# 

# In[122]:


df['TESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df.head()


# In[123]:


df['TESmul12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
df.head()


# In[124]:


df[['Thousands of Passengers','TESadd12','TESmul12']].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# In[125]:


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

# In[126]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()


# ### Train Test Split

# In[127]:


train_data = df.iloc[:109] # Goes up to but not including 109
test_data = df.iloc[108:]


# ### Fitting the Model

# In[128]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# ### Evaluating model against test

# In[129]:


train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8));


# In[130]:


test_predictions = fitted_model.forecast(36).rename('HW Forecast')


# In[131]:


test_predictions.plot(legend=True, label="Prediction")


# In[132]:


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

# In[133]:


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

# In[134]:


mean_absolute_error(test_data, test_predictions)


# To analyse the MAE, you can check the summary key figures of the test data

# In[135]:


test_data.describe()


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

# In[136]:


np.sqrt(mean_squared_error(test_data,test_predictions))


# Our mean squarred error is less than our standard deviation. Therefore, the method is not doing a bad job

# ### Forecasting Future Data
# We will use the whole dataset now to predict the future values

# In[137]:


#We creat the fitted model object
fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

forecast_predictions= fitted_model.forecast(36)

df['Thousands of Passengers'].plot(figsize=(12,8))
forecast_predictions.plot();


# In orange we have the predictions. Pretty accurate.

# ## Stationarity
# Time series data is said to be <em>stationary</em> if it does <em>not</em> exhibit trends or seasonality. That is, fluctuations in the data are entirely due to outside forces and noise. The file <tt>samples.csv</tt> contains made-up datasets that illustrate stationary and non-stationary data.
# 
# 

# In[138]:


df2 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/samples.csv',index_col=0,parse_dates=True)
df2.head()


# In[139]:


df2['a'].plot(ylim=[0,100],title="STATIONARY DATA").autoscale(axis='x',tight=True);


# In[140]:


df2['b'].plot(ylim=[0,100],title="NON-STATIONARY DATA").autoscale(axis='x',tight=True);


# ## Differencing
# Non-stationary data can be made to look stationary through <em>differencing</em>. A simple differencing method calculates the difference between consecutive points.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.diff.html'>statespace.tools.diff</a></strong><font color=black>(series[, k_diff, …])</font>&nbsp;&nbsp;Difference a series simply and/or seasonally along the zero-th axis.</tt></div>
# 
# You can calculate manually the first order difference by substracting:

# In[141]:


df2["b"]-df2["b"].shift(1)


# But statsmodels has a function that does it for you

# In[142]:


from statsmodels.tsa.statespace.tools import diff
df2['d1'] = diff(df2['b'],k_diff=1)

df2['d1'].plot(title="FIRST DIFFERENCE DATA").autoscale(axis='x',tight=True);


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
# 
# For times series the covariance (autocovariance) will be:
# ${\displaystyle \gamma_k = \frac 1 n \sum\limits_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k}-\bar{y})}$
# 
# ### Autocovariance Example:
# Say we have a time series with five observations: {13, 5, 11, 12, 9}.<br>
# We can quickly see that $n = 5$, the mean $\bar{y} = 10$, and we'll see that the variance $\sigma^2 = 8$.<br>
# The following calculations give us our covariance values:
# <br><br>
# $\gamma_0 = \frac {(13-10)(13-10)+(5-10)(5-10)+(11-10)(11-10)+(12-10)(12-10)+(9-10)(9-10)} 5 = \frac {40} 5 = 8.0 \\
# \gamma_1 = \frac {(13-10)(5-10)+(5-10)(11-10)+(11-10)(12-10)+(12-10)(9-10)} 5 = \frac {-20} 5 = -4.0 \\
# \gamma_2 = \frac {(13-10)(11-10)+(5-10)(12-10)+(11-10)(9-10)} 5 = \frac {-8} 5 = -1.6 \\
# \gamma_3 = \frac {(13-10)(12-10)+(5-10)(9-10)} 5 = \frac {11} 5 = 2.2 \\
# \gamma_4 = \frac {(13-10)(9-10)} 5 = \frac {-3} 5 = -0.6$
# <br><br>
# Note that $\gamma_0$ is just the population variance $\sigma^2$
# 
# Let's see if statsmodels gives us the same results! For this we'll create a <strong>fake</strong> dataset:
# 

# In[143]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
# Import the models we'll be using in this section
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols
from pandas.plotting import lag_plot


import warnings
warnings.filterwarnings("ignore")
df = pd.DataFrame({'a':[13, 5, 11, 12, 9]})
arr = acovf(df['a'])
arr


# ### Unbiased Autocovariance
# Note that the number of terms in the calculations above are decreasing.<br>Statsmodels can return an "unbiased" autocovariance where instead of dividing by $n$ we divide by $n-k$.
# 
# $\gamma_0 = \frac {(13-10)(13-10)+(5-10)(5-10)+(11-10)(11-10)+(12-10)(12-10)+(9-10)(9-10)} {5-0} = \frac {40} 5 = 8.0 \\
# \gamma_1 = \frac {(13-10)(5-10)+(5-10)(11-10)+(11-10)(12-10)+(12-10)(9-10)} {5-1} = \frac {-20} 4 = -5.0 \\
# \gamma_2 = \frac {(13-10)(11-10)+(5-10)(12-10)+(11-10)(9-10)} {5-2} = \frac {-8} 3 = -2.67 \\
# \gamma_3 = \frac {(13-10)(12-10)+(5-10)(9-10)} {5-3} = \frac {11} 2 = 5.5 \\
# \gamma_4 = \frac {(13-10)(9-10)} {5-4} = \frac {-3} 1 = -3.0$

# In[144]:


arr2 = acovf(df['a'],unbiased=True)
arr2


# ## Autocorrelation for 1D
# The correlation $\rho$ (rho) between two variables $y_1,y_2$ is given as:
# 
# ### $\rho = \frac {\operatorname E[(y_1−\mu_1)(y_2−\mu_2)]} {\sigma_{1}\sigma_{2}} = \frac {\operatorname {Cov} (y_1,y_2)} {\sigma_{1}\sigma_{2}}$,
# 
# where $E$ is the expectation operator, $\mu_{1},\sigma_{1}$ and $\mu_{2},\sigma_{2}$ are the means and standard deviations of $y_1$ and $y_2$.
# 
# When working with a single variable (i.e. <em>autocorrelation</em>) we would consider $y_1$ to be the original series and $y_2$ a lagged version of it. Note that with autocorrelation we work with $\bar y$, that is, the full population mean, and <em>not</em> the means of the reduced set of lagged factors (see note below).
# 
# Thus, the formula for $\rho_k$ for a time series at lag $k$ is:
# 
# ${\displaystyle \rho_k = \frac {\sum\limits_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k}-\bar{y})} {\sum\limits_{t=1}^{n} (y_t - \bar{y})^2}}$
# 
# This can be written in terms of the covariance constant $\gamma_k$ as:
# 
# ${\displaystyle \rho_k = \frac {\gamma_k n} {\gamma_0 n} = \frac {\gamma_k} {\sigma^2}}$
# 
# For example,<br>
# $\rho_4 = \frac {\gamma_4} {\sigma^2} = \frac{-0.6} {8} = -0.075$
# 
# Note that ACF values are bound by -1 and 1. That is, ${\displaystyle -1 \leq \rho_k \leq 1}$

# In[145]:


arr3 = acf(df['a'])
arr3


# ## Partial Autocorrelation
# Partial autocorrelations measure the linear dependence of one variable after removing the effect of other variable(s) that affect both variables. That is, the partial autocorrelation at lag $k$ is the autocorrelation between $y_t$ and $y_{t+k}$ that is not accounted for by lags $1$ through $k−1$.
# 
# A common method employs the non-recursive <a href='https://en.wikipedia.org/wiki/Autoregressive_model#Calculation_of_the_AR_parameters'>Yule-Walker Equations</a>:
# 
# $\phi_0 = 1\\
# \phi_1 = \rho_1 = -0.50\\
# \phi_2 = \frac {\rho_2 - {\rho_1}^2} {1-{\rho_1}^2} = \frac {(-0.20) - {(-0.50)}^2} {1-{(-0.50)}^2}= \frac {-0.45} {0.75} = -0.60$
# 
# As $k$ increases, we can solve for $\phi_k$ using matrix algebra and the <a href='https://en.wikipedia.org/wiki/Levinson_recursion'>Levinson–Durbin recursion</a> algorithm which maps the sample autocorrelations $\rho$ to a <a href='https://en.wikipedia.org/wiki/Toeplitz_matrix'>Toeplitz</a> diagonal-constant matrix. The full solution is beyond the scope of this course, but the setup is as follows:
# 
# 
# $\displaystyle \begin{pmatrix}\rho_0&\rho_1&\cdots &\rho_{k-1}\\
# \rho_1&\rho_0&\cdots &\rho_{k-2}\\
# \vdots &\vdots &\ddots &\vdots \\
# \rho_{k-1}&\rho_{k-2}&\cdots &\rho_0\\
# \end{pmatrix}\quad \begin{pmatrix}\phi_{k1}\\\phi_{k2}\\\vdots\\\phi_{kk}\end{pmatrix}
# \mathbf = \begin{pmatrix}\rho_1\\\rho_2\\\vdots\\\rho_k\end{pmatrix}$

# In[146]:


# it's 4 because we have 4 lags
# mle stands for maximum likelihood estimation --> this uses the biased estimated coefficients
# yw stands for yule-walker equations
arr4 = pacf_yw(df['a'],nlags=4,method='mle')
arr4


# <div class="alert alert-info"><strong>NOTE:</strong> We passed in <tt><font color=black>method='mle'</font></tt> above in order to use biased ACF coefficients. "mle" stands for "maximum likelihood estimation". Alternatively we can pass <tt>method='unbiased'</tt> (the statsmodels default):</div>

# In[147]:


arr5 = pacf_yw(df['a'],nlags=4,method='unbiased')  
arr5


# instead of YW you can use as well OLS

# In[148]:


arr6 = pacf_ols(df['a'],nlags=4)
arr6


# # Plotting
# The arrays returned by <tt>.acf(df)</tt> and <tt>.pacf_yw(df)</tt> show the magnitude of the autocorrelation for a given $y$ at time $t$. Before we look at plotting arrays, let's look at the data itself for evidence of autocorrelation.
# 
# Pandas has a built-in plotting function that plots increasing $y_t$ values on the horizontal axis against lagged versions of the values $y_{t+1}$ on the vertical axis. If a dataset is non-stationary with an upward trend, then neighboring values should trend in the same way. Let's look at the <strong>Airline Passengers</strong> dataset first.

# In[149]:



# Load a non-stationary dataset
df1 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'

# Load a stationary dataset
df2 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'


# Let's look at a lag plot of the non-stationary data set (the airline)

# In[150]:



lag_plot(df1['Thousands of Passengers']);


# This shows evidence of very strong autocorrelation

# Let's look at a lag plot of the stationary data set (the births)

# In[151]:


lag_plot(df2['Births']);


# There is no autocorrelation. The number of births of today is not correlated with the number of births from yesterday

# ## ACF Plots
# Plotting the magnitude of the autocorrelations over the first few (20-40) lags can say a lot about a time series.
# 
# For example, consider the stationary <strong>Daily Total Female Births</strong> dataset:

# In[152]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[153]:


# Now let's plot the autocorrelation at different lags
title = 'Autocorrelation: Airline Passengers'
lags = 40
plot_acf(df1,title=title,lags=lags);


# There is clearly signs os seasonality in the data.
# 
# The shaded region is a 95% confidence interval. Which means that those points (those lags) outside the blue shaded area are more like to be correlated with the current year. While the higher the lag, the less likely is that there is signiciant correlation

# In[173]:



title = 'Autocorrelation: Daily Female Births'
lags = 40
plot_acf(df2,title=title,lags=lags);


# # AR(p)
# # Autoregressive Model
# In a moving average model as we saw with Holt-Winters, we forecast the variable of interest using a linear combination of predictors. In our example we forecasted numbers of airline passengers in thousands based on a set of level, trend and seasonal predictors.
# 
# In an autoregression model, we forecast using a linear combination of <em>past values</em> of the variable. The term <em>autoregression</em> describes a regression of the variable against itself. An autoregression is run against a set of <em>lagged values</em> of order $p$.
# 

# 
# ### $y_{t} = c + \phi_{1}y_{t-1} + \phi_{2}y_{t-2} + \dots + \phi_{p}y_{t-p} + \varepsilon_{t}$
# 
# where $c$ is a constant, $\phi_{1}$ and $\phi_{2}$ are lag coefficients up to order $p$, and $\varepsilon_{t}$ is white noise.

# 
# For example, an <strong>AR(1)</strong> model would follow the formula
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = c + \phi_{1}y_{t-1} + \varepsilon_{t}$
# 
# whereas an <strong>AR(2)</strong> model would follow the formula
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = c + \phi_{1}y_{t-1} + \phi_{2}y_{t-2} + \varepsilon_{t}$
# 
# and so on.
# 
# Note that the lag coeffients are usually less than one, as we usually restrict autoregressive models to stationary data.<br>
# Specifically, for an <strong>AR(1)</strong> model: $-1 \lt \phi_1 \lt 1$<br>
# and for an <strong>AR(2)</strong> model: $-1 \lt \phi_2 \lt 1, \ \phi_1 + \phi_2 \lt 1, \ \phi_2 - \phi_1 \lt 1$<br>
# 
# Models <strong>AR(3)</strong> and higher become mathematically very complex. Fortunately statsmodels does all the heavy lifting for us.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AR.html'>ar_model.AR</a></strong><font color=black>(endog[, dates, freq, missing])</font>&nbsp;&nbsp;Autoregressive AR(p) model<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.ARResults.html'>ar_model.ARResults</a></strong><font color=black>(model, params[, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Class to hold results from fitting an AR model</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://otexts.com/fpp2/AR.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Autoregressive models</font><br>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Autoregressive_model'>Wikipedia</a></strong>&nbsp;&nbsp;<font color=black>Autoregressive model</font></div>

# ## Perform standard imports and load datasets
# For this exercise we'll look at monthly U.S. population estimates in thousands from January 2011 to December 2018 (96 records, 8 years of data). Population includes resident population plus armed forces overseas. The monthly estimate is the average of estimates for the first of the month and the first of the following month.
# Source: https://fred.stlouisfed.org/series/POPTHM

# In[155]:


# Load specific forecasting tools
from statsmodels.tsa.ar_model import AR,ARResults

# Load the U.S. Population dataset
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/uspopulation.csv',index_col='DATE',parse_dates=True)
df.index.freq = 'MS'


# In[156]:


title='U.S. Monthly Population Estimates'
ylabel='Pop. Est. (thousands)'
xlabel='' # we don't really need a label here

ax = df['PopEst'].plot(figsize=(12,5),title=title);
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# ## Split the data into train/test sets
# The goal in this section is to:
# * Split known data into a training set of records on which to fit the model
# * Use the remaining records for testing, to evaluate the model
# * Fit the model again on the <em>full</em> set of records
# * Predict a future set of values using the model
# 
# As a general rule you should set the length of your test set equal to your intended forecast size. That is, for a monthly dataset you might want to forecast out one more year. Therefore your test set should be one year long.
# 
# <div class="alert alert-info"><strong>NOTE: </strong>For many training and testing applications we would use the <tt>train_test_split()</tt> function available from Python's <a href='https://scikit-learn.org/stable/'>scikit-learn</a> library. This won't work here as <tt>train_test_split()</tt> takes <em>random samples</em> of data from the population.</div>

# Using the previous month to predict is not perfect, but still we get an acceptable result

# In[173]:


# Set one year for testing
train = df.iloc[:84]
test = df.iloc[84:]


# In[175]:


model = AR(train['PopEst'])
AR1fit = model.fit(maxlag=1)
print(f'Lag: {AR1fit.k_ar}')
print(f'Coefficients:\n{AR1fit.params}')

# This is the general format for obtaining predictions
start=len(train)
end=len(train)+len(test)-1
predictions1 = AR1fit.predict(start=start, end=end, dynamic=False).rename('AR(1) Predictions')

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True,figsize=(12,6));


# ### AR(2)

# In[179]:


# Recall that our model was already created above based on the training set
model2 = AR(train['PopEst'])
AR2fit = model2.fit(maxlag=2)
print(f'Lag: {AR2fit.k_ar}')
print(f'Coefficients:\n{AR2fit.params}')

start=len(train)
end=len(train)+len(test)-1
predictions2 = AR2fit.predict(start=start, end=end, dynamic=False).rename('AR(2) Predictions')

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True)
predictions2.plot(legend=True,figsize=(12,6));


# How do we actually find the best number of lags? We can optimize for "p" (number of lags)
# 
# If we leave the maxlg empty, statsmodels tries to find the optimal value. 

# In[184]:


modelp = AR(train['PopEst'])

ARfit = modelp.fit()
print(f'Lag: {ARfit.k_ar}')
print(f'Coefficients:\n{ARfit.params}')

predictionsp = ARfit.predict(start=start, end=end, dynamic=False).rename('AR(p) Predictions')

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True)
predictions2.plot(legend=True)
predictionsp.plot(legend=True,figsize=(12,6));


# But we can see that it actually has similar results as AR(2)
# 
# What we can do is to change the criterion that statsmodels uses to determine p. We will use the t-stat criterion

# In[187]:


modelpp = AR(train['PopEst'])

ARfit = modelpp.fit(ic='t-stat')
print(f'Lag: {ARfit.k_ar}')
print(f'Coefficients:\n{ARfit.params}')

predictionspp = ARfit.predict(start=start, end=end, dynamic=False).rename('AR(pp) Predictions')

test['PopEst'].plot(legend=True)
predictionsp.plot(legend=True)
predictionspp.plot(legend=True,figsize=(12,6));


# That looks much better. 
# But let's look at the mean squared error to see which one has a lowe error

# In[193]:


preds = [predictions1, predictions2, predictionsp, predictionspp]
labels = ['AR(1)','AR(2)','AR(p)', 'AR(pp)']
for i in range(len(preds)):
    error = mean_squared_error(test['PopEst'],preds[i])
    print(f'{labels[i]} MSE was :{error}')


# The last model, which used the t-stat as criterion is the one with the lower MSE

# ## Forecasting
# Now we're ready to train our best model on the greatest amount of data, and fit it to future dates. Using the last model
# 

# In[194]:


# First, retrain the model on the full dataset
model = AR(df['PopEst'])

# Next, fit the model
ARfit = model.fit(ic='t-stat')

# Make predictions
fcast = ARfit.predict(start=len(df), end=len(df)+12, dynamic=False).rename('Forecast')

# Plot the results
df['PopEst'].plot(legend=True)
fcast.plot(legend=True,figsize=(12,6));


# # Descriptive Statistics and Tests
# In upcoming sections we'll talk about different forecasting models like ARMA, ARIMA, Seasonal ARIMA and others. Each model addresses a different type of time series. For this reason, in order to select an appropriate model we need to know something about the data.
# 
# In this section we'll learn how to determine if a time series is <em>stationary</em>, if it's <em>independent</em>, and if two series demonstrate <em>correlation</em> and/or <em>causality</em>.
# 
# * Goals
#   * Be able to perform Augmented Dickey Fuller Test
#   * Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
#   * Calculate the BDS test statistic for independence of a time series
#   * Return’s Ljung-Box Q Statistic
#   * four tests for granger non-causality of 2 timeseries (maybe do this tests on two airline stocks against each other, or gas price versus airline stock/travel costs)

# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccovf.html'>stattools.ccovf</a></strong><font color=black>(x, y[, unbiased, demean])</font>&nbsp;&nbsp;crosscovariance for 1D<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html'>stattools.ccf</a></strong><font color=black>(x, y[, unbiased])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cross-correlation function for 1d<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.periodogram.html'>stattools.periodogram</a></strong><font color=black>(X)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns the periodogram for the natural frequency of X<br>
#     
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html'>stattools.adfuller</a></strong><font color=black>(x[, maxlag, regression, …])</font>&nbsp;&nbsp;Augmented Dickey-Fuller unit root test<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html'>stattools.kpss</a></strong><font color=black>(x[, regression, lags, store])</font>&nbsp;&nbsp;&nbsp;&nbsp;Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html'>stattools.coint</a></strong><font color=black>(y0, y1[, trend, method, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Test for no-cointegration of a univariate equation<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.bds.html'>stattools.bds</a></strong><font color=black>(x[, max_dim, epsilon, distance])</font>&nbsp;&nbsp;Calculate the BDS test statistic for independence of a time series<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.q_stat.html'>stattools.q_stat</a></strong><font color=black>(x, nobs[, type])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns Ljung-Box Q Statistic<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html'>stattools.grangercausalitytests</a></strong><font color=black>(x, maxlag[, …])</font>&nbsp;Four tests for granger non-causality of 2 timeseries<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.levinson_durbin.html'>stattools.levinson_durbin</a></strong><font color=black>(s[, nlags, isacov])</font>&nbsp;&nbsp;&nbsp;Levinson-Durbin recursion for autoregressive processes<br>
# 
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.mse.html'>stattools.eval_measures.mse</a></strong><font color=black>(x1, x2, axis=0)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mean squared error<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.rmse.html'>stattools.eval_measures.rmse</a></strong><font color=black>(x1, x2, axis=0)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;root mean squared error<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.meanabs.html'>stattools.eval_measures.meanabs</a></strong><font color=black>(x1, x2, axis=0)</font>&nbsp;&nbsp;mean absolute error<br>
# </tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test'>Wikipedia:</a></strong>&nbsp;&nbsp;<font color=black>Augmented Dickey–Fuller test</font><br>
# <strong>
# <a href='https://otexts.com/fpp2/accuracy.html'>Forecasting: Principles and Practice:</a></strong>&nbsp;&nbsp;<font color=black>Evaluating forecast accuracy</font>
# 
# </div>

# In[195]:



# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load a seasonal dataset
df1 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'

# Load a nonseasonal dataset
df2 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'

from statsmodels.tsa.stattools import ccovf,ccf,periodogram

from statsmodels.tsa.stattools import adfuller,kpss,coint,bds,q_stat,grangercausalitytests,levinson_durbin

from statsmodels.tools.eval_measures import mse, rmse, meanabs


# # Tests for Stationarity
# A time series is <em>stationary</em> if the mean and variance are fixed between any two equidistant points. That is, no matter where you take your observations, the results should be the same. A times series that shows seasonality is <em>not</em> stationary.
# 
# A test for stationarity usually involves a [unit root](https://en.wikipedia.org/wiki/Unit_root_test) hypothesis test, where the null hypothesis $H_0$ is that the series is <em>nonstationary</em>, and contains a unit root. The alternate hypothesis $H_1$ supports stationarity. The augmented Dickey-Fuller Test is one such test. 
# 
# ## Augmented Dickey-Fuller Test
# To determine whether a series is stationary we can use the [augmented Dickey-Fuller Test](https://en.wikipedia.org/wiki/Augmented_Dickey-Fuller_test). In this test the null hypothesis states that $\phi = 1$ (this is also called a unit test). The test returns several statistics we'll see in a moment. Our focus is on the p-value. A small p-value ($p<0.05$) indicates strong evidence against the null hypothesis.
# 
# To demonstrate, we'll use a dataset we know is <em>not</em> stationary, the airline_passenger dataset. First, let's plot the data along with a 12-month rolling mean and standard deviation:

# In[196]:


df1['12-month-SMA'] = df1['Thousands of Passengers'].rolling(window=12).mean()
df1['12-month-Std'] = df1['Thousands of Passengers'].rolling(window=12).std()

df1[['Thousands of Passengers','12-month-SMA','12-month-Std']].plot();


# Not only is this dataset seasonal with a clear upward trend, the standard deviation increases over time as well.

# In[197]:


print('Augmented Dickey-Fuller Test on Airline Data')
# Note that autolag = AIC, Akaike Information Criteria method
dftest = adfuller(df1['Thousands of Passengers'],autolag='AIC')
dftest


# To understand what those values mean, you can also check more information with the help() function

# In[200]:


help(adfuller)


# Since this is a bit annoying, we can customize it for our own use

# In[201]:


print('Augmented Dickey-Fuller Test on Airline Data')

dfout = pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])

for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val
print(dfout)


# Here we have a very high p-value at 0.99, which provides weak evidence against the null hypothesis, and so we <em>fail to reject</em> the null hypothesis, and decide that our dataset is not stationary.<br>
# Note: in statistics we don't "accept" a null hypothesis - nothing is ever truly proven - we just fail to reject it.
# <br><br>
# Now let's apply the ADF test to stationary data with the Daily Total Female Births dataset.

# In[202]:


df2['30-Day-SMA'] = df2['Births'].rolling(window=30).mean()
df2['30-Day-Std'] = df2['Births'].rolling(window=30).std()

df2[['Births','30-Day-SMA','30-Day-Std']].plot();


# In[203]:


print('Augmented Dickey-Fuller Test on Daily Female Births')
dftest = adfuller(df2['Births'],autolag='AIC')
dfout = pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])

for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val
print(dfout)


# In this case our p-value is very low at 0.000052, and we do reject the null hypothesis. This dataset appears to have no unit root, and is stationary.

# ### Function for running the augmented Dickey-Fuller test
# Since we'll use it frequently in the upcoming forecasts, let's define a function we can copy into future notebooks for running the augmented Dickey-Fuller test. Remember that we'll still have to import <tt>adfuller</tt> at the top of our notebook.

# In[17]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# Let's run this function:
# 

# In[206]:


adf_test(df1['Thousands of Passengers'])


# In[207]:


adf_test(df2['Births'])


# ### Granger Causality Tests

# The Granger Cauality test tells us wether a series can be considered to be able to forecast another series.
# It will test it with various tests such as F-test and chi2-test

# In[ ]:


df3 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/samples.csv',index_col=0, parse_dates=True)
df3.index.freq = 'MS'
df3.head()


# In[218]:


grangercausalitytests(df3[['a','c']],maxlag=3)


# We need to interpret de p-values and since all of them are >0.05, we can say there is no time-causality in the time series

# # Evaluating forecast accuracy
# Two calculations related to linear regression are <a href='https://en.wikipedia.org/wiki/Mean_squared_error'><strong>mean squared error</strong></a> (MSE) and <a href='https://en.wikipedia.org/wiki/Root-mean-square_deviation'><strong>root mean squared error</strong></a> (RMSE)
# 
# The formula for the mean squared error is<br><br>
# &nbsp;&nbsp;&nbsp;&nbsp;$MSE = {\frac 1 L} \sum\limits_{l=1}^L (y_{T+l} - \hat y_{T+l})^2$<br><br>
# where $T$ is the last observation period and $l$ is the lag point up to $L$ number of test observations.
# 
# The formula for the root mean squared error is<br><br>
# &nbsp;&nbsp;&nbsp;&nbsp;$RMSE = \sqrt{MSE} = \sqrt{{\frac 1 L} \sum\limits_{l=1}^L (y_{T+l} - \hat y_{T+l})^2}$<br><br>
# 
# The advantage of the RMSE is that it is expressed in the same units as the data.<br><br>
# 
# A method similar to the RMSE is the <a href='https://en.wikipedia.org/wiki/Mean_absolute_error'><strong>mean absolute error</strong></a> (MAE) which is the mean of the magnitudes of the error, given as<br><br>
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$MAE = {\frac 1 L} \sum\limits_{l=1}^L \mid{y_{T+l}} - \hat y_{T+l}\mid$<br><br>
# 
# A forecast method that minimizes the MAE will lead to forecasts of the median, while minimizing the RMSE will lead to forecasts of the mean.

# let's create some random data

# In[219]:


np.random.seed(42)
df = pd.DataFrame(np.random.randint(20,30,(50,2)),columns=['test','predictions'])
df.plot(figsize=(12,4));


# In[220]:


df.head()


# In[221]:


from statsmodels.tools.eval_measures import mse,rmse,meanabs


# Let's use the most used metric for evaluating prediction accuracy

# In[222]:


rmse(df['test'],df['predictions'])


# In[224]:


df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'


# let's take the data set and check monthly data

# In[226]:


from statsmodels.graphics.tsaplots import month_plot, quarter_plot

month_plot(df['Thousands of Passengers']);


# This displays the range of values and their means (from the whole data series).
# 
# To check the quarter data, we will use:

# In[230]:


dfq = df["Thousands of Passengers"].resample(rule='Q').mean()

quarter_plot(dfq);
# # Choosing ARIMA Orders
# 
# * Goals
#   * Understand PDQ terms for ARIMA (slides)
#   * Understand how to choose orders manually from ACF and PACF
#   * Understand how to use automatic order selection techniques using the functions below
#   
# Before we can apply an ARIMA forecasting model, we need to review the components of one.<br>
# ARIMA, or Autoregressive Independent Moving Average is actually a combination of 3 models:
# * <strong>AR(p)</strong> Autoregression - a regression model that utilizes the dependent relationship between a current observation and observations over a previous period.

# * <strong>I(d)</strong> Integration - uses differencing of observations (subtracting an observation from an observation at the previous time step) in order to make the time series stationary. In other words how many times we need to difference the data to get it stationary so the AR and MA components can work.
# * <strong>MA(q)</strong> Moving Average - a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.It indicates that the regressions error is a linear combination of error terms. We will set up another regression model that focuses on the residual term between a moving average and the real values

# ![image.png](attachment:image.png)

# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt>
# <strong>
# <a href='https://www.alkaline-ml.com/pmdarima/user_guide.html#user-guide'>pmdarima.auto_arima</a></strong><font color=black>(y[,start_p,d,start_q, …])</font>&nbsp;&nbsp;&nbsp;Returns the optimal order for an ARIMA model<br>
# 
# <h3>Optional Function (see note below):</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.arma_order_select_ic.html'>stattools.arma_order_select_ic</a></strong><font color=black>(y[, max_ar, …])</font>&nbsp;&nbsp;Returns information criteria for many ARMA models<br><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.x13.x13_arima_select_order.html'>x13.x13_arima_select_order</a></strong><font color=black>(endog[, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Perform automatic seasonal ARIMA order identification using x12/x13 ARIMA</tt></div>

# In[5]:


# Load a non-stationary dataset
df1 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'

# Load a stationary dataset
df2 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'


# In[6]:


from pmdarima import auto_arima

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


# In[7]:


help(auto_arima)


# Auto_arima, based on the AIC, will give you the mest combination of p,d,q parameters

# In[8]:


auto_arima(df2['Births'])


# We specify we want the model to try from 0 to 6 p and from 0 to 3

# In[11]:


stepwise_fit = auto_arima(df2['Births'], start_p=0, start_q=0, max_p=6, max_q=3, seasonal=False, trace=True)


# For our data, the best p,d,q parameters found are 1,1,1. Note that auto_arima did not try all of the parameters, since he/she realised that the AIC was actually not improving,so it stopped trying more combinations

# This shows a recommended (p,d,q) ARIMA Order of (1,1,1), with no seasonal_order component.
# 
# We can see how this was determined by looking at the stepwise results. The recommended order is the one with the lowest <a href='https://en.wikipedia.org/wiki/Akaike_information_criterion'>Akaike information criterion</a> or AIC score. Note that the recommended model may <em>not</em> be the one with the closest fit. The AIC score takes complexity into account, and tries to identify the best <em>forecasting</em> model.

# Let's have a look now at the description of the winner parameter-combination

# <div class="alert alert-info"><strong>NOTE: </strong>Harmless warnings should have been suppressed, but if you see an error citing unusual behavior you can suppress this message by passing <font color=black><tt>error_action='ignore'</tt></font> into <tt>auto_arima()</tt>. Also, <font color=black><tt>auto_arima().summary()</tt></font> provides a nicely formatted summary table.</div>

# In[12]:


stepwise_fit.summary()


# Now let's look at the non-stationary, seasonal <strong>Airline Passengers</strong> dataset:
# Note that since we set seasonal=True, the model is also runing a SARIMA model

# In[13]:


# With Trace=True we can see the trace results of the parameter combination that the model is using
# m is the type of differencing, so if we difference on quarterly daya, we will set m=4, for monthly m=12, yearly m=1
stepwise_fit = auto_arima(df1['Thousands of Passengers'], start_p=1, start_q=1,
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# The winner is a SARIMA (0,1,1)x(2,1,12) (p,d,q)x(P,D,Q,M). We will talk about SARIMA later
# 
# In the table you can see the coefficients for each of the AR & MA lags

# # ARMA(p,q) and ARIMA(p,d,q)
# # Autoregressive Moving Averages
# This section covers <em>Autoregressive Moving Averages</em> (ARMA) and <em>Autoregressive Integrated Moving Averages</em> (ARIMA).
# 
# Recall that an <strong>AR(1)</strong> model follows the formula
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = c + \phi_{1}y_{t-1} + \varepsilon_{t}$
# 
# while an <strong>MA(1)</strong> model follows the formula
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = \mu + \theta_{1}\varepsilon_{t-1} + \varepsilon_{t}$
# 
# where $c$ is a constant, $\mu$ is the expectation of $y_{t}$ (often assumed to be zero), $\phi_1$ (phi-sub-one) is the AR lag coefficient, $\theta_1$ (theta-sub-one) is the MA lag coefficient, and $\varepsilon$ (epsilon) is white noise.
# 
# An <strong>ARMA(1,1)</strong> model therefore follows
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = c + \phi_{1}y_{t-1} + \theta_{1}\varepsilon_{t-1} + \varepsilon_{t}$
# 
# ARMA models can be used on stationary datasets.
# 
# For non-stationary datasets with a trend component, ARIMA models apply a differencing coefficient as well.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARMA.html'>arima_model.ARMA</a></strong><font color=black>(endog, order[, exog, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Autoregressive Moving Average ARMA(p,q) model<br>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARMAResults.html'>arima_model.ARMAResults</a></strong><font color=black>(model, params[, …])</font>&nbsp;&nbsp;&nbsp;Class to hold results from fitting an ARMA model<br>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html'>arima_model.ARIMA</a></strong><font color=black>(endog, order[, exog, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;Autoregressive Integrated Moving Average ARIMA(p,d,q) model<br>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.html'>arima_model.ARIMAResults</a></strong><font color=black>(model, params[, …])</font>&nbsp;&nbsp;Class to hold results from fitting an ARIMA model<br>	
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.kalmanf.kalmanfilter.KalmanFilter.html'>kalmanf.kalmanfilter.KalmanFilter</a></strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kalman Filter code intended for use with the ARMA model</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model'>Wikipedia</a></strong>&nbsp;&nbsp;<font color=black>Autoregressive–moving-average model</font><br>
# <strong>
# <a href='https://otexts.com/fpp2/non-seasonal-arima.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Non-seasonal ARIMA models</font></div>

# ___
# ## Autoregressive Moving Average - ARMA(p,q)
# In this first section we'll look at a stationary dataset, determine (p,q) orders, and run a forecasting ARMA model fit to the data. In practice it's rare to find stationary data with no trend or seasonal component, but the first four months of the <em>Daily Total Female Births</em> dataset should work for our purposes.
# ### Plot the source data

# In[16]:


# Load specific forecasting tools
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from pmdarima import auto_arima # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load datasets
df1 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df1.index.freq = 'D'
df1 = df1[:120]  # we only want the first four months

df2 = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/TradeInventories.csv',index_col='Date',parse_dates=True)
df2.index.freq='MS'


# In[18]:


# this function was created previously (control f adf_test)
adf_test(df1["Births"])


# ### Determine the (p,q) ARMA Orders using <tt>pmdarima.auto_arima</tt>
# This tool should give just $p$ and $q$ value recommendations for this dataset.

# In[20]:


auto_arima(df1["Births"],seasonal=False).summary()


# ### Split the data into train/test sets
# As a general rule you should set the length of your test set equal to your intended forecast size. For this dataset we'll attempt a 1-month forecast.

# In[21]:


# Set one month for testing
train = df1.iloc[:90]
test = df1.iloc[90:]


# ### Fit an ARMA(p,q) Model
# If you want you can run <tt>help(ARMA)</tt> to learn what incoming arguments are available/expected, and what's being returned.

# In[22]:


model = ARMA(train['Births'],order=(2,2))
results = model.fit()
results.summary()


# ### Obtain a month's worth of predicted values

# In[23]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end).rename('ARMA(2,2) Predictions')


# ### Plot predictions against known values

# In[24]:


title = 'Daily Total Female Births'
ylabel='Births'
xlabel='' # we don't really need a label here

ax = test['Births'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# The orange line is the predicted values,  which even though it looks like it is a very bad prediction, it makes completely sense since ARMA shows the average value. 
# The model is not able to predict the noise, but is able to predict the average value. 
# 
# In fact if we calculate the average value of test and predictions, they have the same average value:

# In[25]:


test.mean()


# In[26]:


predictions.mean()


# Now let's work on the ARIMA mode,so we will ad the I component into the ARMA model:

# ___
# ## Autoregressive Integrated Moving Average - ARIMA(p,d,q)
# The steps are the same as for ARMA(p,q), except that we'll apply a differencing component to make the dataset stationary.<br>
# First let's take a look at the <em>Real Manufacturing and Trade Inventories</em> dataset.
# ### Plot the Source Data

# In[28]:


# HERE'S A TRICK TO ADD COMMAS TO Y-AXIS TICK VALUES
import matplotlib.ticker as ticker

formatter = ticker.StrMethodFormatter('{x:,.0f}')

title = 'Real Manufacturing and Trade Inventories'
ylabel='Chained 2012 Dollars'
xlabel='' # we don't really need a label here

ax = df2['Inventories'].plot(figsize=(12,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# ### Run an ETS Decomposition 
# We probably won't learn a lot from it, but it never hurts to run an ETS Decomposition plot.

# In[29]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df2['Inventories'], model='additive')  # model='add' also works
result.plot();


# Here we see that the seasonal component does not contribute significantly to the behavior of the series.
# ### Use <tt>pmdarima.auto_arima</tt> to determine ARIMA Orders
# 
# So for the purpose of this code, we will ignore the seasonal component (otherwise we should use SARIMA, instead of ARIMA)

# In[30]:


auto_arima(df2['Inventories'],seasonal=False).summary()


# This suggests that we should fit an SARIMAX(0,1,0) model to best forecast future values of the series. Before we train the model, let's look at augmented Dickey-Fuller Test, and the ACF/PACF plots to see if they agree. These steps are optional, and we would likely skip them in practice.

# In[31]:


from statsmodels.tsa.statespace.tools import diff
df2['d1'] = diff(df2['Inventories'],k_diff=1)

# Equivalent to:
# df1['d1'] = df1['Inventories'] - df1['Inventories'].shift(1)

adf_test(df2['d1'],'Real Manufacturing and Trade Inventories')


# This confirms that we reached stationarity after the first difference.
# ### Run the ACF and PACF plots
# A <strong>PACF Plot</strong> can reveal recommended AR(p) orders, and an <strong>ACF Plot</strong> can do the same for MA(q) orders.<br>
# Alternatively, we can compare the stepwise <a href='https://en.wikipedia.org/wiki/Akaike_information_criterion'>Akaike Information Criterion (AIC)</a> values across a set of different (p,q) combinations to choose the best combination.

# In[32]:


title = 'Autocorrelation: Real Manufacturing and Trade Inventories'
lags = 40
plot_acf(df2['Inventories'],title=title,lags=lags);


# In[33]:


title = 'Partial Autocorrelation: Real Manufacturing and Trade Inventories'
lags = 40
plot_pacf(df2['Inventories'],title=title,lags=lags);


# This tells us that the AR component should be more important than MA. From the <a href='https://people.duke.edu/~rnau/411arim3.htm'>Duke University Statistical Forecasting site</a>:<br>
# > <em>If the PACF displays a sharp cutoff while the ACF decays more slowly (i.e., has significant spikes at higher lags), we    say that the stationarized series displays an "AR signature," meaning that the autocorrelation pattern can be explained more    easily by adding AR terms than by adding MA terms.</em><br>
# 
# Let's take a look at <tt>pmdarima.auto_arima</tt> done stepwise to see if having $p$ and $q$ terms the same still makes sense:

# In[34]:


stepwise_fit = auto_arima(df2['Inventories'], start_p=0, start_q=0,
                          max_p=2, max_q=2, m=12,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# ### Split the data into train/test sets

# In[35]:


len(df2)


# In[36]:


# Set one year for testing
train = df2.iloc[:252]
test = df2.iloc[252:]


# ### Fit an ARIMA(1,1,1) Model

# In[38]:


model = ARIMA(train['Inventories'],order=(1,1,1))
results = model.fit()
results.summary()


# In[39]:


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')


# Passing <tt>dynamic=False</tt> means that forecasts at each point are generated using the full history up to that point (all lagged values).
# 
# Passing <tt>typ='levels'</tt> predicts the levels of the original endogenous variables. If we'd used the default <tt>typ='linear'</tt> we would have seen linear predictions in terms of the differenced endogenous variables.
# 
# For more information on these arguments visit https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html

# In[41]:


# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['Inventories'][i]}")


# In[42]:


# Plot predictions against known values
title = 'Real Manufacturing and Trade Inventories'
ylabel='Chained 2012 Dollars'
xlabel='' # we don't really need a label here

ax = test['Inventories'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# ### Evaluate the Model

# In[43]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(test['Inventories'], predictions)
print(f'ARIMA(1,1,1) MSE Error: {error:11.10}')


# In[44]:


from statsmodels.tools.eval_measures import rmse

error = rmse(test['Inventories'], predictions)
print(f'ARIMA(1,1,1) RMSE Error: {error:11.10}')


# ### Retrain the model on the full data, and forecast the future

# In[45]:


model = ARIMA(df2['Inventories'],order=(1,1,1))
results = model.fit()
fcast = results.predict(len(df2),len(df2)+11,typ='levels').rename('ARIMA(1,1,1) Forecast')


# In[46]:


# Plot predictions against known values
title = 'Real Manufacturing and Trade Inventories'
ylabel='Chained 2012 Dollars'
xlabel='' # we don't really need a label here

ax = df2['Inventories'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# # SARIMA(p,d,q)(P,D,Q)m
# # Seasonal Autoregressive Integrated Moving Averages
# We have finally reached one of the most fascinating aspects of time series analysis: seasonality.
# 
# Where ARIMA accepts the parameters $(p,d,q)$, SARIMA accepts an <em>additional</em> set of parameters $(P,D,Q)m$ that specifically describe the seasonal components of the model. Here $P$, $D$ and $Q$ represent the seasonal regression, differencing and moving average coefficients, and $m$ represents the number of data points (rows) in each seasonal cycle.
# 
# <strong>NOTE:</strong> The statsmodels implementation of SARIMA is called SARIMAX. The “X” added to the name means that the function also supports <em>exogenous</em> regressor variables. We'll cover these in the next section.
# 
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html'>sarimax.SARIMAX</a></strong><font color=black>(endog[, exog, order, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html'>sarimax.SARIMAXResults</a></strong><font color=black>(model, params, …[, …])</font>&nbsp;&nbsp;Class to hold results from fitting a SARIMAX model.</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/statespace.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Time Series Analysis by State Space Methods</font></div>

# ## Perform standard imports and load datasets

# In[47]:


# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/co2_mm_mlo.csv')


# In[ ]:


df.head()


# We need to combine two integer columns (year and month) into a DatetimeIndex. We can do this by passing a dictionary into <tt>pandas.to_datetime()</tt> with year, month and day values.<br>
# For more information visit https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

# In[48]:


# Add a "date" datetime column
df['date']=pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))

#another way to merge dates is  pd.to_datetime({'year':df['year'],'month':df['month'],'day':1})


# Set "date" to be the index
df.set_index('date',inplace=True)
df.index.freq = 'MS'
df.head()


# In[49]:


df.info()


# ### Plot the source data

# In[50]:


title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel='' # we don't really need a label here

ax = df['interpolated'].plot(figsize=(12,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# ### Run an ETS Decomposition

# In[51]:


result = seasonal_decompose(df['interpolated'], model='add')
result.plot();


# Although small in scale compared to the overall values, there is a definite annual seasonality.

# ### Run <tt>pmdarima.auto_arima</tt> to obtain recommended orders
# This may take awhile as there are a lot more combinations to evaluate.

# In[52]:


# For SARIMA Orders we set seasonal=True and pass in an m value
# since seasonality is yearl and we have monthly data then m=12
auto_arima(df['interpolated'],seasonal=True,m=12).summary()


# ### Split the data into train/test sets

# In[54]:


# Set one year for testing
train = df.iloc[:717]
test = df.iloc[717:]


# ### Fit

# In[55]:


model = SARIMAX(train['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()


# In[56]:


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(0,1,3)(1,0,1,12) Predictions')


# Passing <tt>dynamic=False</tt> means that forecasts at each point are generated using the full history up to that point (all lagged values).
# 
# Passing <tt>typ='levels'</tt> predicts the levels of the original endogenous variables. If we'd used the default <tt>typ='linear'</tt> we would have seen linear predictions in terms of the differenced endogenous variables.
# 
# For more information on these arguments visit https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html

# In[57]:


# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['interpolated'][i]}")


# In[58]:


# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''

ax = test['interpolated'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# ### Evaluate the Model

# In[59]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(test['interpolated'], predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) MSE Error: {error:11.10}')

from statsmodels.tools.eval_measures import rmse

error = rmse(test['interpolated'], predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) RMSE Error: {error:11.10}')


# Remember that in order to understand and intepret de error you need always to look at the data. 

# In[60]:


test['interpolated'].mean()


# These are outstanding results!, since a RMSE of 0.35 is tiny compared to 408.333
# ### Retrain the model on the full data, and forecast the future

# In[61]:


model = SARIMAX(df['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
fcast = results.predict(len(df),len(df)+11,typ='levels').rename('SARIMA(0,1,3)(1,0,1,12) Forecast')

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''

ax = df['interpolated'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# # SARIMAX
# 
# ## Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
# So far the models we've looked at consider past values of a dataset and past errors to determine future trends, seasonality and forecasted values. We look now to models that encompass these non-seasonal (p,d,q) and seasonal (P,D,Q,m) factors, but introduce the idea that external factors (environmental, economic, etc.) can also influence a time series, and be used in forecasting.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html'>sarimax.SARIMAX</a></strong><font color=black>(endog[, exog, order, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html'>sarimax.SARIMAXResults</a></strong><font color=black>(model, params, …[, …])</font>&nbsp;&nbsp;Class to hold results from fitting a SARIMAX model.</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/statespace.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Time Series Analysis by State Space Methods</font><br>
# <strong>
# <a href='https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_sarimax_stata.html'>Statsmodels Example:</a></strong>&nbsp;&nbsp;<font color=black>SARIMAX</font></div>

# ## Perform standard imports and load datasets

# In[4]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/RestaurantVisitors.csv',index_col='date',parse_dates=True)
df.index.freq = 'D'


# ### Inspect the data
# For this section we've built a Restaurant Visitors dataset that was inspired by a <a href='https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting'>recent Kaggle competition</a>. The data considers daily visitors to four restaurants located in the United States, subject to American holidays. For the exogenous variable we'll see how holidays affect patronage. The dataset contains 478 days of restaurant data, plus an additional 39 days of holiday data for forecasting purposes.

# Notice that even though the restaurant visitor columns contain integer data, they appear as floats. This is because the bottom of the dataframe has 39 rows of NaN data to accommodate the extra holiday data we'll use for forecasting, and pandas won't allow NaN's as integers. We could leave it like this, but since we have to drop NaN values anyway, let's also convert the columns to dtype int64.

# In[5]:


df1 = df.dropna()
df1.tail()


# In[ ]:


# Change the dtype of selected columns from float to int (we cannot isk having 1.5 customers)
cols = ['rest1','rest2','rest3','rest4','total']
for col in cols:
    df1[col] = df1[col].astype(int)
df1.head()


# ### Plot the source data

# In[6]:


title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel='' # we don't really need a label here

ax = df1['total'].plot(figsize=(16,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# ## Look at holidays
# Rather than prepare a separate plot, we can use matplotlib to shade holidays behind our restaurant data.

# In[8]:


title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel='' # we don't really need a label here

ax = df1['total'].plot(figsize=(16,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in df1.query('holiday==1').index:     # for days where holiday == 1  #you can also do it the classical way: df1[df1['holiday']==1].index
    ax.axvline(x=x, color='k', alpha = 0.3);  # add a semi-transparent grey line


# ### Run an ETS Decomposition

# In[9]:


result = seasonal_decompose(df1['total'])
result.plot();


# In[12]:


result.seasonal.plot(figsize=(18,5));


# ## Test for stationarity

# In[10]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[13]:


adf_test(df1['total'])


# ### Run <tt>pmdarima.auto_arima</tt> to obtain recommended orders
# This may take awhile as there are a lot of combinations to evaluate.

# In[ ]:


# For SARIMA Orders we set seasonal=True and pass in an m value
auto_arima(df1['total'],seasonal=True,m=7).summary()


# Excellent! This provides an ARIMA Order of (1,0,0) and a seasonal order of (2,0,0,7) Now let's train & test the SARIMA model, evaluate it, then compare the result to a model that uses an exogenous variable.
# ### Split the data into train/test sets
# We'll assign 42 days (6 weeks) to the test set so that it includes several holidays.

# In[17]:


# Set four weeks for testing
train = df1.iloc[:436]
test = df1.iloc[436:]


# ### Fit a SARIMA(1,0,0)(2,0,0,7) Model
# NOTE: To avoid a <tt>ValueError: non-invertible starting MA parameters found</tt> we're going to set <tt>enforce_invertibility</tt> to False.

# In[18]:


model = SARIMAX(train['total'],order=(1,0,0),seasonal_order=(2,0,0,7),enforce_invertibility=False)
#Why invertible process ? invertibility means that recent lags have more weigh than old lags. In other words
#the coefficient theta are < 1. Statsmodels believes this is the "normal", so if forces the coefficients to be 
#always <1. But we don't want this, so we don't want to enfornce invertibility. We want to allow that the coefficients
#of the lags are => 1
results = model.fit()
results.summary()


# In[19]:


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False).rename('SARIMA(1,0,0)(2,0,0,7) Predictions')


# Passing <tt>dynamic=False</tt> means that forecasts at each point are generated using the full history up to that point (all lagged values).
# 
# For more information on these arguments visit https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html

# In[21]:


# Plot predictions against known values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = test['total'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in test.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);


# ### Evaluate the Model

# In[22]:


from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

error1 = mean_squared_error(test['total'], predictions)
error2 = rmse(test['total'], predictions)

print(f'SARIMA(1,0,0)(2,0,0,7) MSE Error: {error1:11.10}')
print(f'SARIMA(1,0,0)(2,0,0,7) RMSE Error: {error2:11.10}')


# ## Now add the exog variable

# In[24]:


model = SARIMAX(train['total'],exog=train['holiday'],order=(1,0,0),seasonal_order=(2,0,0,7),enforce_invertibility=False)
results = model.fit()
results.summary()


# In[25]:


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
exog_forecast = test[['holiday']]  # requires two brackets to yield a shape of (35,1)
predictions = results.predict(start=start, end=end, exog=exog_forecast).rename('SARIMAX(1,0,0)(2,0,0,7) Predictions')


# In[26]:


# Plot predictions against known values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = test['total'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in test.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);


# We can see that the exogenous variable (holidays) had a positive impact on the forecast by raising predicted values at 3/17, 4/14, 4/16 and 4/17! Let's compare evaluations:
# ### Evaluate the Model

# In[27]:


# Print values from SARIMA above
print(f'SARIMA(1,0,0)(2,0,0,7) MSE Error: {error1:11.10}')
print(f'SARIMA(1,0,0)(2,0,0,7) RMSE Error: {error2:11.10}')
print()

error1x = mean_squared_error(test['total'], predictions)
error2x = rmse(test['total'], predictions)

# Print new SARIMAX values
print(f'SARIMAX(1,0,0)(2,0,0,7) MSE Error: {error1x:11.10}')
print(f'SARIMAX(1,0,0)(2,0,0,7) RMSE Error: {error2x:11.10}')


# Our RMSE is lower, which means adding the exogenous variable helps us to predict better the future

# ### Retrain the model on the full data, and forecast the future
# We're going to forecast 39 days into the future, and use the additional holiday data

# In[28]:


model = SARIMAX(df1['total'],exog=df1['holiday'],order=(1,0,0),seasonal_order=(2,0,0,7),enforce_invertibility=False)
results = model.fit()
exog_forecast = df[478:][['holiday']]
fcast = results.predict(len(df1),len(df1)+38,exog=exog_forecast).rename('SARIMAX(1,0,0)(2,0,0,7) Forecast')


# In[29]:


# Plot the forecast alongside historical values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = df1['total'].plot(legend=True,figsize=(16,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in df.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);


# # VAR(p)
# ## Vector Autoregression
# In our previous SARIMAX example, the forecast variable $y_t$ was influenced by the exogenous predictor variable, but not vice versa. That is, the occurrence of a holiday affected restaurant patronage but not the other way around.
# 
# However, there are some cases where variables affect each other. <a href='https://otexts.com/fpp2/VAR.html'>Forecasting: Principles and Practice</a> describes a case where changes in personal consumption expenditures $C_t$ were forecast based on changes in personal disposable income $I_t$.
# > However, in this case a bi-directional relationship may be more suitable: an increase in $I_t$ will lead to an increase in $C_t$ and vice versa.<br>An example of such a situation occurred in Australia during the Global Financial Crisis of 2008–2009. The Australian government issued stimulus packages that included cash payments in December 2008, just in time for Christmas spending. As a result, retailers reported strong sales and the economy was stimulated. Consequently, incomes increased.
# 
# Aside from investigating multivariate time series, vector autoregression is used for
# * <a href='https://www.statsmodels.org/devel/vector_ar.html#impulse-response-analysis'>Impulse Response Analysis</a> which involves the response of one variable to a sudden but temporary change in another variable
# * <a href='https://www.statsmodels.org/devel/vector_ar.html#forecast-error-variance-decomposition-fevd'>Forecast Error Variance Decomposition (FEVD)</a> where the proportion of the forecast variance of one variable is attributed to the effect of other variables
# * <a href='https://www.statsmodels.org/devel/vector_ar.html#dynamic-vector-autoregressions'>Dynamic Vector Autoregressions</a> used for estimating a moving-window regression for the purposes of making forecasts throughout the data sample
# 
# ### Formulation
# We've seen that an autoregression AR(p) model is described by the following:
# 
# &nbsp;&nbsp;&nbsp;&nbsp; $y_{t} = c + \phi_{1}y_{t-1} + \phi_{2}y_{t-2} + \dots + \phi_{p}y_{t-p} + \varepsilon_{t}$
# 
# where $c$ is a constant, $\phi_{1}$ and $\phi_{2}$ are lag coefficients up to order $p$, and $\varepsilon_{t}$ is white noise.

# A $K$-dimensional VAR model of order $p$, denoted <strong>VAR(p)</strong>, considers each variable $y_K$ in the system.<br>
# 
# For example, The system of equations for a 2-dimensional VAR(1) model is:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{1,t} = c_1 + \phi_{11,1}y_{1,t-1} + \phi_{12,1}y_{2,t-1} + \varepsilon_{1,t}$<br>
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{2,t} = c_2 + \phi_{21,1}y_{1,t-1} + \phi_{22,1}y_{2,t-1} + \varepsilon_{2,t}$
# 
# where the coefficient $\phi_{ii,l}$ captures the influence of the $l$th lag of variable $y_i$ on itself,<br>
# the coefficient $\phi_{ij,l}$ captures the influence of the $l$th lag of variable $y_j$ on $y_i$,<br>
# and $\varepsilon_{1,t}$ and $\varepsilon_{2,t}$ are white noise processes that may be correlated.<br>
# 
# Carrying this further, the system of equations for a 2-dimensional VAR(3) model is:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{1,t} = c_1 + \phi_{11,1}y_{1,t-1} + \phi_{12,1}y_{2,t-1} + \phi_{11,2}y_{1,t-2} + \phi_{12,2}y_{2,t-2} + \phi_{11,3}y_{1,t-3} + \phi_{12,3}y_{2,t-3} + \varepsilon_{1,t}$<br>
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{2,t} = c_2 + \phi_{21,1}y_{1,t-1} + \phi_{22,1}y_{2,t-1} + \phi_{21,2}y_{1,t-2} + \phi_{22,2}y_{2,t-2} + \phi_{21,3}y_{1,t-3} + \phi_{22,3}y_{2,t-3} + \varepsilon_{2,t}$<br><br>
# 
# and the system of equations for a 3-dimensional VAR(2) model is:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{1,t} = c_1 + \phi_{11,1}y_{1,t-1} + \phi_{12,1}y_{2,t-1} + \phi_{13,1}y_{3,t-1} + \phi_{11,2}y_{1,t-2} + \phi_{12,2}y_{2,t-2} + \phi_{13,2}y_{3,t-2} + \varepsilon_{1,t}$<br>
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{2,t} = c_2 + \phi_{21,1}y_{1,t-1} + \phi_{22,1}y_{2,t-1} + \phi_{23,1}y_{3,t-1} + \phi_{21,2}y_{1,t-2} + \phi_{22,2}y_{2,t-2} + \phi_{23,2}y_{3,t-2} + \varepsilon_{2,t}$<br>
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{3,t} = c_3 + \phi_{31,1}y_{1,t-1} + \phi_{32,1}y_{2,t-1} + \phi_{33,1}y_{3,t-1} + \phi_{31,2}y_{1,t-2} + \phi_{32,2}y_{2,t-2} + \phi_{33,2}y_{3,t-2} + \varepsilon_{3,t}$<br><br>
# 
# The general steps involved in building a VAR model are:
# * Examine the data
# * Visualize the data
# * Test for stationarity
# * If necessary, transform the data to make it stationary
# * Select the appropriate order <em>p</em>
# * Instantiate the model and fit it to a training set
# * If necessary, invert the earlier transformation
# * Evaluate model predictions against a known test set
# * Forecast the future
# 
# Recall that to fit a SARIMAX model we passed one field of data as our <em>endog</em> variable, and another for <em>exog</em>. With VAR, both fields will be passed in as <em>endog</em>.

# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html'>vector_ar.var_model.VAR</a></strong><font color=black>(endog[, exog, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fit VAR(p) process and do lag order selection<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html'>vector_ar.var_model.VARResults</a></strong><font color=black>(endog, …[, …])</font>&nbsp;&nbsp;Estimate VAR(p) process with fixed number of lags<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.dynamic.DynamicVAR.html'>vector_ar.dynamic.DynamicVAR</a></strong><font color=black>(data[, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Estimates time-varying vector autoregression (VAR(p)) using equation-by-equation least squares</tt>
#    
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/vector_ar.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Vector Autoregressions</font><br>
# <strong>
# <a href='https://otexts.com/fpp2/VAR.html'>Forecasting: Principles and Practice:</a></strong>&nbsp;&nbsp;<font color=black>Vector Autoregressions</font><br>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Vector_autoregression'>Wikipedia:</a></strong>&nbsp;&nbsp;<font color=black>Vector Autoregression</font>
# </div>

# ### Perform standard imports and load dataset
# For this analysis we'll also compare money to spending. We'll look at the M2 Money Stock which is a measure of U.S. personal assets, and U.S. personal spending. Both datasets are in billions of dollars, monthly, seasonally adjusted. They span the 21 years from January 1995 to December 2015 (252 records).<br>
# Sources: https://fred.stlouisfed.org/series/M2SL https://fred.stlouisfed.org/series/PCE

# In[8]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

# Load specific forecasting tools
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load datasets
df = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/M2SL.csv',index_col=0, parse_dates=True)
df.index.freq = 'MS'

sp = pd.read_csv('./original/TSA_COURSE_NOTEBOOKS/Data/PCE.csv',index_col=0, parse_dates=True)
sp.index.freq = 'MS'


# In[11]:


df = df.join(sp)
df = df.dropna()
df.shape


# In[14]:


df.columns = ["Money", "Spending"]


# In[16]:


title = 'M2 Money Stock vs. Personal Consumption Expenditures'
ylabel='Billions of dollars'
xlabel=''

ax = df['Spending'].plot(figsize=(12,5),title=title,legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
df['Money'].plot(legend=True);


# ## Test for stationarity, perform any necessary transformations

# In[17]:


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[18]:


adf_test(df['Money'],title='Money')


# In[19]:


adf_test(df['Spending'], title='Spending')


# Neither variable is stationary, so we'll take a first order difference of the entire DataFrame and re-run the augmented Dickey-Fuller tests. It's advisable to save transformed values in a new DataFrame, as we'll need the original when we later invert the transormations and evaluate the model.

# In[20]:


df_transformed = df.diff()


# In[21]:


df_transformed = df_transformed.dropna()
adf_test(df_transformed['Money'], title='MoneyFirstDiff')
print()
adf_test(df_transformed['Spending'], title='SpendingFirstDiff')


# Still they are not stationary. We will need to keep differencing until they are both stationary.
# 
# Note that we need to have the same number of rows for both series. 
# 
# So in case one of that was already statinary and the other not, we still need to difference both of them

# In[22]:


df_transformed = df_transformed.diff().dropna()
adf_test(df_transformed['Money'], title='MoneySecondDiff')
print()
adf_test(df_transformed['Spending'], title='SpendingSecondDiff')


# ### Train/test split
# It will be useful to define a number of observations variable for our test set. For this analysis, let's use 12 months.

# In[23]:


nobs=12  #number of observations
train, test = df_transformed[0:-nobs], df_transformed[-nobs:]

print(train.shape)
print(test.shape)


# In[ ]:




