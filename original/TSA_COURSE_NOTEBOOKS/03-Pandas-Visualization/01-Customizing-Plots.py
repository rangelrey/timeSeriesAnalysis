#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Customizing Pandas Plots
# In this section we'll show how to control the position and appearance of axis labels and legends.<br>
# For more info on the following topics visit https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df2 = pd.read_csv('df2.csv')


# ## Colors, Widths and Linestyles
# The pandas <tt>.plot()</tt> method takes optional arguments that allow you to control linestyles, colors, widths and more.

# In[2]:


# START WITH A SIMPLE LINE PLOT
df2['c'].plot(figsize=(8,3));


# In[3]:


df2['c'].plot.line(figsize=(8,3));


# <table style="display: inline-block">
#     <tr><th>PROPERTY</th><th>CODE</th><th>VALUE</th><th>EFFECT</th></tr>
#     <tr><td>linestyle</td><td><tt>ls</tt></td><td><tt>'-'</tt></td><td>solid line (default)</td></tr>
#     <tr><td>linestyle</td><td><tt>ls</tt></td><td><tt>'--'</tt></td><td>dashed line</td></tr>
#     <tr><td>linestyle</td><td><tt>ls</tt></td><td><tt>'-.'</tt></td><td>dashed/dotted line</td></tr>
#     <tr><td>linestyle</td><td><tt>ls</tt></td><td><tt>':'</tt></td><td>dotted line</td></tr>
#     <tr><td>color</td><td><tt>c</tt></td><td>string</td><td></td></tr>
#     <tr><td>linewidth</td><td><tt>lw</tt></td><td>float</td><td></td></tr>
# </table>

# In[4]:


df2['c'].plot.line(ls='-.', c='r', lw='4', figsize=(8,3));


# For more on linestyles, click <a href='https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle'>here</a>.

# ## Adding Titles and Axis Labels

# In[5]:


# START WITH A SIMPLE MULTILINE PLOT
df2.plot(figsize=(8,3));


# Before we tackle the issue of legend placement, let's add a title and axis labels.
# 
# In the previous section we learned how to pass a title into the pandas .plot() function; as it turns out, we can't pass axis labels this way.<br>

# ### Object-oriented plotting
# 
# When we call <tt>df.plot()</tt>, pandas returns a <tt>matplotlib.axes.AxesSubplot</tt> object. We can set labels
# on that object so long as we do it in the same jupyter cell. Setting an autoscale is done the same way.

# In[6]:


title='Custom Pandas Plot'
ylabel='Y-axis data'
xlabel='X-axis data'

ax = df2.plot(figsize=(8,3),title=title)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.autoscale(axis='x',tight=True);


# <font color=green>NOTE: The plot shrinks a bit so that the figure size can accommodate the new axis labels.</font>

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

# In[7]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=1);


# We can pass a second argument, <tt>bbox_to_anchor</tt> that treats the value passed in through <tt>loc</tt> as an anchor point, and positions the legend along the x and y axes based on a two-value tuple.

# In[8]:


# FIRST, PLACE THE LEGEND IN THE LOWER-LEFT
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3);


# In[9]:


# NEXT, MOVE THE LEGEND A LITTLE TO THE RIGHT AND UP
ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(0.1,0.1));


# ### Placing the Legend Outside the Plot
# In the above plot we passed <tt>(0.1,0.1)</tt> as our two-item tuple. This places the legend slightly to the right and slightly upward.<br>To place the legend outside the plot on the right-hand side, pass a value greater than or equal to 1 as the first item in the tuple.

# In[10]:


ax = df2.plot(figsize=(8,3))
ax.autoscale(axis='x',tight=True)
ax.legend(loc=3, bbox_to_anchor=(1.0,0.1));


# That's it! Feel free to experiment with different settings and plot types.
