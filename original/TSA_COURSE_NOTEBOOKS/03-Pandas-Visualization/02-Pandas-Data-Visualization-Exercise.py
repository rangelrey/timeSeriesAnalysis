#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Pandas Data Visualization Exercises
# 
# This is just a quick exercise to review the various plots we showed earlier. Use <tt>df3.csv</tt> to replicate the following plots.
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>IMPORTANT NOTE!</strong> Make sure you don't run the cells directly above the example output shown, <br>otherwise you will end up writing over the example output!</div>

# In[1]:


# RUN THIS CELL
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df3 = pd.read_csv('df3.csv')
print(len(df3))
print(df3.head())


# So <tt>df3</tt> has 500 records and 3 columns. The data represents factory production numbers and reported numbers of defects on certain days of the week.

# ### 1. Recreate this scatter plot of 'produced' vs 'defective'. Note the color and size of the points. Also note the figure size. See if you can figure out how to stretch it in a similar fashion.

# In[ ]:


# CODE HERE


# In[2]:


# DON'T WRITE HERE


# ### 2. Create a histogram of the 'produced' column.

# In[ ]:





# In[3]:


# DON'T WRITE HERE


# ### 3. Recreate the following histogram of 'produced', tightening the x-axis and adding lines between bars.

# In[ ]:





# In[4]:


# DON'T WRITE HERE


# ### 4. Create a boxplot that shows 'produced' for each 'weekday' (hint: this is a groupby operation)

# In[ ]:





# In[5]:


# DON'T WRITE HERE


# ### 5. Create a KDE plot of the 'defective' column

# In[ ]:





# In[6]:


# DON'T WRITE HERE


# ### 6. For the above KDE plot, figure out how to increase the linewidth and make the linestyle dashed.<br>(Note: You would usually <em>not</em> dash a KDE plot line)

# In[ ]:





# In[7]:


# DON'T WRITE HERE


# ### 7. Create a <em>blended</em> area plot of all the columns for just the rows up to 30. (hint: use .loc)

# In[ ]:





# In[8]:


# DON'T WRITE HERE


# ## Bonus Challenge!
# 
# <strong>Notice how the legend in our previous figure overlapped some of actual diagram.<br> Can you figure out how to display the legend outside of the plot as shown below?</strong>

# In[ ]:





# In[9]:


# DON'T WRITE HERE


# # Great Job!
