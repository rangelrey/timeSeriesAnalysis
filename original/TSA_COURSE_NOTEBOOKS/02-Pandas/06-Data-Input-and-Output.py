#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# <div class="alert alert-info"><strong>NOTE:</strong> Typically we will just be either reading csv files directly or using pandas-datareader to pull data from the web. Consider this lecture just a quick overview of what is possible with pandas (we won't be working with SQL or Excel files in this course)</div>

# # Data Input and Output
# 
# This notebook is the reference code for getting input and output, pandas can read a variety of file types using its pd.read_ methods. Let's take a look at the most common data types:

# In[1]:


import numpy as np
import pandas as pd


# ## CSV
# Comma Separated Values files are text files that use commas as field delimeters.<br>
# Unless you're running the virtual environment included with the course, you may need to install <tt>xlrd</tt> and <tt>openpyxl</tt>.<br>
# In your terminal/command prompt run:
# 
#     conda install xlrd
#     conda install openpyxl
# 
# Then restart Jupyter Notebook.
# (or use pip install if you aren't using the Anaconda Distribution)
# 
# ### CSV Input

# In[2]:


df = pd.read_csv('example.csv')
df


# ### CSV Output

# In[3]:


df.to_csv('example.csv',index=False)


# ## Excel
# Pandas can read and write MS Excel files. However, this only imports data, not formulas or images. A file that contains images or macros may cause the <tt>.read_excel()</tt>method to crash. 

# ### Excel Input

# In[4]:


pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')


# ### Excel Output

# In[5]:


df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')


# ## HTML
# Pandas can read table tabs off of HTML.<br>
# Unless you're running the virtual environment included with the course, you may need to install <tt>lxml</tt>, <tt>htmllib5</tt>, and <tt>BeautifulSoup4</tt>.<br>
# In your terminal/command prompt run:
# 
#     conda install lxml
#     conda install html5lib
#     conda install beautifulsoup4
# 
# Then restart Jupyter Notebook.
# (or use pip install if you aren't using the Anaconda Distribution)

# ### HTML Input
# 
# Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects:

# In[6]:


df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')


# In[8]:


df[0].head()


# ____

# # Great Job!
