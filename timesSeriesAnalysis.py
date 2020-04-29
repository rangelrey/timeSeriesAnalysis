#!/usr/bin/env python
# coding: utf-8

# # Numpy Reminders: 
# Let's have a look at some of the most basic numpy concepts that we will use in the more advanced sections

# In[4]:


import numpy as np


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


# In[122]:


#Import data
df1 = pd.read_csv("./original/TSA_COURSE_NOTEBOOKS/03-Pandas-Visualization/df1.csv",index_col=0)


# In[123]:


df1["A"].plot.hist()


# In[ ]:




