#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import matplotlib as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # readind the dataset

# In[4]:


suv_df=pd.read_csv("C:/Users/konda/Downloads/suv_data.csv")
suv_df.head(10)


# # Seperating independent and dependent variables

# In[13]:


x=suv_df.iloc[:,[2,3]].values  #iloc is index based function, it accepts row and column values(that row and columns can be more than one, and we can use slicing)
y=suv_df.iloc[:,4].values
#z=suv_df[["Gender","User ID"]] normal way of storing independent variable
#x to print x array
#y to print y array


# # importing train_test_split

# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[ ]:





# # Importing Logistic regression classifier

# In[20]:


from sklearn.linear_model import LogisticRegression


# # Fitting the data into the classifier

# In[21]:


classifier= LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# # Making the predicitons

# In[22]:


y_pred=classifier.predict(x_test)


# # Calculating accuracy score

# In[23]:


from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_test,y_pred)*100


# In[ ]:





# In[ ]:




