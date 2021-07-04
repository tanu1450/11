#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundadtion
# ## Data Science and Business Analytics Internship | GRIPJUL'21

# ### Task-1 : Prediction using Supervised ML
# ### Author - Tanushree gaur

# #### Problem Statement : Prediction of a student based on the number of study hours.

# ### Importing necessary libraries

# In[11]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing data

# In[13]:


url="http://bit.ly/w-data"
df=pd.read_csv(url)
print("Data imported succesfully")
df.head(25)


# In[14]:


df.shape


# ## Plotting the distribution of scores

# In[18]:


plt.title('Hours vs Percentage')
plt.xlabel('Hours studdied')
plt.ylabel('Percentage score')
plt.scatter(df.Hours,df.Scores)
plt.show()


# ## Creating training and test dataset

# In[38]:


x=df.iloc[:, :-1].values
y=df.iloc[:, 1].values


# ## Splitting the data into training and test tests

# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)


# ## Creating regression model and training the model

# In[41]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
print("score:", regressor.score(x_train,y_train))
print("trainig complete")


# ## Plotting the regression line on training set

# In[44]:


line=regressor.coef_*x_train +regressor.intercept_
plt.title('Regression line(training set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage score')
plt.scatter(x_train,y_train)
plt.plot(x_train,line)
plt.show()


# ## Plotting the regression line on test set

# In[46]:


line=regressor.coef_*x_test +regressor.intercept_
plt.title('Regression line(test set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage score')
plt.scatter(x_test,y_test)
plt.plot(x_test,line)
plt.show()


# ## Predicting the scores

# In[56]:


y_pred=regressor.predict(x_test)
y_pred


# ## Comparing actual vs Predicted scores

# In[57]:


df1=pd.DataFrame({'Actaual': y_test, 'Predicted': y_pred })
df1


# ## evaluating the model

# In[59]:


from sklearn.metrics import r2_score
print('Accuracy:',r2_score(y_test,y_pred)*100,'%')


# ### Our model is giving 92% accuracy

# ## Predicting the score

# In[61]:


pred=regressor.predict([[9]])
print('No. of Hours studied={}'.format(9))
print('Predicted Score={}'.format(pred[0]))


# ## Conclussion:

# ### From the above result we can conclude that if a student studies for 9 hours, then his score will be 91.58 marks

# ### Completed TASK-1
# ### Thankyou
# ### Tanushree Gaur

# In[ ]:




