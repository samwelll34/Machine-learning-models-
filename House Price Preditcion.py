#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


HouseDF = pd.read_csv('USA_Housing.csv')


# In[3]:


HouseDF.head()


# In[4]:


HouseDF.info()


# In[5]:


HouseDF.describe()


# In[6]:


HouseDF.columns


# In[7]:


sns.pairplot(HouseDF)


# In[10]:


sns.displot(HouseDF['Price'])


# In[18]:


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[25]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[22]:


print(lm.intercept_)


# In[26]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[27]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[31]:


sns.histplot((y_test-predictions),bins=50);


# In[32]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




