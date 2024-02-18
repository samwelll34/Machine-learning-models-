#!/usr/bin/env python
# coding: utf-8

# # #KEYWORD SEARCH ANALYSIS WITH PYTHON 

# In[1]:


pip install requests


# In[2]:


pip install pytrends


# In[1]:


import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
trends = TrendReq()


# In[2]:


trends.build_payload(kw_list=["Data Science"])
data = trends.interest_by_region()
print(data.sample(10))


# In[3]:


data.info()


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.sample(20)


# In[15]:


df = data.sample(15)
df.reset_index().plot(x="geoName", y="Data Science", figsize=(80,40), kind="bar")
plt.show()


# In[16]:


data = trends.trending_searches(pn="india")
print(data.head(10))


# In[17]:


keyword = trends.suggestions(keyword="Programming")
data = pd.DataFrame(keyword)
print(data.head())


# In[ ]:




