#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np


# In[8]:


# Question c
data = [177.424455, 108.118507, 107.584384, 105.142201, 110.218627, 114.349620, 118.343457, 133.335459, 135.194189, 133.143959]
plt.plot(range(1,11), data, 'r-x', label="RMSE-d")
plt.legend()
plt.xlabel('d')
plt.ylabel('cross validation error')
plt.show()


# In[10]:


# Question e
data = [105.232654, 105.731522, 107.528536, 109.710005, 111.950080, 114.132572, 116.229941, 118.256502, 120.239390, 122.199700, 124.145186]
plt.plot(range(0,55,5), data, 'r-x', label="RMSE-lam_mu")
plt.legend()
plt.xlabel('lam_mu')
plt.ylabel('cross validation error')
plt.show()


# In[ ]:




