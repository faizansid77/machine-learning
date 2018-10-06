#!/usr/bin/env python
# coding: utf-8

# In[327]:


import numpy as np
import matplotlib.pyplot as plt


# In[328]:


df=np.loadtxt("binclassv2.txt",delimiter=',')


# In[329]:


x=np.zeros(shape=(np.shape(df)[0],3))
x[:,:1]=1
x[:,1:]=df[:,:2]
y=np.zeros(shape=(np.shape(df)[0],1))
y[:200,:]=df[:200,2:]
y[200:,:]=0
x1=df[:200:,:1]
x2=df[200:,:1]
y1=df[:200,1:2]
y2=df[200:,1:2]


# In[330]:


plt.scatter(x1,y1,color='r')
plt.scatter(x2,y2,color='b')


# In[331]:


def sigmoid(x):
    g=1/(1+2.71**(-x))
    return g
sigmoid=np.vectorize(sigmoid)


# In[332]:


epoch=1000
m=np.size(x)


# In[333]:


alpha=0.0001
theta=np.zeros(shape=(3,1))
for i in range(epoch):
    temp=(alpha/m)*np.dot(np.transpose(x),sigmoid(np.dot(x,theta))-y)
    theta=theta-temp


# In[334]:


plt.scatter(x1,y1,color='orange')
plt.scatter(x2,y2,color='g')
slope=-theta[2]/theta[1]
intercept=-theta[0]/theta[1]
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, 'b--')

