#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().system('pip install pandas_datareader')
import pandas_datareader as data


# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# In[2]:


get_ipython().system('pip install pandas_datareader')
import pandas_datareader as data
start = '2010-01-01'
end = '2019-12-31'
df = data.DataReader('AAPL','yahoo',start,end)
df.head()


# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook

# In[3]:


df.tail()


# In[4]:


df=df.reset_index()
df.head()


# In[5]:


df = df.drop(['Date','Adj Close'], axis = 1)
df.head ()


# In[6]:


plt.plot(df.Close)


# In[7]:


df


# In[8]:


ma100 = df.Close.rolling(100).mean()
ma100


# In[9]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')


# In[10]:


ma200 = df.Close.rolling(200).mean()
ma200


# In[11]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')


# In[12]:


df.shape


# In[13]:


#spilting data into taraning and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
 
print(data_training.shape)
print(data_testing.shape)


# In[14]:


data_training.head()


# In[15]:


data_testing.head()


# In[16]:


get_ipython().system('pip install sklearn')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[17]:


data_training_array = scaler.fit_transform(data_training)
data_training_array


# In[18]:


data_training_array.shape


# In[19]:


x_train = []
y_train = []
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])
x_train , y_train = np.array(x_train),np.array(y_train)


# In[20]:


x_train.shape


# In[21]:


# ML model
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
from keras.layers import Dense, Dropout,LSTM
from keras.models import sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.utils import plot_model


# In[22]:


model = Sequential()
model.add(LSTM(units= 50,activation = 'relu',return_sequences = True,
            input_shape =(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units= 60,activation = 'relu',return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units= 80,activation = 'relu',return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units= 120,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1 ))


# In[23]:


model.summary()


# In[24]:


model.compile(optimizer = 'adam' , loss = 'mean_squared_error')
model.fit(x_train, y_train , epochs= 50)


# In[25]:


model.save('keras_model.h5')


# In[26]:


data_testing.head()


# In[27]:


data_training.tail(100)


# In[28]:


past_100_days = data_training.tail(100)


# In[29]:


final_df = past_100_days.append(data_testing, ignore_index=True)


# In[30]:


final_df.head()


# In[31]:



input_data = scaler.fit_transform(data_testing)
input_data


# In[32]:


input_data.shape


# In[33]:


x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i ])
    y_test.append(input_data[i, 0])


# In[34]:


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[35]:


#making predictions
y_predicated = model.predict(x_test)


# In[36]:


y_predicated.shape


# In[37]:


y_test


# In[38]:


y_predicated


# In[39]:


scaler.scale_


# In[40]:


scale_factor =1/0.02249339
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor


# In[41]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label ='Original Price')
plt.plot(y_predicated,'r', label ='Predicated Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:




