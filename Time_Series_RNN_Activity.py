
# coding: utf-8

# In[1]:


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import pandas
import matplotlib.pyplot as plt
dataset = pandas.read_csv('/media/achint/INSOFE/RNN_Class/Time_Series_Activity/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()


# In[2]:


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[3]:


# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset
dataset = dataset.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


print(len(train), len(test))
print(train.shape)

# In[4]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[5]:


# reshape into X=t and Y=t+1
look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)
print(testX.shape)
print(len(trainX))


# ## Create an RNN that could be used for predicting number of airline passengers on the trainX, trainY data.
# 
# Since the data is normalized by the ```scalar``` object, use ```scalar.inverse_transform``` on your predictions to rescale

model=Sequential()
model.add(SimpleRNN(20, input_shape=(1,4), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')

model.fit(trainX,testX,epochs=1,verbose=0)
