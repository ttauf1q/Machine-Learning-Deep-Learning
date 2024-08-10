#importing libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#initializing dataframe
dataset_train = pd.read_csv('D:/Deep Learning/Recurrent Neural Networks (RNN)/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#feature scaling using normalization formula
from sklearn.preprocessing import MinMaxScaler
sc =  MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#splitting the data for training 
X_train = []
y_train = []
for i in range (60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train,y_train = np.array(X_train), np.array(y_train)

#reshaping the training data into 3D for future manipulation
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#importing the libraries needed 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Intialising the RNN
regressor = Sequential()

#Adding First LSTM Layers in the model and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape=( X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding Second LSTM Layers in the model and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding Third LSTM Layers in the model and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding Fourth LSTM Layers in the model and dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding Output Layer
regressor.add(Dense(units= 1))

#Adding optimizer in the layer
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs= 100, batch_size= 32)