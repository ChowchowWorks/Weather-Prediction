import numpy as np
import helper as h
import lstm 
import torch
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

data, d = h.loadWeatherData('weather_data.csv')
data = h.removeOutliers(data)
#for i in range(data.shape[1]):
#   h.plotdata(data, i, d)

# remove outlier data
    # specifically rows that include humidity data that exceeded 1


# separate the y values 
X, y = h.splitData(data)

# split the data into train and test sets 
    # DO NOT TOUCH THE TEST SET

X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise only the training X values 

X_train = h.standardizer(X_train)

# add bias to data 
X_train = h.addBias(X_train)

# LSTM Model
X_train = torch.tensor(X_train, dtype= torch.float32)
y_train = torch.tensor(y_train, dtype= torch.float32)
X_test = torch.tensor(X_test, dtype= torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32)

nn1 = lstm.LSTM(X_train.shape, 1)

#train model 
nn1 = lstm.trainModel(nn1, dataloader, 5)
# run model 
y_pred = lstm.runModel(nn1, X_test)
# compute MSE
mse = lstm.computeMSE(y_pred, y_test)
