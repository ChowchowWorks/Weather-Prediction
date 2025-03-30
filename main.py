import numpy as np
import helper as h
import lstm 
import torch
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

data, d = h.loadWeatherData('weather_data.csv')

# remove outlier data
    # specifically rows that include humidity data that exceeded 1
data = h.removeOutliers(data)
#for i in range(data.shape[1]):
#   h.plotdata(data, i, d)

# separate the y values 
X, y = h.splitData(data)

# split the data into train and test sets 
    # DO NOT TOUCH THE TEST SET

X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise only the training X values 

X_train = h.standardizer(X_train)
X_test = h.standardizer(X_test)

# add bias to data 
X_train = h.addBias(X_train)
X_test = h.addBias(X_test)

