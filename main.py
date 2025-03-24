import numpy as np
import csv
import helper as h
import matplotlib.pyplot as plt

data, d = h.loadWeatherData('weather_data.csv')
#for i in range(data.shape[1]):
#   h.plotdata(data, i, d)

# remove outlier data
    # specifically rows that include humidity data that exceeded 1
data = h.removeOutliers(data)

# separate the y values 
X, y = h.splitData(data)

# split the data into train and test sets 
    # DO NOT TOUCH THE TEST SET

X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise only the training X values 

X_train = h.standardizer(X_train)

# add bias to data 
X_train = h.addBias(X_train)