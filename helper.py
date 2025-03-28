import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def loadWeatherData(filename):
    # read in the weather CSV into a np array
    # return array
    rawlst = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter= ',')
        header = next(reader)
        for row in reader:
            rawlst.append(row)
    d = {}
    for i in range(0, len(header)):
        d[i] = header[i]
    arr = np.array(rawlst, dtype = float)
    return arr , d

def removeOutliers(data):
    new = data[data[:, -1] <= 1]
    return new

def splitData(data):
    nrow, ncol = data.shape
    X = data[:,:-1]
    y = data[:, -1]
    return X, y 

def trainTest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def standardizer(X):
    mean = np.mean(X, axis = 0)
    sdv = np.std(X, axis = 0)
    standardized = (X - mean)/sdv
    return standardized

def addBias(X):
    bias = np.ones((X.shape[0],1))
    newX = np.hstack((bias, X))
    return newX

def addTimeSteps(data):
    # create a timesteps array
    timesteps = np.arange(data.shape[0])
    data = np.hstack((timesteps, data))
    return data




    
