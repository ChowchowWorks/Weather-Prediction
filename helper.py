import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

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

def create_windows(data, n_steps, forecast_steps):
    X, y = [], []
    for i in range(n_steps, len(data) - forecast_steps):
        X.append(data[i - n_steps:i, :-1])
        y.append(data[i + forecast_steps - 1, -1])
    return np.array(X), np.array(y)

def flatten_windows(X):
    return X.reshape((X.shape[0], -1))

# Plot predictions vs actual values
def plot_predictions_vs_actual(predictions, actual, task_name, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title(f'Predictions vs Actual for {task_name} - {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Relative Humidity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task_name}_predictions_vs_actual ({model_name}).png')
    plt.show()

# Function to store evaluation metrics in a DataFrame and print it
def evaluation_metrics(mae_1h, mae_6h, mae_24h, mse_1h, mse_6h, mse_24h, rmse_1h, rmse_6h, rmse_24h):
    # Create a DataFrame with the evaluation metrics
    metrics_df = pd.DataFrame({
        'MAE': [mae_1h, mae_6h, mae_24h],
        'MSE': [mse_1h, mse_6h, mse_24h],
        'RMSE': [rmse_1h, rmse_6h, rmse_24h]
    }, index=['1 Hour', '6 Hours', '24 Hours'])

    # Print the metrics DataFrame
    print(metrics_df)
    return metrics_df

def plotData(x, y, *figname):
    fig, ax = plt.subplots()  # Create a figure and an axis
    ax.plot(x, y, marker='o', linestyle='-')  # Plot the data

    # Save the figure if a filename is provided
    if figname:
        plt.savefig(figname[0])  # Take the first argument as the filename

    plt.show()  
    plt.close()
    
def MSE(actual, pred):
    mse = mean_squared_error(actual, pred)
    return mse
