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

def plotdata(data, col, d):
    y_val = data[:,col]
    name = d[col]
    time_steps = np.arange(len(data))
    plt.plot(time_steps, y_val, label = name, color = 'b', linestyle = '-')
    plt.xlabel("Time Steps")
    plt.ylabel(name)
    plt.title(f"Time Series Plot of {name}")

    save_folder = "plots"
    os.makedirs(save_folder, exist_ok=True)
    save_filename = f"{name}_plot.png"
    save_path = os.path.join(save_folder, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    df = pd.DataFrame(data, columns=header)

    # Heatmap for feature correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    
    # Save the heatmap
    heatmap_path = os.path.join(save_folder, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    
    plt.show()

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
        X.append(data[i - n_steps:i, :-1])  # Use N previous time points as features
        y.append(data[i + forecast_steps - 1, -1])  # Predict target after forecast_steps
    return np.array(X), np.array(y)

def flatten_windows(X):
    return X.reshape((X.shape[0], -1))  # Flatten (samples, N, features) → (samples, N * features)



    
