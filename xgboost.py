import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Helper Functions
def loadWeatherData(filename):
    rawlst = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        for row in reader:
            rawlst.append(row)
    d = {i: header[i] for i in range(len(header))}
    arr = np.array(rawlst, dtype=float)
    return arr, d

def removeOutliers(data):
    return data[data[:, -1] <= 1]

def splitData(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def trainTest(X, y):
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def standardizer(X):
    mean = np.mean(X, axis=0)
    sdv = np.std(X, axis=0)
    return (X - mean) / sdv

def addBias(X):
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))

def create_windows(data, n_steps, forecast_steps):
    X, y = [], []
    for i in range(n_steps, len(data) - forecast_steps):
        X.append(data[i - n_steps:i, :-1])
        y.append(data[i + forecast_steps - 1, -1])
    return np.array(X), np.array(y)

def flatten_windows(X):
    return X.reshape((X.shape[0], -1))

# Load dataset
data_path = 'weather_data.csv'
data, feature_dict = loadWeatherData(data_path)

# Remove outliers
data = removeOutliers(data)

# Split data into features and target
X, y = splitData(data)

# Standardize features
X = standardizer(X)

# Train-test split
X_train, X_test, y_train, y_test = trainTest(X, y)

# Set number of past time steps for prediction
n_steps = 24

# Create windows for different forecast horizons
X_train_1h, y_train_1h = create_windows(np.hstack((X_train, y_train.reshape(-1,1))), n_steps, 1)
X_test_1h, y_test_1h = create_windows(np.hstack((X_test, y_test.reshape(-1,1))), n_steps, 1)

X_train_6h, y_train_6h = create_windows(np.hstack((X_train, y_train.reshape(-1,1))), n_steps, 6)
X_test_6h, y_test_6h = create_windows(np.hstack((X_test, y_test.reshape(-1,1))), n_steps, 6)

X_train_24h, y_train_24h = create_windows(np.hstack((X_train, y_train.reshape(-1,1))), n_steps, 24)
X_test_24h, y_test_24h = create_windows(np.hstack((X_test, y_test.reshape(-1,1))), n_steps, 24)

# Flatten windows
X_train_1h, X_test_1h = flatten_windows(X_train_1h), flatten_windows(X_test_1h)
X_train_6h, X_test_6h = flatten_windows(X_train_6h), flatten_windows(X_test_6h)
X_train_24h, X_test_24h = flatten_windows(X_train_24h), flatten_windows(X_test_24h)

# Train and evaluate function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, objective='reg:squarederror')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return predictions, y_test, mae, mse, rmse

# Evaluate models
predictions_1h, y_test_1h, mae_1h, mse_1h, rmse_1h = train_and_evaluate(X_train_1h, X_test_1h, y_train_1h, y_test_1h)
predictions_6h, y_test_6h, mae_6h, mse_6h, rmse_6h = train_and_evaluate(X_train_6h, X_test_6h, y_train_6h, y_test_6h)
predictions_24h, y_test_24h, mae_24h, mse_24h, rmse_24h = train_and_evaluate(X_train_24h, X_test_24h, y_train_24h, y_test_24h)

# Plot results
def plot_predictions_vs_actual(predictions, actual, task_name):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title(f'Predictions vs Actual for {task_name} - XGBoost')
    plt.xlabel('Time')
    plt.ylabel('Relative Humidity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task_name}_predictions_vs_actual (XGBoost).png')
    plt.show()

plot_predictions_vs_actual(predictions_1h, y_test_1h, "1 Hour")
plot_predictions_vs_actual(predictions_6h, y_test_6h, "6 Hours")
plot_predictions_vs_actual(predictions_24h, y_test_24h, "24 Hours")

# Store results in a DataFrame
metrics_df = pd.DataFrame(
    [[mae_1h, mse_1h, rmse_1h], [mae_6h, mse_6h, rmse_6h], [mae_24h, mse_24h, rmse_24h]],
    columns=['MAE', 'MSE', 'RMSE'],
    index=['1 Hour', '6 Hours', '24 Hours']
)

print(metrics_df)
