import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data_path = 'weather_data.csv'
df = pd.read_csv(data_path)

# Use the default index as the time sequence since no timestamp column exists
df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

# Features and target
features = ['temperature', 'wind_speed', 'mean_sea_level_pressure', 'surface_solar_radiation', 'surface_thermal_radiation', 'total_cloud_cover']
target = 'relative_humidity'

# Remove outliers
def removeOutliers(data):
    return data[data[:, -1] <= 1]

# Convert dataframe to numpy array and apply outlier removal
data = df[features + [target]].values
data = removeOutliers(data)

# Split data into features and target
def splitData(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y 

X, y = splitData(data)

# Train-test split (80-20, no shuffling)
def trainTest(X, y):
    return train_test_split(X, y, test_size=0.2, shuffle=False)

X_train, X_test, y_train, y_test = trainTest(X, y)

# Standardize features
def standardizer(X):
    mean = np.mean(X, axis=0)
    sdv = np.std(X, axis=0)
    return (X - mean) / sdv

X_train = standardizer(X_train)
X_test = standardizer(X_test)

# Add bias term
def addBias(X):
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))

X_train = addBias(X_train)
X_test = addBias(X_test)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
model.fit(X_train, y_train)

# Forecasting
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# Plot actual vs predicted (only for test set)
plt.figure(figsize=(12,6))
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_test)), predictions, label='Predicted', linestyle='dashed')
plt.xlabel('Time Index')
plt.ylabel('Relative Humidity')
plt.title(f'Actual vs Predicted Relative Humidity (Test Set)\nMAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
plt.legend()
plt.savefig("xgb_plot.png")
plt.show()
