import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data_path = 'weather_data.csv'
df = pd.read_csv(data_path)

# Features and target
features = ['temperature', 'wind_speed', 'mean_sea_level_pressure', 'surface_solar_radiation', 'surface_thermal_radiation', 'total_cloud_cover']
target = 'relative_humidity'

# Remove outliers
def removeOutliers(data):
    return data[data[:, -1] <= 1]

# Convert dataframe to numpy array and apply outlier removal
data = df[features + [target]].values
data = removeOutliers(data)

# Create data windows for prediction (N time points before forecast)
def create_windows(data, n_steps, forecast_steps):
    X, y = [], []
    for i in range(n_steps, len(data) - forecast_steps):
        X.append(data[i - n_steps:i, :-1])  # N previous time points as features
        y.append(data[i + forecast_steps - 1, -1])  # Relative humidity after forecast_steps
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to be 2D (samples, features)
    X = X.reshape((X.shape[0], -1))  # Flatten the time steps into a single dimension (n_samples, n_features)
    
    return X, y

# Train-test split (80-20, no shuffling, time-based split)
def time_split(data, test_size=0.2):
    # Split data into 100 point groups and assign 80% for training, 20% for testing
    n = len(data)
    test_len = int(n * test_size)
    
    # The training set is the first portion, and the test set is the subsequent portion
    train_data = data[:n - test_len]
    test_data = data[n - test_len:]
    
    return train_data, test_data

# Create features for 1-hour, 6-hour, and 24-hour prediction tasks
n_steps = 24  # Example for using the past 6 time points (adjustable)

# Split data into train and test sets
train_data, test_data = time_split(data)

# Task 1: Predict relative humidity for the next hour (forecast_steps = 1)
X_train_1h, y_train_1h = create_windows(train_data, n_steps, 1)
X_test_1h, y_test_1h = create_windows(test_data, n_steps, 1)

# Task 2: Predict relative humidity for 6 hours ahead (forecast_steps = 6)
X_train_6h, y_train_6h = create_windows(train_data, n_steps, 6)
X_test_6h, y_test_6h = create_windows(test_data, n_steps, 6)

# Task 3: Predict relative humidity for 24 hours ahead (forecast_steps = 24)
X_train_24h, y_train_24h = create_windows(train_data, n_steps, 24)
X_test_24h, y_test_24h = create_windows(test_data, n_steps, 24)

# Train XGBoost model for each task and evaluate

# Function to train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # Forecasting
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    return predictions, y_test, mae, mse, rmse

# Task 1: Evaluate model for next hour prediction
predictions_1h, y_test_1h, mae_1h, mse_1h, rmse_1h = train_and_evaluate(X_train_1h, X_test_1h, y_train_1h, y_test_1h)

# Task 2: Evaluate model for 6-hour ahead prediction
predictions_6h, y_test_6h, mae_6h, mse_6h, rmse_6h = train_and_evaluate(X_train_6h, X_test_6h, y_train_6h, y_test_6h)

# Task 3: Evaluate model for 24-hour ahead prediction
predictions_24h, y_test_24h, mae_24h, mse_24h, rmse_24h = train_and_evaluate(X_train_24h, X_test_24h, y_train_24h, y_test_24h)

# Plot predictions vs actual values for all three tasks

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

# Plot for 1-hour predictions
plot_predictions_vs_actual(predictions_1h, y_test_1h, "1 Hour")

# Plot for 6-hour predictions
plot_predictions_vs_actual(predictions_6h, y_test_6h, "6 Hours")

# Plot for 24-hour predictions
plot_predictions_vs_actual(predictions_24h, y_test_24h, "24 Hours")

# Store results in a matrix for easier visualization
metrics = np.array([[mae_1h, mse_1h, rmse_1h],
                    [mae_6h, mse_6h, rmse_6h],
                    [mae_24h, mse_24h, rmse_24h]])

# Create a DataFrame for better readability
metrics_df = pd.DataFrame(metrics, columns=['MAE', 'MSE', 'RMSE'], index=['1 Hour', '6 Hours', '24 Hours'])

# Plot the matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='coolwarm', cbar=True)
plt.title('Model Evaluation Metrics (MAE, MSE, RMSE) - XGBoost')
plt.tight_layout()
plt.savefig('metrics_heatmap (XGBoost).png')
plt.show()

# Optionally print the results to the console
print(metrics_df)
