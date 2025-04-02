import csv
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Helper function to load weather data
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

# Load the actual weather data (relative humidity target)
weather_data, _ = loadWeatherData('weather_data.csv')

# Load the six LSTM output feature files (1hr, 6hr, 24hr)
features_1hr, _ = loadWeatherData('1hr_pred.csv')
features_6hr, _ = loadWeatherData('6hr_pred.csv')
features_24hr, _ = loadWeatherData('24hr_pred.csv')
features_1hr_test, _ = loadWeatherData('test_matrix_1hr.csv')
features_6hr_test, _ = loadWeatherData('test_matrix_6hr.csv')
features_24hr_test, _ = loadWeatherData('test_matrix_24hr.csv')

# Convert the loaded data to numpy arrays for easier processing
weather_data = np.array(weather_data, dtype=float)
features_1hr = np.array(features_1hr, dtype=float)
features_6hr = np.array(features_6hr, dtype=float)
features_24hr = np.array(features_24hr, dtype=float)
features_1hr_test = np.array(features_1hr_test, dtype=float)
features_6hr_test = np.array(features_6hr_test, dtype=float)
features_24hr_test = np.array(features_24hr_test, dtype=float)

# Split weather data into training and testing (80% train, 20% test)
train_data = weather_data[:int(0.8 * len(weather_data))]
test_data = weather_data[int(0.8 * len(weather_data)):]

# Features and target variable
X_train_1hr = features_1hr[:int(0.8 * len(features_1hr))]
X_test_1hr = features_1hr[int(0.8 * len(features_1hr)):]

X_train_6hr = features_6hr[:int(0.8 * len(features_6hr))]
X_test_6hr = features_6hr[int(0.8 * len(features_6hr)):]

X_train_24hr = features_24hr[:int(0.8 * len(features_24hr))]
X_test_24hr = features_24hr[int(0.8 * len(features_24hr)):]

y_train = train_data[:, -1]  # Assuming relative humidity is the last column in weather_data
y_test = test_data[:, -1]

# Train XGBoost model on 1-hour, 6-hour, and 24-hour features
def train_xgb_model(X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    return predictions, mae, mse, rmse

# Train models for different time horizons and evaluate
predictions_1hr, mae_1hr, mse_1hr, rmse_1hr = train_xgb_model(X_train_1hr, y_train, X_test_1hr, y_test)
predictions_6hr, mae_6hr, mse_6hr, rmse_6hr = train_xgb_model(X_train_6hr, y_train, X_test_6hr, y_test)
predictions_24hr, mae_24hr, mse_24hr, rmse_24hr = train_xgb_model(X_train_24hr, y_train, X_test_24hr, y_test)

# Evaluate performance metrics
metrics_df = np.array([[mae_1hr, mse_1hr, rmse_1hr], [mae_6hr, mse_6hr, rmse_6hr], [mae_24hr, mse_24hr, rmse_24hr]])
print("Model Evaluation Metrics (MAE, MSE, RMSE):")
print("Time Horizon | MAE    | MSE    | RMSE")
for i, time_horizon in enumerate(["1 Hour", "6 Hours", "24 Hours"]):
    print(f"{time_horizon}     | {metrics_df[i][0]:.4f} | {metrics_df[i][1]:.4f} | {metrics_df[i][2]:.4f}")

# Plot results for 1 Hour, 6 Hour, and 24 Hour predictions
def plot_predictions_vs_actual(predictions, actual, task_name):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title(f'Predictions vs Actual for {task_name} - XGBoost')
    plt.xlabel('Time')
    plt.ylabel('Relative Humidity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task_name}_predictions_vs_actual_XGBoost.png')
    plt.show()

plot_predictions_vs_actual(predictions_1hr, y_test, "1 Hour")
plot_predictions_vs_actual(predictions_6hr, y_test, "6 Hours")
plot_predictions_vs_actual(predictions_24hr, y_test, "24 Hours")
