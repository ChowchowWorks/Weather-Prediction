import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load dataset
data_path = 'weather_data.csv'
df = pd.read_csv(data_path)

df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

# Features and target
features = ['temperature', 'wind_speed', 'mean_sea_level_pressure', 'surface_solar_radiation', 'surface_thermal_radiation', 'total_cloud_cover']
target = 'relative_humidity'

# Remove outliers
def removeOutliers(data):
    return data[data[:, -1] <= 1]

data = df[features + [target]].values
data = removeOutliers(data)

# Split data into features and target
def splitData(data):
    X = data[:, :-1]  # Only features, no target variable
    y = data[:, -1]   # Target variable
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

# Helper function to create sequences for different forecast horizons
def create_sequences(data, target, time_steps, forecast_steps):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_steps):
        X.append(data[i:i+time_steps])  # Input sequence of features (X)
        y.append(target[i+time_steps+forecast_steps-1])  # Output (target) sequence (y)
    return np.array(X), np.array(y)

# Define forecast horizons for 1h, 6h, and 24h predictions
forecast_steps_1h = 1
forecast_steps_6h = 6
forecast_steps_24h = 24

# Create sequences for all forecast horizons
time_steps = 24  # Using past 24 hours for prediction

X_train_seq_1h, y_train_seq_1h = create_sequences(X_train, y_train, time_steps, forecast_steps_1h)
X_test_seq_1h, y_test_seq_1h = create_sequences(X_test, y_test, time_steps, forecast_steps_1h)

X_train_seq_6h, y_train_seq_6h = create_sequences(X_train, y_train, time_steps, forecast_steps_6h)
X_test_seq_6h, y_test_seq_6h = create_sequences(X_test, y_test, time_steps, forecast_steps_6h)

X_train_seq_24h, y_train_seq_24h = create_sequences(X_train, y_train, time_steps, forecast_steps_24h)
X_test_seq_24h, y_test_seq_24h = create_sequences(X_test, y_test, time_steps, forecast_steps_24h)

# Build RNN Model
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=input_shape),
        SimpleRNN(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train and evaluate the model for a given forecast horizon
def train_and_evaluate_model(X_train_seq, X_test_seq, y_train_seq, y_test_seq):
    model = build_rnn_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    history = model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)
    
    # Predictions for the test set
    y_pred = model.predict(X_test_seq)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test_seq, y_pred)
    mse = mean_squared_error(y_test_seq, y_pred)
    rmse = np.sqrt(mse)
    return y_pred, y_test_seq, mae, mse, rmse

# Evaluate models for 1h, 6h, and 24h predictions
y_pred_1h, y_test_1h, mae_1h, mse_1h, rmse_1h = train_and_evaluate_model(X_train_seq_1h, X_test_seq_1h, y_train_seq_1h, y_test_seq_1h)
y_pred_6h, y_test_6h, mae_6h, mse_6h, rmse_6h = train_and_evaluate_model(X_train_seq_6h, X_test_seq_6h, y_train_seq_6h, y_test_seq_6h)
y_pred_24h, y_test_24h, mae_24h, mse_24h, rmse_24h = train_and_evaluate_model(X_train_seq_24h, X_test_seq_24h, y_train_seq_24h, y_test_seq_24h)

# Plot Actual vs Predicted for 1h, 6h, and 24h predictions

def plot_predictions_vs_actual(predictions, actual, task_name):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual", color='blue')
    plt.plot(predictions, label="Predicted", color='red', linestyle='--')
    plt.legend()
    plt.title(f"Actual vs Predicted for {task_name} Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Relative Humidity")
    plt.savefig(f"{task_name}_predictions_vs_actual.png")
    plt.show()

# Plot for 1-hour predictions
plot_predictions_vs_actual(y_pred_1h, y_test_1h, "1 Hour")

# Plot for 6-hour predictions
plot_predictions_vs_actual(y_pred_6h, y_test_6h, "6 Hours")

# Plot for 24-hour predictions
plot_predictions_vs_actual(y_pred_24h, y_test_24h, "24 Hours")

# Error Plots for 1h, 6h, and 24h predictions
def plot_error(predictions, actual, task_name):
    error = predictions - actual
    plt.figure(figsize=(10, 5))
    plt.plot(error, label="Error", color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.title(f"Error Plot for {task_name} Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Prediction Error")
    plt.savefig(f"{task_name}_error_plot.png")
    plt.show()

# Error plot for 1-hour predictions
plot_error(y_pred_1h, y_test_1h, "1 Hour")

# Error plot for 6-hour predictions
plot_error(y_pred_6h, y_test_6h, "6 Hours")

# Error plot for 24-hour predictions
plot_error(y_pred_24h, y_test_24h, "24 Hours")

# Metrics Matrix for MAE, MSE, RMSE
metrics = np.array([[mae_1h, mse_1h, rmse_1h],
                    [mae_6h, mse_6h, rmse_6h],
                    [mae_24h, mse_24h, rmse_24h]])

metrics_df = pd.DataFrame(metrics, columns=['MAE', 'MSE', 'RMSE'], index=['1 Hour', '6 Hours', '24 Hours'])

# Plot the metrics as a heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='coolwarm', cbar=True)
plt.title('Model Evaluation Metrics (MAE, MSE, RMSE)')
plt.tight_layout()
plt.savefig('metrics_heatmap.png')
plt.show()

# Save predictions and actual values for all three tasks
output_df = pd.DataFrame({
    'Actual_1h': y_test_1h.flatten(),
    'Predicted_1h': y_pred_1h.flatten(),
    'Actual_6h': y_test_6h.flatten(),
    'Predicted_6h': y_pred_6h.flatten(),
    'Actual_24h': y_test_24h.flatten(),
    'Predicted_24h': y_pred_24h.flatten(),
})

output_df.to_csv('predictions_results.csv', index=False)
