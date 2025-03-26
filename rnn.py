import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

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

# Helper function to create sequences
def create_sequences(data, target, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])  # Input sequence of features (X)
        y.append(target[i+time_steps])  # Output (target) sequence (y)
    return np.array(X), np.array(y)

time_steps = 24  # Using past 24 hours
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

# Build RNN Model
model = Sequential([
    SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(time_steps, X_train_seq.shape[2])),
    SimpleRNN(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)

# Predictions for the entire test set (20% of the data)
y_pred = model.predict(X_test_seq)

# Evaluation Metrics
mae = mean_absolute_error(y_test_seq, y_pred)
mse = mean_squared_error(y_test_seq, y_pred)
rmse = np.sqrt(mse)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Plot Actual vs Predicted for the entire test set
plt.figure(figsize=(10, 5))
plt.plot(y_test_seq, label="Actual", color='blue')  # Plot the entire test set
plt.plot(y_pred, label="Predicted", color='red', linestyle='dashed')  # Plot the entire predictions
plt.legend()
plt.title("Actual vs Predicted Relative Humidity")
plt.xlabel("Time Steps")
plt.ylabel("Relative Humidity")
plt.savefig("rnn_plot_full.png")
plt.show()

# Save predictions and actual values
output_df = pd.DataFrame({
    'Actual': y_test_seq.flatten(),
    'Predicted': y_pred.flatten()
})
output_df.to_csv("predictions.csv", index=False)
