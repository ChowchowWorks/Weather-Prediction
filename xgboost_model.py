import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import helper as h
import hyperparameters as hp

####### PREPARATION OF DATA SET #######
# Load dataset
data_path = 'weather_data.csv'
data, feature_dict = h.loadWeatherData(data_path)

# Remove outliers
data = h.removeOutliers(data)

# Split data into features and target
X, y = h.splitData(data)

# Standardize features
X = h.standardizer(X)

# Train-test split
X_train, X_test, y_train, y_test = h.trainTest(X, y)

# Set number of past time steps for prediction
n_steps = 24

# Create windows for different forecast horizons
X_train_1h, y_train_1h = h.create_windows(np.hstack((X_train, y_train.reshape(-1, 1))), n_steps, 1)
X_test_1h, y_test_1h = h.create_windows(np.hstack((X_test, y_test.reshape(-1, 1))), n_steps, 1)

X_train_6h, y_train_6h = h.create_windows(np.hstack((X_train, y_train.reshape(-1, 1))), n_steps, 6)
X_test_6h, y_test_6h = h.create_windows(np.hstack((X_test, y_test.reshape(-1, 1))), n_steps, 6)

X_train_24h, y_train_24h = h.create_windows(np.hstack((X_train, y_train.reshape(-1, 1))), n_steps, 24)
X_test_24h, y_test_24h = h.create_windows(np.hstack((X_test, y_test.reshape(-1, 1))), n_steps, 24)

# Flatten windows
X_train_1h, X_test_1h = h.flatten_windows(X_train_1h), h.flatten_windows(X_test_1h)
X_train_6h, X_test_6h = h.flatten_windows(X_train_6h), h.flatten_windows(X_test_6h)
X_train_24h, X_test_24h = h.flatten_windows(X_train_24h), h.flatten_windows(X_test_24h)

####### MODEL CREATION AND VALIDATION #######

# Function for cross-validation
def cross_validate_XGB_model(X_train, y_train, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    
    best_model = None
    best_mae = float('inf')
    
    for train_idx, val_idx in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_val_fold)
        
        mae = mean_absolute_error(y_val_fold, predictions)
        
        if mae < best_mae:
            best_mae = mae
            best_model = model
        
        mae_scores.append(mae)
    
    return best_model, np.mean(mae_scores)

# Function to evaluate the model on the test set (MAE, MSE, RMSE) and return predictions
def evaluate_XGB_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return predictions, mae, mse, rmse

# Find best depth for each forecast horizon with labels
best_depth_1h = hp.find_best_max_depth_cv(X_train_1h, y_train_1h, label="1 Hour")
best_depth_6h = hp.find_best_max_depth_cv(X_train_6h, y_train_6h, label="6 Hours")
best_depth_24h = hp.find_best_max_depth_cv(X_train_24h, y_train_24h, label="24 Hours")

# Find best learning rate for each forecast horizon with labels
best_lr_1h = hp.find_best_learning_rate_cv(X_train_1h, y_train_1h, label="1 Hour")
best_lr_6h = hp.find_best_learning_rate_cv(X_train_6h, y_train_6h, label="6 Hours")
best_lr_24h = hp.find_best_learning_rate_cv(X_train_24h, y_train_24h, label="24 Hours")

# Find bestestimators for each forecast horizon with labels
best_estimators_1h = hp.find_best_estimators_cv(X_train_1h, y_train_1h, label="1 Hour")
best_estimators_6h = hp.find_best_estimators_cv(X_train_6h, y_train_6h, label="6 Hour")
best_estimators_24h = hp.find_best_estimators_cv(X_train_24h, y_train_24h, label="24 Hour")

model_1h = XGBRegressor(n_estimators=best_estimators_1h, learning_rate=best_lr_1h, max_depth=best_depth_1h, objective='reg:squarederror')
model_6h = XGBRegressor(n_estimators=best_estimators_6h, learning_rate=best_lr_6h, max_depth=best_depth_6h, objective='reg:squarederror')
model_24h = XGBRegressor(n_estimators=best_estimators_24h, learning_rate=best_lr_24h, max_depth=best_depth_24h, objective='reg:squarederror')

# Train final model using the best from cross-validation
best_model_1h, mae_1h = cross_validate_XGB_model(X_train_1h, y_train_1h, model_1h)
best_model_6h, mae_6h = cross_validate_XGB_model(X_train_6h, y_train_6h, model_6h)
best_model_24h, mae_24h = cross_validate_XGB_model(X_train_24h, y_train_24h, model_24h)

# Evaluate the model on the test set and get predictions and performance metrics
predictions_1h, mae_1h, mse_1h, rmse_1h = evaluate_XGB_model(best_model_1h, X_test_1h, y_test_1h)
predictions_6h, mae_6h, mse_6h, rmse_6h = evaluate_XGB_model(best_model_6h, X_test_6h, y_test_6h)
predictions_24h, mae_24h, mse_24h, rmse_24h = evaluate_XGB_model(best_model_24h, X_test_24h, y_test_24h)

####### FINAL RESULTS #######

h.evaluation_metrics(mae_1h, mae_6h, mae_24h, mse_1h, mse_6h, mse_24h, rmse_1h, rmse_6h, rmse_24h)

# Plot results for the test set using the predictions
h.plot_predictions_vs_actual(predictions_1h, y_test_1h, "1 Hour", "XGBoost")
h.plot_predictions_vs_actual(predictions_6h, y_test_6h, "6 Hours", "XGBoost")
h.plot_predictions_vs_actual(predictions_24h, y_test_24h, "24 Hours", "XGBoost")
