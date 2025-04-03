import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from itertools import product
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from skopt.space import Integer, Real
from skopt import BayesSearchCV
import optuna
import helper as h

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

####### EVALUATE BEST DEPTH #######

def find_best_max_depth_cv(X_train, y_train, n_splits=5, label=""):
    max_depth_values = range(3, 13)
    mean_mae_scores = []

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for depth in max_depth_values:
        fold_mae_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=depth, objective='reg:squarederror')
            model.fit(X_train_fold, y_train_fold)

            y_val_pred = model.predict(X_val_fold)
            fold_mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))

        mean_mae_scores.append(np.mean(fold_mae_scores))

    # Plot Max Depth vs MAE
    plt.figure(figsize=(8, 5))
    plt.plot(max_depth_values, mean_mae_scores, marker='o', linestyle='-', color='b', label='CV MAE')
    plt.xlabel("Max Depth")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Max Depth vs CV MAE for {label}")
    plt.legend()
    plt.grid()
    plt.show()

    best_depth = max_depth_values[np.argmin(mean_mae_scores)]
    print(f"Best Max Depth for {label} (Based on CV MAE):", best_depth)
    return best_depth

####### EVALUATE BEST LEARNING RATE #######

def find_best_learning_rate_cv(X_train, y_train, n_splits=5, label=""):
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    mean_mae_scores = []

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for lr in learning_rates:
        fold_mae_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = XGBRegressor(n_estimators=100, learning_rate=lr, max_depth=7, objective='reg:squarederror')
            model.fit(X_train_fold, y_train_fold)

            y_val_pred = model.predict(X_val_fold)
            fold_mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))

        mean_mae_scores.append(np.mean(fold_mae_scores))

    # Plot Learning Rate vs MAE
    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, mean_mae_scores, marker='o', linestyle='-', color='g', label='CV MAE')
    plt.xlabel("Learning Rate")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Learning Rate vs CV MAE for {label}")
    plt.legend()
    plt.grid()
    plt.show()

    best_lr = learning_rates[np.argmin(mean_mae_scores)]
    print(f"Best Learning Rate for {label} (Based on CV MAE):", best_lr)
    return best_lr

####### EVALUATE BEST ESTIMATOR #######

def find_best_estimators_cv(X_train, y_train, n_splits=5, label=""):
    estimators_values = [50, 100, 150, 200, 250, 300, 350, 400]      # Define a range of n_estimators to test
    mean_mae_scores = []

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for estimators in estimators_values:
        fold_mae_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = XGBRegressor(n_estimators=estimators, learning_rate=0.1, max_depth=7, objective='reg:squarederror')
            model.fit(X_train_fold, y_train_fold)

            y_val_pred = model.predict(X_val_fold)
            fold_mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))

        mean_mae_scores.append(np.mean(fold_mae_scores))

    # Plot n_estimators vs MAE
    plt.figure(figsize=(8, 5))
    plt.plot(estimators_values, mean_mae_scores, marker='o', linestyle='-', color='b', label='CV MAE')
    plt.xlabel("Number of Estimators")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Number of Estimators vs CV MAE")
    plt.legend()
    plt.grid()
    plt.show()

    best_estimators = estimators_values[np.argmin(mean_mae_scores)]
    print(f"Best Estimators for {label} (Based on CV MAE):", best_estimators)
    return best_estimators

####### GRID SEARCH TO FIND BEST PARAMETERS #######

def grid_search_xgb(X_train, y_train, param_grid, n_splits=5, label=""):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    param_combinations = list(product(
        param_grid['max_depth'], param_grid['learning_rate'], param_grid['n_estimators']
    ))
    
    results = []
    
    def evaluate_params(params):
        max_depth, learning_rate, n_estimators = params
        fold_mae_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model = XGBRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, objective='reg:squarederror',
                early_stopping_rounds=10, eval_metric='mae'
            )
            
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            y_val_pred = model.predict(X_val_fold)
            fold_mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))
        
        avg_mae = np.mean(fold_mae_scores)
        results.append((max_depth, learning_rate, n_estimators, avg_mae))
        return params, avg_mae
    
    Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in param_combinations)
    
    results = sorted(results, key=lambda x: x[3])
    best_params, best_mae = (results[0][:3], results[0][3])
    print(f"Best Params for {label}: {best_params} with MAE: {best_mae:.4f}")
    
    # Convert results to numpy array for plotting
    results_arr = np.array(results)
    
    # 3D Scatter plot for max_depth vs learning_rate vs MAE
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(results_arr[:, 0], results_arr[:, 1], results_arr[:, 3], c=results_arr[:, 3], cmap='viridis')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('MAE')
    ax.set_title(f'MAE across Hyperparameters for {label}')
    plt.show()
    
    # Heatmap for n_estimators vs MAE
    heatmap_data = {(d, n): mae for d, lr, n, mae in results}
    depths = sorted(set(results_arr[:, 0]))
    n_estims = sorted(set(results_arr[:, 2]))
    heatmap_matrix = np.array([[heatmap_data.get((d, n), np.nan) for n in n_estims] for d in depths])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_matrix, xticklabels=n_estims, yticklabels=depths, annot=True, cmap='coolwarm')
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.title(f'Heatmap of MAE for {label}')
    plt.show()
    
    return best_params

####### BAYESIAN OPTIMISATION TO FIND BEST PARAMETERS #######

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    }

    model = XGBRegressor(objective='reg:squarederror', **params, random_state=42)
    model.fit(X_train, y_train)

    Y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, Y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)

best_xgb = XGBRegressor(objective='reg:squarederror', **study.best_params, random_state=42)
best_xgb.fit(X_train, y_train)

Y_pred_xgb = best_xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, Y_pred_xgb)
mse_xgb = mean_squared_error(y_test, Y_pred_xgb)

print(f"Tuned XGBoost - MAE: {mae_xgb}, MSE: {mse_xgb}")
