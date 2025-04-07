import csv
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import helper as h
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

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
features_1hr, _ = loadWeatherData('feature_matrix_1hr.csv')
features_6hr, _ = loadWeatherData('feature_matrix_6hr.csv')
features_24hr, _ = loadWeatherData('feature_matrix_24hr.csv')
features_1hr_test, _ = loadWeatherData('test_matrix_1hr.csv')
features_6hr_test, _ = loadWeatherData('test_matrix_6hr.csv')
features_24hr_test, _ = loadWeatherData('test_matrix_24hr.csv')

X_train_1hr , y_train_1hr = h.splitData(features_1hr)
X_train_6hr , y_train_6hr = h.splitData(features_6hr)
X_train_24hr , y_train_24hr = h.splitData(features_24hr)

X_test_1hr , y_test_1hr = h.splitData(features_1hr_test)
X_test_6hr, y_test_6hr = h.splitData(features_6hr_test)
X_test_24hr , y_test_24hr = h.splitData(features_24hr_test)

class xgbmodel():
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, min_child_weight = 1, max_delta_step = 1, reg_alpha=0, reg_lambda=1):
        self.n_estimators = None
        self.max_depth = None
        self.learning_rate = None
        self.subsample = None
        self.colsample_bytree = None
        self.reg_alpha = None
        self.reg_lambda = None
        self.min_child_weight = self.max_delta_step = None
        self.model = None

    def build_model(self):
        model = XGBRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth, 
            learning_rate = self.learning_rate,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            min_child_weight = self.min_child_weight,
            max_delta_step = self.max_delta_step,
            reg_alpha = self.reg_alpha,
            reg_lambda = self.reg_lambda
        )
        self.model = model
        return model
    def predict(self, X_test, y_test):
        if self.model == None:
            raise ValueError("Model is not trained. Call fit() first.")
        predictions = self.model.predict(X_test)

        # Evaluate performance
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
    
        return predictions, mae, mse, rmse
    
    def update_params(self, d):
        self.n_estimators = d['n_estimators']
        self.max_depth = d['max_depth']
        self.learning_rate = d['learning_rate']
        self.subsample = d['subsample']
        self.colsample_bytree = d['colsample_bytree']
        self.min_child_weight = d['min_child_weight']
        self.max_delta_step = d['max_delta_step']
        self.reg_alpha = d['reg_alpha']
        self.reg_lambda = d['reg_lambda']



def objective_1hr(trial):
    # Suggest hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step = 1),
        "max_delta_step" : trial.suggest_int("max_delta_step", 0 , 10, step = 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True)
    }
    
    # Train XGBoost model
    model = xgbmodel(**params)  
    
    # Use cross-validation to evaluate the model
    score = cross_val_score(model.build_model(), X_train_1hr, y_train_1hr, cv=5, scoring='neg_mean_absolute_error')
    
    return score.mean()  # Optuna maximizes this (since MAE is negative)

# Run optimization
study_1hr = optuna.create_study(direction="maximize")  # Maximize negative MAE
study_1hr.optimize(objective_1hr, n_trials=30)

best_params_1hr = study_1hr.best_params

# Best hyperparameters
print("Best hyperparameters:", study_1hr.best_params)

def objective_6hr(trial):
    # Suggest hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step = 1),
        "max_delta_step" : trial.suggest_int("max_delta_step", 0 , 10, step = 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True)
    }
    
    # Train XGBoost model
    model = xgbmodel(**params)  
    
    # Use cross-validation to evaluate the model
    score = cross_val_score(model.build_model(), X_train_1hr, y_train_1hr, cv=5, scoring='neg_mean_absolute_error')
    
    return score.mean()  # Optuna maximizes this (since MAE is negative)

# Run optimization
study_6hr = optuna.create_study(direction="maximize")  # Maximize negative MAE
study_6hr.optimize(objective_6hr, n_trials=30)

best_params_6hr = study_6hr.best_params

# Best hyperparameters
print("Best hyperparameters:", study_6hr.best_params)


def objective_24hr(trial):
    # Suggest hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step = 1),
        "max_delta_step" : trial.suggest_int("max_delta_step", 0 , 10, step = 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True)
    }
    
    # Train XGBoost model
    model = xgbmodel(**params)  
    
    # Use cross-validation to evaluate the model
    score = cross_val_score(model.build_model(), X_train_1hr, y_train_1hr, cv=5, scoring='neg_mean_absolute_error')
    
    return score.mean()  # Optuna maximizes this (since MAE is negative)

# Run optimization
study_24hr = optuna.create_study(direction="maximize")  # Maximize negative MAE
study_24hr.optimize(objective_24hr, n_trials=30)

best_params_24hr = study_24hr.best_params

# Best hyperparameters
print("Best hyperparameters:", study_24hr.best_params)
xg_1hr = xgbmodel()
xg_6hr = xgbmodel()
xg_24hr = xgbmodel()

xg_1hr.update_params(best_params_1hr)
xg_6hr.update_params(best_params_6hr)
xg_24hr.update_params(best_params_24hr)

xg_1hr.build_model()
xg_6hr.build_model()
xg_24hr.build_model()

xg_1hr.model.fit(X_train_1hr, y_train_1hr)
xg_1hr.model.save_model('nnxg_1hr.model')
xg_6hr.model.fit(X_train_6hr, y_train_6hr)
xg_6hr.model.save_model('nnxg_6hr.model')
xg_24hr.model.fit(X_train_24hr, y_train_24hr)
xg_24hr.model.save_model('nnxg_24hr.model')

# Train models for different time horizons and evaluate
predictions_1hr, mae_1hr, mse_1hr, rmse_1hr = xg_1hr.predict(X_test_1hr, y_test_1hr)
predictions_6hr, mae_6hr, mse_6hr, rmse_6hr = xg_6hr.predict(X_test_6hr, y_test_6hr)
predictions_24hr, mae_24hr, mse_24hr, rmse_24hr = xg_24hr.predict(X_test_24hr, y_test_24hr)



# Evaluate performance metrics
metrics_df = pd.DataFrame(
    [[mae_1hr, mse_1hr, rmse_1hr], [mae_6hr, mse_6hr, rmse_6hr], [mae_24hr, mse_24hr, rmse_24hr]],
    columns=['MAE', 'MSE', 'RMSE'],
    index=['1 Hour', '6 Hours', '24 Hours']
)

print(metrics_df)

# Plot results for 1 Hour, 6 Hour, and 24 Hour predictions
def plot_predictions_vs_actual(predictions, actual, task_name):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title(f'Predictions vs Actual for {task_name} - LSTM_XGBoost')
    plt.xlabel('Time')
    plt.ylabel('Relative Humidity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task_name}_predictions_vs_actual_XGBoost.png')
    plt.show()
    plt.close()

plot_predictions_vs_actual(predictions_1hr, y_test_1hr, "1 Hour")
plot_predictions_vs_actual(predictions_6hr, y_test_6hr, "6 Hours")
plot_predictions_vs_actual(predictions_24hr, y_test_24hr, "24 Hours")

residuals_1hr = y_test_1hr - predictions_1hr
residuals_6hr = y_test_6hr - predictions_6hr
residuals_24hr = y_test_24hr - predictions_24hr

plt.hist(residuals_1hr, bins= 45)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
plt.close()
plt.hist(residuals_6hr, bins = 45)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
plt.close()
plt.hist(residuals_24hr, bins = 45)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
plt.close()