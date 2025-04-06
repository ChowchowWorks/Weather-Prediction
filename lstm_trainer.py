import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("weather_data.csv")

# adds a column of bias
def addBias_df(dataframe):
  dataframe = dataframe.assign(bias = 1)
  # move bias to the first column
  dataframe = dataframe[['bias'] + [col for col in dataframe.columns if col != 'bias']]
  return dataframe

# removes outliers from humidity column only
def removeOutliers(dataframe):
  dataframe = dataframe[dataframe['relative_humidity'] <= 1]
  return dataframe

# splits the data into X values and Y values (returns arrays)
def splitDataIntoArray(dataframe):
  data = dataframe.to_numpy()
  X = data[:, :-1]
  y = data[:, -1]
  return X, y

# converts array to DF
def toDF(array):
  df = pd.DataFrame(array)
  return df

# this is the standardisation scaler
scaler = StandardScaler()

# creating windows for prediction
def create_windows(data, n_steps, forecast_steps):
    X, y = [], []
    for i in range(n_steps, len(data) - forecast_steps):
        X.append(data[i - n_steps:i, :-1])  # Use N previous time points as features
        y.append(data[i + forecast_steps - 1, -1])  # Predict target after forecast_steps
    return np.array(X), np.array(y)

def saturation_vapour_pressure(temp):
  return 0.6108 * np.exp((17.27 * temp) / (237.3 + temp))

# creating leading y values
def ylead(df, lead_period, feature):
    variable = "lead_" + str(lead_period)
    df[variable] = df[feature].shift(-lead_period)
    return df

# split data into X values and y values
X, y = splitDataIntoArray(df)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)
print(X_train.shape)

# standardise X_train
X_train = np.array(scaler.fit_transform(X_train))
#X_train = np.hstack((X_train, Day_or_Night[:X_train.shape[0]]))
# standardise X_test
X_test = np.array(scaler.transform(X_test))
#X_test = np.hstack((X_test, Day_or_Night[X_train.shape[0]:]))

# create windows for training data
# N is the number of steps the model will backtrack to predict new data
N = 24

# combine X_train and y_train
y_train = y_train.reshape(y_train.shape[0], 1)
trainingData = np.hstack((X_train, y_train))

# create training data
X_train_1, y_train_1 = create_windows(trainingData, N, 1)
X_train_6, y_train_6 = create_windows(trainingData, N, 6)
X_train_24, y_train_24 = create_windows(trainingData, N, 24)

# create windows for testing data
# combine X_test and y_test
y_test = y_test.reshape(y_test.shape[0], 1)
testingData = np.hstack((X_test, y_test))

X_test_1, y_test_1 = create_windows(testingData, N, 1)
X_test_6, y_test_6 = create_windows(testingData, N, 6)
X_test_24, y_test_24 = create_windows(testingData, N, 24)


from sklearn.base import BaseEstimator, RegressorMixin
import keras
from keras import Sequential
from keras import layers
import keras_tuner as kt
from keras import callbacks
from keras import optimizers

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

class kerasLSTM(BaseEstimator, RegressorMixin):
    def __init__(self, layer_1_units, layer_2_units, batch_size, epochs, learning_rate = 0.01, dropout = 0.2):
        self.units_1 = layer_1_units
        self.units_2 = layer_2_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.model = None
        self.input_shape = None

    def build_model(self, input_shape):
        self.input_shape = input_shape
        model = Sequential([
            layers.LSTM(self.units_1, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units_2, return_sequences=False),
            layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(loss='mse', optimizer= optimizers.RMSprop(learning_rate = self.learning_rate), metrics = ['mae'])
        return model

    def fit(self, X, y):
        self.model = self.build_model(X.shape[1:])
        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, validation_split= 0.2, callbacks = [early_stopping], verbose=1)
        return history

    def predict(self, X):
        return self.model.predict(X)

    def update_params(self, units_1, units_2 ,batch_size, epochs, learning_rate, dropout):
        self.units_1 = units_1
        self.units_2 = units_2
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout

def objective_1(trial):
    # Sample hyperparameters
    units_1 = trial.suggest_int('layer_1_units', 1, 50, step=10)
    units_2 = trial.suggest_int('layer_2_units', 1, 50, step=10)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Initialize LSTM model
    lstm_model = kerasLSTM(layer_1_units= units_1,
                           layer_2_units= units_2,
                           batch_size=batch_size,
                           epochs=epochs,
                           learning_rate=learning_rate,
                           dropout=dropout)
    
    # Fit model
    history = lstm_model.fit(X_train_1, y_train_1)
    
    # Get the final validation loss from the training history
    val_loss = history.history['val_loss'][-1]  # Last epoch's validation loss
    
    return val_loss  # Optuna minimizes this

import optuna

# Run Bayesian Optimization with Optuna
study_1 = optuna.create_study(direction='minimize')  # Minimize validation loss
study_1.optimize(objective_1, n_trials=10)

# Print best hyperparameters
print("Best LSTM parameters:", study_1.best_params)

best_params_1 = study_1.best_params

nn_1hr = kerasLSTM(31, 31, 32, 10,)
nn_1hr.update_params(best_params_1['layer_1_units'], best_params_1['layer_2_units'], best_params_1['batch_size'], best_params_1['epochs'], best_params_1['learning_rate'], best_params_1['dropout'])
nn_1hr.fit(X_train_1, y_train_1)

nn_1hr.model.save('nn_1hr.h5')

def objective_6(trial):
    # Sample hyperparameters
    units_1 = trial.suggest_int('layer_1_units', 1, 50, step=10)
    units_2 = trial.suggest_int('layer_2_units', 1, 50, step=10)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Initialize LSTM model
    lstm_model = kerasLSTM(layer_1_units= units_1,
                           layer_2_units= units_2,
                           batch_size=batch_size,
                           epochs=epochs,
                           learning_rate=learning_rate,
                           dropout=dropout)
    
    # Fit model
    history = lstm_model.fit(X_train_6, y_train_6)
    
    # Get the final validation loss from the training history
    val_loss = history.history['val_loss'][-1]  # Last epoch's validation loss
    
    return val_loss  # Optuna minimizes this

# Run Bayesian Optimization with Optuna
study_6 = optuna.create_study(direction='minimize')  # Minimize validation loss
study_6.optimize(objective_6, n_trials=10)

# Print best hyperparameters
print("Best LSTM parameters:", study_6.best_params)

best_params_6 = study_6.best_params

nn_6hr = kerasLSTM(31, 31, 32, 10,)
nn_6hr.update_params(best_params_6['layer_1_units'], best_params_6['layer_2_units'], best_params_6['batch_size'], best_params_6['epochs'], best_params_6['learning_rate'], best_params_6['dropout'])
nn_6hr.fit(X_train_6, y_train_6)

nn_6hr.model.save('nn_6hr.h5')

def objective_24(trial):
    # Sample hyperparameters
    units_1 = trial.suggest_int('layer_1_units', 1, 50, step=10)
    units_2 = trial.suggest_int('layer_2_units', 1, 50, step=10)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50, step=5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Initialize LSTM model
    lstm_model = kerasLSTM(layer_1_units= units_1,
                           layer_2_units= units_2,
                           batch_size=batch_size,
                           epochs=epochs,
                           learning_rate=learning_rate,
                           dropout=dropout)
    
    # Fit model
    history = lstm_model.fit(X_train_24, y_train_24)
    
    # Get the final validation loss from the training history
    val_loss = history.history['val_loss'][-1]  # Last epoch's validation loss
    
    return val_loss  # Optuna minimizes this

# Run Bayesian Optimization with Optuna
study_24 = optuna.create_study(direction='minimize')  # Minimize validation loss
study_24.optimize(objective_24, n_trials=10)

# Print best hyperparameters
print("Best LSTM parameters:", study_24.best_params)

best_params_24 = study_24.best_params

nn_24hr = kerasLSTM(31, 31, 32, 10,)
nn_24hr.update_params(best_params_24['layer_1_units'], best_params_24['layer_2_units'], best_params_24['batch_size'], best_params_24['epochs'], best_params_24['learning_rate'], best_params_24['dropout'])
nn_24hr.fit(X_train_24, y_train_24)

nn_24hr.model.save('nn_24hr.h5')

mse_1hr, mae_1hr = nn_1hr.model.evaluate(X_test_1, y_test_1)
mse_6hr, mae_6hr = nn_6hr.model.evaluate(X_test_6, y_test_6)
mse_24hr, mae_24hr = nn_24hr.model.evaluate(X_test_24, y_test_24)

print("prediction results:")

print("1hr:")
print("mse:",round(mse_1hr,4))
print("mae:",round(mae_1hr,4))

print("6hr:")
print("mse:",round(mse_6hr,4))
print("mae:",round(mae_6hr,4))

print("24hr:")
print("mse:",round(mse_24hr,4))
print("mae:",round(mae_24hr,4))

from keras import Model

class MyModel(Model):
     def __init__(self, nn, input_shape):
        super().__init__()
        self.model = Sequential()

        # Copy all layers except the last (Dense) layer
        for layer in nn.model.layers[:-1]:
            new_layer = layer.__class__.from_config(layer.get_config())  # Clone layer structure
            self.model.add(new_layer)  # Add to new model

        # Build the model with a dummy input to initialize weights
        dummy_input = np.zeros(input_shape)  # Shape: (batch_size=1, timesteps, features)
        _ = self.model(dummy_input)

        # Copy weights after the model has been built
        for new_layer, old_layer in zip(self.model.layers, nn.model.layers[:-1]):
            new_layer.set_weights(old_layer.get_weights())

no_dense_1hr = MyModel(nn_1hr, X_train_1.shape)
no_dense_1hr.save('no_dense_1hr')
no_dense_6hr = MyModel(nn_6hr, X_train_6.shape)
no_dense_6hr.save('no_dense_hr.h5')
no_dense_24hr = MyModel(nn_24hr, X_train_24.shape)
no_dense_24hr.save('no_dense_24hr.h5')

feature_matrix_1hr = no_dense_1hr.model.predict(X_train_1)
feature_matrix_1hr = np.hstack((feature_matrix_1hr, y_train_1.reshape(-1,1)))
print(feature_matrix_1hr.shape)
np.savetxt('feature_matrix_1hr.csv', feature_matrix_1hr, delimiter=',')

feature_matrix_6hr = no_dense_6hr.model.predict(X_train_6)
feature_matrix_6hr = np.hstack((feature_matrix_6hr, y_train_6.reshape(-1,1)))
print(feature_matrix_6hr.shape)
np.savetxt('feature_matrix_6hr.csv', feature_matrix_6hr, delimiter=',')

feature_matrix_24hr = no_dense_24hr.model.predict(X_train_24)
feature_matrix_24hr = np.hstack((feature_matrix_24hr, y_train_24.reshape(-1,1)))
print(feature_matrix_24hr.shape)
np.savetxt('feature_matrix_24hr.csv', feature_matrix_24hr, delimiter=',')

test_matrix_1hr = no_dense_1hr.model.predict(X_test_1)
test_matrix_1hr = np.hstack((test_matrix_1hr, y_test_1.reshape(-1,1)))
print(test_matrix_1hr.shape)
np.savetxt('test_matrix_1hr.csv', test_matrix_1hr, delimiter=',')

test_matrix_6hr = no_dense_6hr.model.predict(X_test_6)
test_matrix_6hr = np.hstack((test_matrix_6hr, y_test_6.reshape(-1,1)))
print(test_matrix_6hr.shape)
np.savetxt('test_matrix_6hr.csv', test_matrix_6hr, delimiter=',')

test_matrix_24hr = no_dense_24hr.model.predict(X_test_24)
test_matrix_24hr = np.hstack((test_matrix_24hr, y_test_24.reshape(-1,1)))
print(test_matrix_24hr.shape)
np.savetxt('test_matrix_24hr.csv', test_matrix_24hr, delimiter=',')