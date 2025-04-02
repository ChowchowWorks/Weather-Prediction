import helper as h
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from keras import Sequential
from keras import layers

# load data into file 
data, d = h.loadWeatherData("weather_data.csv")
# remove outliers 
data = h.removeOutliers(data)

# add bias
data = h.addBias(data)
# split X y 
X , y = h.splitData(data)

# split train-test
X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise 
X_train = h.standardizer(X_train)

X_train1, y_train1 = h.create_windows(np.hstack((X_train, np.reshape(y_train, (y_train.shape[0], 1)))), 24, 1)

class kerasLSTM(BaseEstimator, RegressorMixin):
    def __init__(self, units=50, batch_size=32, epochs=10):
        self.units = units
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def build_model(self, input_shape):
        model = Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            layers.LSTM(self.units, return_sequences=False),
            layers.Dense(1)
        ])
        model.compile(loss='mse', optimizer='rmsprop')
        return model
    
    def fit(self, X, y):
        self.model = self.build_model(X.shape[1:])
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

param_grid = {
    'units': [1, 5, 10],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

model = kerasLSTM()
grid = GridSearchCV(estimator= model, param_grid= param_grid, scoring= 'neg_mean_absolute_error', cv= 5)
grid_result = grid.fit(X_train1, y_train1)

print("Best parameters:", grid_result.best_params_)