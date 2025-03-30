import helper as h
import numpy as np
import matplotlib.pyplot as plt
import keras as k 
from keras import Sequential
from keras import layers

#### Keras LSTM Model #####    

class kerasLSTM(Sequential):
    def __init__(self, data_shape, hidden_size):
        super().__init__()
        self.add(layers.LSTM(units = 1)) #tune units

        self.add(layers.Dense(1))

def buildKerasLstm(data_shape, hidden_size, optimizer, loss, metrics):
    if type(optimizer)!= str:
        optimizer = str(optimizer)
    if type(loss)!= str:
        loss = str(optimizer)
    if type(metrics)!= str:
        metrics = str(metrics)

    model = kerasLSTM(data_shape, hidden_size)
    model.compile(optimizer= optimizer,loss= loss, metrics=[metrics])

    return model

def kerasinput(data, time_steps):
    return np.reshape(data, (data.shape[0], time_steps, data.shape[1]))

# load data into file 
data, d = h.loadWeatherData("weather_data.csv")
# remove outliers 
data = h.removeOutliers(data)
# split X y 
X , y = h.splitData(data)

# split train-test
X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise 
X_train = h.standardizer(X_train)

forecast = [1, 6, 24]
results = {}

for i in forecast:
    results[i] = {}
    # create windows 
    n_steps = 24
    forecast_steps = i
    X_train1, y_train1 = h.create_windows(np.hstack((X_train, np.reshape(y_train, (y_train.shape[0], 1)))), n_steps, forecast_steps)

    # Keras LSTM
    nn = buildKerasLstm(X_train1.shape,1, 'rmsprop', 'mse', 'mae')
    history = nn.fit(X_train1, y_train1, 32, 10, validation_split=0.2)
    plt.plot(range(len(history.history['loss'])), history.history['loss'])
    plt.show()
    plt.close()

    # evaluate test data
    X_test1, y_test1 = h.create_windows(np.hstack((h.standardizer(X_test), np.reshape(y_test, (y_test.shape[0], 1)))), n_steps, forecast_steps)
    mse, mae = nn.evaluate(X_test1, y_test1, verbose= 1)

    results[i]['mse'] = mse
    results[i]['mae'] = mae

print("Prediction Results:")
for key in results:
    print(f"MSE ({key}) = {results[key]['mse']}")
    print(f"MAE ({key}) = {results[key]['mae']}")