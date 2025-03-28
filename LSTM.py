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
        self.add(layers.LSTM(units = 5)) #tune units

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

# Keras LSTM
nn2 = buildKerasLstm(X_train.shape, 1, "rmsprop", "mse", "mae")
lstm_train = kerasinput(X_train, 1)
history = nn2.fit(lstm_train, y_train,epochs=200, batch_size=32, shuffle=False, validation_split=0.2) # tune epochs

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Epochs')
plt.legend()
plt.grid()
plt.show()

lstm_test = lstm.kerasinput(X_test, 1)
mse = nn2.evaluate(lstm_test, y_test)
print(mse)

pred = nn2.predict(lstm_test)

x = np.arange(len(y_test))
plt.plot(x, y_test, label = "actual", color = "blue")
plt.plot(x, pred, label = "predicted", color = "orange", alpha = 0.8)
plt.title("Actual vs Predicted")
plt.xlabel("time")
plt.ylabel("relative humidity")
plt.show()