import numpy as np
import helper as h
import lstm 
import torch
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

data, d = h.loadWeatherData('weather_data.csv')

# remove outlier data
    # specifically rows that include humidity data that exceeded 1
data = h.removeOutliers(data)
#for i in range(data.shape[1]):
#   h.plotdata(data, i, d)

# separate the y values 
X, y = h.splitData(data)

# split the data into train and test sets 
    # DO NOT TOUCH THE TEST SET

X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise only the training X values 

X_train = h.standardizer(X_train)
X_test = h.standardizer(X_test)

# add bias to data 
X_train = h.addBias(X_train)
X_test = h.addBias(X_test)

# Keras LSTM
nn2 = lstm.buildKerasLstm(X_train.shape, 1, "rmsprop", "mse", "mae")
lstm_train = lstm.kerasinput(X_train, 1)
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