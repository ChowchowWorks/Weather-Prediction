import numpy as np
import helper as h
import LSTM 

import torch
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

data, d = h.loadWeatherData('weather_data.csv')
#for i in range(data.shape[1]):
#   h.plotdata(data, i, d)

# remove outlier data
    # specifically rows that include humidity data that exceeded 1
data = h.removeOutliers(data)

# separate the y values 
X, y = h.splitData(data)

# split the data into train and test sets 
    # DO NOT TOUCH THE TEST SET

X_train, X_test, y_train, y_test = h.trainTest(X, y)

# standardise only the training X values 

X_train = h.standardizer(X_train)

# add bias to data 
X_train = h.addBias(X_train)

# LSTM Model
X_train = torch.tensor(X_train, dtype= torch.float32)
y_train = torch.tensor(y_train, dtype= torch.float32)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset)

lstm = LSTM.LSTM(X_train.shape, 1)
trainer = L.Trainer(max_epochs=100)
trainer.fit(lstm, train_dataloaders= dataloader)


import torch
import torch.nn.functional as F  # For MSE calculation

# Ensure model is in evaluation mode
lstm.eval()

# Convert test data to tensor
X_test = torch.tensor(h.addBias(X_test), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)  # Actual values

# Add batch dimension if necessary
if len(X_test.shape) == 2:
    X_test = X_test.unsqueeze(0)

# Disable gradient computation for inference
with torch.no_grad():
    y_pred = lstm(X_test)

# Compute Mean Squared Error (MSE)
mse = F.mse_loss(y_pred, y_test)

print("Mean Squared Error (MSE):", mse.item())  # Convert tensor to scalar
