import helper as h
import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
import keras as k 
import tensorflow as tf
from keras import Sequential
from keras import layers


import lightning as L 
from torch.utils.data import TensorDataset, dataloader

class LSTM(L.LightningModule):
    
    def __init__(self, data_shape, hidden_size):
        super().__init__()
        input_size = data_shape[1]
        self.lstm = nn.LSTM(input_size= input_size, hidden_size= hidden_size)
    
    def forward(self, input):
        lstm_out , temp = self.lstm(input)
        prediction = lstm_out[:,-1]
        return prediction
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.01)
    
    def training_step(self, batch, batch_idx):
        X_train, y_train = batch
        pred = self.forward(X_train)  # Get predictions from the model
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, y_train)  # Compute loss
        return loss
    
def trainModel(model, dataloader, epochs):
    
    trainer = L.Trainer(max_epochs=epochs, accelerator="auto")
    trainer.fit(model, train_dataloaders= dataloader)

    return model
        
def runModel(model, X_test):
    # ensure LSTM in evaluation setting
    model.eval()
    # conver test data to tensor
    X_test = torch.tensor(h.addBias(X_test), dtype=torch.float32)

    # Add batch dimensions if necessary
    if len(X_test.shape) == 2:
        X_test = X_test.unsqueeze(0)
    
    # Disable gradient computation for inference
    with torch.no_grad():
        y_pred = model(X_test)

    return y_pred

def computeMSE(y_pred, y_test):
    mse = F.mse_loss(y_pred, y_test)
    print("Mean Squared Error (MSE):", mse.item())
    return mse

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