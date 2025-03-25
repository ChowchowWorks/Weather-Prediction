import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam

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
        
        
        

    