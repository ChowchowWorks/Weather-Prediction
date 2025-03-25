import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L 
from torch.utils.data import TensorDataset, dataloader

class LSTMC(L.LightningModule):

    def __init__(self):
        # create and initialise Weight and Bias tensors
        super().__init__()
        
        # assign weight through normal distribution 
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # initialise weights 
        self.stmw = nn.Parameter(torch.normal(mean = mean, std = std), requires_grad = True) #stmw stands for short-term memory weight


    def lstm_unit(self, input, long, short):
        # do lstm math

    def forward(self, input):
        # make forward pass through unrolled lstm

    def configure_optimizers(self):
        # configure Adam Optimizer
        return super().configure_optimizers()
    
    def training_step(self, batch, batch_idx):
        # calculate loss and log training progress