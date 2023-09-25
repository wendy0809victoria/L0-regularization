import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from l0_layers import L0Conv2d, L0Dense
from base_layers import MAPConv2d, MAPDense
from utils import get_flat_fts
from copy import deepcopy
import torch.nn.functional as F

# Loading dataset
X = np.loadtxt('X_train.txt')
y = np.loadtxt('Y_train.txt')
input_dim = X.shape[1]
inputs = torch.tensor(X)
labels = torch.tensor(y)
torch.set_default_dtype(torch.float32)
inputs = inputs.to(torch.float32)
labels = labels.float()

# MLP model for L1 regularization
class L1MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], output_dim=3, l1_penalty=0.001):
        super(L1MLP, self).__init__()
        self.input_dim = input_dim  # 1 input layer
        self.hidden_dims = hidden_dims  # 4 hidden layers
        self.output_dim = output_dim  # 1 output layer
        self.l1_penalty = l1_penalty  # L1 norm tuning parameter

        # Define hidden_layers
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Define output_layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # Forward propogation
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        x = self.output_layer(x)
        return x
    
    def l1_loss(self):
        # Define L1 regularization loss
        l1_reg = 0.0
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)  # Calculate L1 norm for each parameter
        return self.l1_penalty * l1_reg  # Multiply by the L1 penalty

# Define training model 
model = L1MLP(input_dim)

# Define loss functions, using MSE loss
criterion = nn.MSELoss()

# Using SGD (Stochastic Gradient Descent), with learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Taking 100 training epochs as an example
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    l1_loss = model.l1_loss()
    total_loss = loss + l1_loss
    total_loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item()}')
     