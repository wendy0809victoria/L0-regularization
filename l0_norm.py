import argparse
from utils import AverageMeter, accuracy
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser(description='PyTorch MLP Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='L0MLP', type=str,
                    help='name of experiment')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument('--beta_ema', type=float, default=0.999)
parser.add_argument('--lambas', nargs='*', type=float, default=[1., 1., 1., 1.])
parser.add_argument('--local_rep', action='store_true')
parser.add_argument('--temp', type=float, default=2./3.)
parser.add_argument('--multi_gpu', action='store_true')
parser.set_defaults(tensorboard=True)

best_prec1 = 100
writer = None
total_steps = 0
exp_flops, exp_l0 = [], []
args = parser.parse_args()
print(args.weight_decay, args.lambas, args.local_rep, args.temp)

class L0MLP(nn.Module):
    def __init__(self, input_dim, num_classes, layer_dims=[128, 64, 32, 16], N=50000, beta_ema=0.999,
                 weight_decay=1, lambas=(0.5, 0.5, 0.5, 0.5, 0.5), local_rep=False, temperature=2./3.):
        super(L0MLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.beta_ema = beta_ema
        self.weight_decay = self.N * weight_decay
        self.lambas = lambas

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate_init, lamba = 0.2 if i == 0 else 0.5, lambas[i] if len(lambas) > 1 else lambas[0]
            layers += [L0Dense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=self.weight_decay,
                               lamba=lamba, local_rep=local_rep, temperature=temperature), nn.ReLU()]

        layers.append(L0Dense(self.layer_dims[-1], num_classes, droprate_init=0.5, weight_decay=self.weight_decay, lamba=lambas[-1], local_rep=local_rep, temperature=temperature))
        self.output = nn.Sequential(*layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        return self.output(x)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
    
# Define training model 
model = L0MLP(input_dim, 3)

# Define loss functions, using MSE loss
criterion = nn.MSELoss()

# Using SGD (Stochastic Gradient Descent), with learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

l = []
e = []

# Taking 100 training epochs as an example
for epoch in range(1000):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    l0_loss = model.regularization()
    total_loss = loss + l0_loss
    total_loss.backward()
    optimizer.step()
    
    e_fl, e_l0 = model.get_exp_flops_l0() if not args.multi_gpu else \
            model.module.get_exp_flops_l0()
    exp_flops.append(e_fl)
    exp_l0.append(e_l0)
    
    if not args.multi_gpu:
        if model.beta_ema > 0.:
            model.update_ema()
    else:
        if model.module.beta_ema > 0.:
            model.module.update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    l.append(total_loss)
    e.append(epoch+1)   
    print(f'Epoch [{epoch+1}/1000], L0Loss: {total_loss.item()}')
    
# l = np.array(l)
# e = np.array(e)
# plt.plot(e, l)
