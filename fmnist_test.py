import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from collections import namedtuple
from itertools import product

from IPython.display import display, clear_output
import pandas as pd
import math
import time
import json

torch.set_printoptions(linewidth = 150)
torch.set_grad_enabled(True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 5, padding = (1,1))
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 4, padding = (1,1))
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = 3, padding = (1,1))
        self.bn3 = nn.BatchNorm2d(40)
        
        self.fc1 = nn.Linear(in_features = 40*3*3, out_features = 180)
        self.bn4 = nn.BatchNorm1d(180)
        self.fc2 = nn.Linear(in_features = 180, out_features = 90)
        self.bn5 = nn.BatchNorm1d(90)
        self.out = nn.Linear(in_features = 90, out_features = 10)
        
    def forward(self, t):
        t = F.relu(self.bn1(self.conv1(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = F.relu(self.bn2(self.conv2(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = F.relu(self.bn3(self.conv3(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = F.relu(self.bn4(self.fc1(t.reshape(-1, 40*3*3))))
        t = F.relu(self.bn5(self.fc2(t)))
        
        t = self.out(t)
        
        return t

## loading the test_set

test_set = torchvision.datasets.FashionMNIST(
    root = './datasets',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)


## testset normalization
loader_test = DataLoader(test_set, batch_size = len(test_set), num_workers = 1)
data_test = next(iter(loader_test))
mean_test = data_test[0].mean()
std_test = data_test[0].std()

test_set_normalized = torchvision.datasets.FashionMNIST(
    root = './datasets',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_test, std_test)
    ])
)

## loading the network

PATH = 'networks/network.pt'
net = Network().cuda()
net.load_state_dict(torch.load(PATH))
net.eval()


## testing
loader_test = DataLoader(test_set_normalized, batch_size = 1000, num_workers = 1)

device_test = torch.device('cuda')
total_loss = 0
total_correct = 0

for batch_test in loader_test:
    
    images = batch_test[0].to(device_test)
    labels = batch_test[1].to(device_test)
    preds_test = net(images)  ## pass batch
    loss_test = F.cross_entropy(preds_test, labels) 
    
    total_loss += loss_test.item()*loader.batch_size
    total_correct += preds_test.argmax(dim = 1).eq(labels).sum().item()
    
print(total_loss)
print('accuracy:', total_correct/len(test_set))
