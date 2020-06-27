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
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = self.bn1(t)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = self.bn2(t)
        
        t = F.relu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = self.bn3(t)
        
        t = F.relu(self.fc1(t.reshape(-1, 40*3*3)))
        t = self.bn4(t)
        
        t = F.relu(self.fc2(t))
        t = self.bn5(t)
        
        t = self.out(t)
        
        return t



class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs


class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader):
        
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment = f' -{run}')
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images.to(getattr(run,'device', 'cpu')))
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss/len(self.loader.dataset)
        accuracy = self.epoch_num_correct/len(self.loader.dataset)
        
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accracy', accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
        
        clear_output(wait = True)
        display(df)
        
    def track_loss(self, loss):
        self.epoch_loss += loss.item()*self.loader.batch_size
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
        
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim = 1).eq(labels).sum().item()
    
    def save(self, filename):
        
        pd.DataFrame.from_dict(
            self.run_data,
            orient = 'columns'
        ).to_csv(f'{filename}.csv')
        
        with open(f'{filename}.json','w', encoding = 'utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii = False, indent = 4)



## loading the dataset

train_set = torchvision.datasets.FashionMNIST(
    root = './datasets',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)


## dataset with normalization
loader = DataLoader(train_set, batch_size = len(train_set), num_workers = 1)
data = next(iter(loader))
mean = data[0].mean()
std = data[0].std()

train_set_normalized = torchvision.datasets.FashionMNIST(
    root = './datasets',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
)




## training the network
params = OrderedDict(
    lr = [0.011],
    batch_size = [1000],
    device = ['cuda']
)

num_epochs = 100
m = RunManager()
for run in RunBuilder.get_runs(params):

    device = torch.device(run.device)
    network = Network().to(device)
    loader = DataLoader(train_set_normalized, batch_size = run.batch_size, num_workers = 1)
    optimizer = optim.Adam(network.parameters(), lr = run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(num_epochs):
        m.begin_epoch()
        
        for batch in loader:        ## get batch
            
            images = batch[0].to(device)
            labels = batch[1].to(device)
            preds = network(images)  ## pass batch
            loss = F.cross_entropy(preds, labels)   ## calculate loss

            optimizer.zero_grad()
            loss.backward()  ## calculate gradients
            optimizer.step()  #update weights
            
            m.track_loss(loss)
            m.track_num_correct(preds, labels)
        
        m.end_epoch()
    
    m.end_run()

m.save('results')


## saving the network
PATH = 'networks/network.pt'
torch.save(network.state_dict(), PATH)