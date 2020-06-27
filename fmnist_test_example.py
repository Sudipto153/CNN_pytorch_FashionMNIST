import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF


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


## loading the network
PATH = 'networks/network.pt'
net = Network().cuda()
net.load_state_dict(torch.load(PATH))
net.eval()

## getting the classes
test_set = torchvision.datasets.FashionMNIST(
    root = './datasets',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)
classes = test_set.classes


imsize = 28
loader_ex = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])

image = Image.open('images/image1.jpg')   ## image directory goes here __
x = TF.to_grayscale(image)
x = loader_ex(x)
idx = net(x.unsqueeze(0).cuda()).argmax(dim = 1)[0]
print(classes[idx])
image