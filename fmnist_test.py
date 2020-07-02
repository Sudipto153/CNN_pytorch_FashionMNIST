import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from model import Network

from collections import OrderedDict
from collections import namedtuple
from itertools import product

torch.set_printoptions(linewidth = 150)
torch.set_grad_enabled(True)


def main():

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


if __name__ == "__main__":
    main()