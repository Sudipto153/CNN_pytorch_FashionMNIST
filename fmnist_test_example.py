import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF

from model import Network



def main():

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


if __name__ == "__main__":
    main()