import torch
import torch.utils
from torchvision import datasets, transforms

def load_data(batch_size=64):
    # 对图片进行预处理，后面的datasets.MNIST可直接套用此函数
    # .Compose()函数是一系列操作的list，保证([])中的操作顺序执行
    # .ToTensor()函数是将图片从(height, weight, channels)形式转变为(channels, height, weight)形式的tensor张量，并且将像素值从[0,255]变为[0,1]
    # .Normalize()函数是将像素值从[0,1]转变为[-1,1]，因为均值为0效果比较好，其中(0.5,)是因为只有一个channel
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

    # 加载数据集
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader