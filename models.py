# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout


class MnistCNN(nn.Module):

    def __init__(self):
      super(MnistCNN,self).__init__()
      self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
      self.conv2 = nn.Conv2d(10,10,kernel_size=5,stride=1)
      self.pool = nn.MaxPool2d(kernel_size=2,stride=2) #2x2 maxpool
      self.fc1 = nn.Linear(4*4*10,100)
      self.fc2 = nn.Linear(100,10)
  
    def forward(self,x):
      x = F.relu(self.conv1(x)) #24x24x10
      x = self.pool(x) #12x12x10
      x = F.relu(self.conv2(x)) #8x8x10
      x = self.pool(x) #4x4x10    
      x = x.view(-1, 4*4*10) #flattening
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

    # def __init__(self):
      
        # super(MnistCNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3)
        # self.conv2 = nn.Conv2d(32, 64, 3)
        # self.fc3 = nn.Linear(1024, 128)
        # self.fc4 = nn.Linear(128, 10)

    # def forward(self, x):
    #     h = F.relu(self.conv1(x))
    #     h = F.relu(self.conv2(h))
    #     h = F.dropout2d(F.max_pool2d(h, 6), p=0.25)
    #     h = F.dropout2d(self.fc3(h.view(h.size(0), -1)), p=0.5)
    #     h = self.fc4(h)
    #     return F.log_softmax(h,-1)

        # #modifed cnn layers
        # super(MnistCNN, self).__init__()
        # self.cnn_layers = Sequential(
        #     # Defining a 2D convolution layer
        #     Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(4),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2),

        #     # Defining another 2D convolution layer
        #     Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(4),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2),
        # )

        # self.linear_layers = Sequential(
        #     Linear(4 * 7 * 7, 10)
        # )



    # Defining the forward pass    
    # def forward(self, x):
    #     x = self.cnn_layers(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.linear_layers(x)
    #     return x


class CifarCNN(nn.Module):

    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 10)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 4)

        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.max_pool2d(h, 4)

        h = F.relu(self.fc5(h.view(h.size(0), -1)))
        h = F.relu(self.fc6(h))
        h = self.fc7(h)
        return F.log_softmax(h,-1)


class Generator(nn.Module):

    def __init__(self, in_ch):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, in_ch, 4, stride=2, padding=1)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.deconv3(h)))
        h = F.tanh(self.deconv4(h))
        return h


class Discriminator(nn.Module):

    def __init__(self, in_ch):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        if in_ch == 1:
            self.fc4 = nn.Linear(1024, 1)
        else:
            self.fc4 = nn.Linear(2304, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.sigmoid(self.fc4(h.view(h.size(0), -1)))
        return h

if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    x = torch.normal(mean=0, std=torch.ones(10, 3, 32, 32))
    model = CifarCNN()
    model(Variable(x))
