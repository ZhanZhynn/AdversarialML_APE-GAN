# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

from models import MnistCNN, CifarCNN
from utils import accuracy, fgsm, noise_attack, si_ni_fgsm

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# from IPython.display import Image, display



def load_dataset(args):
    if args.data == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.expanduser("~/.torch/data/mnist"), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.expanduser("~/.torch/data/mnist"), train=False, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=128, shuffle=False)
    elif args.data == "cifar":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.expanduser("~/.torch/data/cifar10"), train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.expanduser("~/.torch/data/cifar10"), train=False, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=128, shuffle=False)
    return train_loader, test_loader


def load_cnn(args):
    if args.data == "mnist":
        return MnistCNN
    elif args.data == "cifar":
        return CifarCNN


def main(args):
    check_path = args.checkpoint
    os.makedirs(check_path, exist_ok=True)
    attacktype = args.attack

    print(check_path)

    print("Generating Model ...")
    print("-" * 30)


    train_loader, test_loader = load_dataset(args)
    CNN = load_cnn(args)
    model = CNN().cuda()
    cudnn.benchmark = True

    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    loss_func = nn.CrossEntropyLoss().cuda()

    train_loss_lst = []
    train_acc_lst = []
    test_loss_lst = []
    test_acc_lst = []

    epochs = args.epochs
    print_str = "\t".join(["{}"] + ["{:.6f}"] * 4)
    print("\t".join(["{:}"] * 5).format("Epoch", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."))
    for e in range(epochs):
        train_loss, train_acc, train_n = 0, 0, 0
        test_loss, test_acc, test_n = 0, 0, 0

        model.train()
        for x, t in tqdm(train_loader, total=len(train_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * t.size(0)
            train_acc += accuracy(y, t)
            train_n += t.size(0)

        model.eval()
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)

            test_loss += loss.item() * t.size(0)
            test_acc += accuracy(y, t)
            test_n += t.size(0)
        scheduler.step()
        print(print_str.format(e, train_loss / train_n, test_loss / test_n,
                               train_acc / train_n * 100, test_acc / test_n * 100))

        train_loss_lst.append(train_loss / train_n)
        train_acc_lst.append(train_acc / train_n * 100)
        test_loss_lst.append(test_loss / test_n)
        test_acc_lst.append(test_acc / test_n * 100)

    #plot the error and accuracy graph
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax[0].plot(train_loss_lst, 'r',label="Train")
    ax[0].plot(test_loss_lst, 'g', label="Test")
    ax[0].set_title("Error Graph")
    ax[0].set_ylabel("Error (cross-entropy)")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(train_acc_lst, 'r',label="Train")
    ax[1].plot(test_acc_lst, 'g', label="Test")
    ax[1].set_title("Accuracy Graph")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    fig.show()
    # plt.show()
    # plt.savefig(os.path.join(check_path, "plot.png"))
    fig.savefig(os.path.join(check_path, "fig.png"))

    # img_name = "fig"
    # image_path = '/content/APE-GAN/checkpoint/cifar/' + str(img_name) + '.png'
    # # display(Image('/content/APE-GAN/checkpoint/mnist/result_30.png'))
    # display(Image(image_path))

    
    # Generate Adversarial Examples
    print("-" * 30)
    print("Genrating Adversarial Examples ...")
    eps = args.eps
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data = None, None
    for x, t in tqdm(train_loader, total=len(train_loader), leave=False):
        x, t = Variable(x.cuda()), Variable(t.cuda())
        y = model(x)
        train_acc += accuracy(y, t)

        if attacktype == "fgsm": #FGSM attack
          x_adv = fgsm(model, x, t, loss_func, eps)
        elif attacktype == "noise_attack": #noise attack
          x_adv = noise_attack(model, x, t, loss_func, eps)
        else:
          x_adv = si_ni_fgsm(model, x, t, loss_func, eps)


        y_adv = model(x_adv)
        adv_acc += accuracy(y_adv, t)
        train_n += t.size(0)

        x, x_adv = x.data, x_adv.data
        if normal_data is None:
            normal_data, adv_data = x, x_adv
        else:
            normal_data = torch.cat((normal_data, x))
            adv_data = torch.cat((adv_data, x_adv))

    print("Accuracy(normal) {:.6f}, Accuracy(FGSM) {:.6f}".format(train_acc / train_n * 100, adv_acc / train_n * 100))
    torch.save({"normal": normal_data, "adv": adv_data}, "data.tar")
    torch.save({"state_dict": model.state_dict()}, "cnn.tar")

    return train_loss_lst, train_acc_lst,test_loss_lst, test_acc_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--milestones", type=list, default=[50, 75])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/test")
    parser.add_argument("--attack", type=str, default="fgsm") #either fgsm, noise_attack or si_ni_fgsm

    args = parser.parse_args()
    main(args)
