#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import autograd
import torch.optim as optim
from torchvision import datasets, transforms

from options import args_parser
from FedNets import MLP, CNNMnist, CNNCifar


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data, volatile=True), autograd.Variable(target)
        log_probs = net_g(data)
        test_loss += F.nll_loss(log_probs, target, size_average=False).data[0]
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNCifar(args=args).cuda()
        else:
            net_glob = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNMnist(args=args).cuda()
        else:
            net_glob = CNNMnist(args=args)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).cuda()
        else:
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    list_loss = []
    net_glob.train()
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            data, target = autograd.Variable(data), autograd.Variable(target)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
            batch_loss.append(loss.data[0])
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
