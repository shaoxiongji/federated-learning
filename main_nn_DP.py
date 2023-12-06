#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar

from opacus import PrivacyEngine
from opacus import *

from torch.utils.data import DataLoader, random_split
import torchvision


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
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
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        
        total_count = len(dataset_train)
        train_count = int(0.2 * total_count)  # 20% for training - simulates 5 clients 
        test_count = int(0.1 * total_count)   # 10% for testing
        validation = total_count - train_count - test_count # 10% for validation

        dataset_train, dataset_test, validation_dataset = random_split(dataset_train, [train_count, test_count, validation])

        img_size = dataset_train[0][0].shape

    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet':
       # net_glob = ResNetTest(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]).to(args.device)
       net_glob = torchvision.models.resnet18()
      # net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
       net_glob.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
      # net_glob.fc = torch.nn.Linear(512, 1)
       net_glob.to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    list_loss = []
    epl_list = []
    #net_glob.train()

    DELTA = 1e-5            #Should be less than 1/# data items 

    # enter PrivacyEngine
  #  privacy_engine = PrivacyEngine()
  #  model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
  #      module=net_glob,
  #      optimizer=optimizer,
  #      data_loader=train_loader,
  #      target_epsilon=1.2,
  #      target_delta=DELTA,
  #      max_grad_norm=1.0,
  #      epochs=20
  #  )

    model = net_glob
    model.train()
    data_loader = train_loader

    for epoch in range(args.epochs):
        batch_loss = []
        #for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            #output = net_glob(data)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
           # epsilon = privacy_engine.get_epsilon(DELTA)
            if batch_idx % 50 == 0:
               # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #    epoch, batch_idx * len(data), len(train_loader.dataset),
                #           100. * batch_idx / len(train_loader), loss.item()))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))
                #print(f"(ε = {epsilon:.2f}, δ = {DELTA})")
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
       # epl_list.append(epsilon)

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    
    # plot epsilon
    plt.figure()
    plt.plot(range(len(epl_list)), epl_list)
    plt.xlabel('epochs')
    plt.ylabel('epsilon')
    plt.savefig('epsilon_{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, DELTA))



    # testing
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    #test_acc, test_loss = test(net_glob, test_loader)
    test_acc, test_loss = test(model, test_loader)
