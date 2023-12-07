#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import xlsxwriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import timm
from functools import partial
from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar, CNNASL


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
    print('Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    torch.cuda.empty_cache
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
        img_size = dataset_train[0][0].shape
 
    elif args.dataset == 'asl':
        #Load dataset
        dataset = torch.load('train_data.pt')
        total_count = len(dataset)
        train_count = int(0.3*total_count)
        test_count = int(0.3*total_count)
        validation = total_count - train_count - test_count
        dataset_train, dataset_test, dataset_validation = random_split(dataset,[train_count, test_count,validation])


    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'asl':
        net_glob = CNNASL(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'mnist':
        net_glob = torchvision.models.resnet18()
        net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net_glob.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        net_glob.to(args.device)
    elif args.model == 'resnet' and args.dataset == 'asl':
        net_glob = torchvision.models.resnet18()
        net_glob.to(args.device)
    elif args.model == 'effic' and args.dataset == 'mnist':
        net_glob = torchvision.models.efficientnet_b0()
        net_glob.features[0] = torchvision.ops.Conv2dNormActivation(1, 32, kernel_size=(3, 3), stride=(2, 2), norm_layer=torch.nn.BatchNorm2d, activation_layer=torch.nn.SiLU)
        net_glob.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),torch.nn.Linear(in_features=1280, out_features=10))
        net_glob.to(args.device)
    elif args.model == 'effic' and args.dataset == 'asl':
        net_glob = torchvision.models.efficientnet_b0()
        net_glob.to(args.device)
    elif args.model == 'effic_b3' and args.dataset == 'mnist':
        net_glob = torchvision.models.efficientnet_b3()
        net_glob.features[0] = torchvision.ops.Conv2dNormActivation(1, 40, kernel_size=(3, 3), stride=(2, 2), norm_layer=torch.nn.BatchNorm2d, activation_layer=torch.nn.SiLU)
        net_glob.to(args.device)
    elif args.model == 'mnasnet' and args.dataset == 'mnist':
        net_glob = torchvision.models.mnasnet1_3()
        net_glob.layers[0] = torch.nn.Conv2d(1, 40, 3, padding=1, stride=2, bias=False) #40 for mnasnet1_3 and 16 for 0_5
        net_glob.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10,bias=True)
        net_glob.to(args.device)
    elif args.model == 'mnasnet1_3' and args.dataset == 'asl':
        net_glob = torchvision.models.mnasnet1_3()
        net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)


    # load testing set
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
    elif args.dataset == 'asl':
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    list_loss = []
    list_accuracy = []
    list_test_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        net_glob.train()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)

        print('Train loss:', loss_avg,'\n')
        list_loss.append(loss_avg)

        print('\nTest on', len(dataset_test), 'samples')
        test_acc, test_loss = test(net_glob, test_loader)
        list_accuracy.append(test_acc)
        list_test_loss.append(test_loss)



    #workbook = xlsxwriter.Workbook('{}_{}_LossOver{}Epochs.xlsx'.format(args.model,args.dataset,args.epochs))
    #worksheet = workbook.add_worksheet()
    #row = 0
    #col = 0
    #for los in (list_loss):
    #    worksheet.write(row,col,los)
    #    row+=1
    #workbook.close()

    workbook = xlsxwriter.Workbook('{}_{}_TestLossOver{}Epochs.xlsx'.format(args.model,args.dataset,args.epochs))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for los in (list_test_loss):
        worksheet.write(row,col,los)
        row+=1
    workbook.close()

    workbook = xlsxwriter.Workbook('{}_{}_TestAccuracyOver{}Epochs.xlsx'.format(args.model,args.dataset,args.epochs))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for acc in (list_accuracy):
        worksheet.write(row,col,acc)
        row+=1
    workbook.close()



    print('\nValidation on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
