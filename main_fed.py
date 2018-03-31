#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, autograd
from sklearn import metrics
from tensorboardX import SummaryWriter

from options import args_parser
from Update import LocalUpdate
from FedNets import MLP, CNN
from averaging import average_weights


def test(net_g, data_loader, args):
    # testing
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        log_probs = net_g(data)
        test_loss += F.nll_loss(log_probs, target, size_average=False).data[0] # sum up batch loss
        y_pred = log_probs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()

    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape[-1]
    # sample users
    num_users = args.num_users
    num_items = int(len(dataset_train)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset_train))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # build model
    if args.model == 'cnn':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNN(args=args).cuda()
        else:
            net_glob = CNN(args=args)
    elif args.model == 'mlp':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = MLP(dim_in=img_size*img_size, dim_hidden=64, dim_out=args.num_classes).cuda()
        else:
            net_glob = MLP(dim_in=img_size*img_size, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    val_acc_list, net_list = [], []
    for iter in tqdm(range(args.epochs)):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
            w, loss = net_local.update_weights(net=net_glob)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = average_weights(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if args.epochs % 10 == 0:
            print('\nTrain loss:', loss_avg)
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac))

    # testing
    list_acc, list_loss = [], []
    for c in tqdm(range(num_users)):
        net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[c], tb=summary)
        acc, loss = net_local.test(net=net_glob)
        list_acc.append(acc)
        list_loss.append(loss)
    print("average acc:", sum(list_acc)/len(list_acc))

