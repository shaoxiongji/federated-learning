#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResNetTest
from models.Fed import FedAvg
from models.test import test_img
import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import time as t
from opacus.validators import ModuleValidator
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        #self.data = data_tensor[:, :-1]
        self.data = data_tensor[:, :-1].reshape(-1,1, 9, 100) #[batch_size, channels, height, width]
        self.targets = data_tensor[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def IMU_noniid(dataset, num_users,labels):
    """
    Sample non-I.I.D client data from IMU dataset 
    Altered from Mnist_noniid definition
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 90, 10
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 30, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = [int(x) for x in dict_users[i]]
    return dict_users

#Define function to adjust Privacy Budget
def adjustPB(PBList, accList):
    print(PBList)
    newAcc = []
    newPB = PBList.copy()
    newAcc = accList.copy()
    newAcc.sort()
    i = len(newAcc)
    indexList = []
    for item in newAcc: 
        index = accList.index(item)
        if (index in indexList):
            accList[index] = 0
        index = accList.index(item)
        indexList.append(index)
        print(index)
        newPB[index] = newPB[index] - (0.1*i)
        #Set some bounds
        if (newPB[index] > 2):
            newPB[index] = 2
        if (newPB[index] < 0.7):
            newPB[index] = 0.7
        i -= 1
    print(newPB)
    return newPB
        


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'ASL1':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

        #Load images from the directory and apply the transformation
        dataset = datasets.ImageFolder('data/DataSet_ASL/archive', transform=transform)

        #Save the dataset
        torch.save(dataset, 'train_data.pt')
        #load dataset. Comment out the other lines after you run it once to save time
        dataset = torch.load('train_data.pt')

        #percentages are arbitrary
        total_count = len(dataset)
        train_count = int(0.1 * total_count)  # 10% for training
        test_count = int(0.1 * total_count)   # 10% for testing
        validation = total_count - train_count - test_count # 80% for validation

        dataset_train, dataset_test, validation_dataset = random_split(dataset, [train_count, test_count, validation])

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'ASL':
        dataset = torch.load('train_data.pt')
        total_count = len(dataset)
        train_count = int(0.30*total_count)
        test_count = int(0.056*total_count)
        validation = total_count - train_count - test_count
        dataset_train, dataset_test, dataset_validation = random_split(dataset,[train_count, test_count,validation])
        test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
        validation_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False)
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            
    elif args.dataset == 'motion':
        dataset = torch.load('IMU_data.pt')
        dataset = dataset.float()
        dataset = CustomDataset(dataset)

        #percentages are arbitrary
        total_count = len(dataset)
        print("Total Count: ", total_count)
        train_count = int(0.8 * total_count)  # 80% for training
        test_count = int(0.1 * total_count)   # 10% for testing
        validation = total_count - train_count - test_count # 10% for validation

        dataset_train, dataset_test, validation_dataset = random_split(dataset, [train_count, test_count, validation])

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'IMU':
        dataset = torch.load('IMU_data.pt')
        #dataset = dataset.type(torch.LongTensor)
        dataset = torch.tensor(dataset, dtype=torch.long, device=args.device)
        #dataset = dataset.float()
        #dataset = dataset.long()
        dataset = CustomDataset(dataset)

        total_count = len(dataset)
        print("Total count: ", total_count)
        train_count = 900  # ~65% for training
        test_count = total_count - train_count

        dataset_train, dataset_test = random_split(dataset, [train_count, test_count])
        train_indices = dataset_train.indices
        all_labels = dataset.targets#.numpy()
        train_labels = all_labels[train_indices]
        dict_users = mnist_iid(dataset_train, args.num_users)
        #dict_users = IMU_noniid(dataset_train, args.num_users,train_labels)
        #img_size = dataset_train[0][0].shape

    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'ASL':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet':
       # net_glob = ResNetTest(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]).to(args.device)
       net_glob = torchvision.models.resnet18()
      # net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
       net_glob.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
      # net_glob.fc = torch.nn.Linear(512, 1)
       net_glob.to(args.device)
    elif args.model == 'mnasnet':
        net_glob = torchvision.models.mnasnet1_3()
           # net_glob.to(args.device)
    elif args.model == 'mlp':
        len_in = 1
      #  for x in img_size:
      #      len_in *= x
      #  net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    #Ensures that model works with Privacy Engine 
#    net_glob = ModuleValidator.fix(net_glob)
    print(net_glob)


    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()


    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    file1 = open("output_FL_Resnet_MNIST.txt", "w") 

    epsList = [ ]

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    # Uncomment for adaptive dp
    idxs_users = []
    privacyIndx = []
    i = 0
    for i in range(args.num_users):
        idxs_users.append(i)
        privacyIndx.append(1.2 + (0.1 * i))
    print(privacyIndx)
    
        
    for iter in range(args.epochs):
        print("Epoch: ", iter)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        #Comment out when using adaptive dp
       # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #print("Idx List: " , idxs_users)
        if (iter != 0):
            privacyIndx = adjustPB(privacyIndx, accuracyList)
            print(privacyIndx)

        accuracyList = []
        for idx in idxs_users:
            print(" User: " , idx)
            #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], privacy=1.2)           #use for regular DP
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], privacy=privacyIndx[idx])       #use for adpative DP
            w, loss, eps= local.train(net=copy.deepcopy(net_glob).to(args.device))

            acc_train, l = test_img(net_glob, dataset_train, args)
            print('Accuracy: ', acc_train, "\n")
            accuracyList.append(acc_train)

            print("Epsilon: ", eps)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                epsList.append(eps)
            else:
                w_locals.append(copy.deepcopy(w))
                epsList.append(eps)
            loss_locals.append(copy.deepcopy(loss))

        print("Epsilon List: ", epsList)

        # update global weights
        if args.global_aggr == 'FedAvg':
            w_glob = FedAvg(w_locals)
            print('this actually runs')
        else:
            print('something wrong')
            
            
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
    
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        
        #net_glob.eval()
        acc_train, l = test_img(net_glob,dataset_test, args)
        print('Accuracy: ', acc_train, "\n")

        file1.write(str(acc_train))
         
    file1.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('fed_{}_{}_{}_C{}_Non_iid{}_DP_5_clients.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

