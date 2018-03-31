#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.norm1d = nn.BatchNorm1d(num_features=dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.norm1d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


# class CNN(nn.Module):
#     def __init__(self, args):
#         super(CNN, self).__init__()
#         self.num_classes = args.num_classes
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.fc = nn.Linear(32, self.num_classes)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = torch.mean(x, -1)
#         x = torch.mean(x, -1)
#         x = self.fc(x)
#         return self.softmax(x)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)