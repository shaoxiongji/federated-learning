#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def average_weights_loss(w, loss):
    loss_inverse = [1.0/abs(l) for l in loss]
    w_loss = [l/sum(loss_inverse) for l in loss_inverse]
    w_avg = w[0]
    for k in w_avg.keys():
        w_avg[k] = w_avg[k]*w_loss[0]
    for k in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]*w_loss[i]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def update_params(net_from, net_to):
    """
    copy parameters from self.net -> self.net_pi
    :return:
    """
    for m_from, m_to in zip(net_from.modules(), net_to.modules()):
        if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()
