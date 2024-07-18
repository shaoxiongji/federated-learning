#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
