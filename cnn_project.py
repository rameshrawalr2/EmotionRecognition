#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:27:30 2018

@author: ramesh
"""
#import numpy as np
#import PIL as Image
from  utils import getImageData

class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

def main():
    X, Y = getImageData()

    # reshape X for tf: N x w x h x c
    X = X.transpose((0, 2, 3, 1))
    print("X.shape:", X.shape)

    model = CNN(
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )
    model.fit(X, Y)
