#! /usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import numpy as np

################################################################################
def lp_metric(x, y=None, p=2):

    if y==None:
        y = np.zeros(x.shape)

    return np.sum((x-y)**p, axis=1)**(1/p)

################################################################################

ti = time.time()

N = 1_000 # number of points to sample

pp = [2/10, 2/5, 2/4, 2/3, 1, 2, 3, 10, 20, 50, 100, 500, 1_000] # metric's parameter

dd = np.arange(1,201) # dimensionality


## Sampling points

np.random.seed(0)

DD = np.empty((N, dd.size)) # array to store distances

for p in pp:

    for idd, d in enumerate(dd):
        x = np.random.rand(N, d)
        DD[:, idd] = lp_metric(x=x, p=p) # the origin [0,..., 0]  

    Dmin = np.min(DD, axis=0)
    Dmax = np.max(DD, axis=0)
    D_diff = Dmax-Dmin

tf = time.time()

print(f'Running time: {tf-ti:.2f} seconds')
