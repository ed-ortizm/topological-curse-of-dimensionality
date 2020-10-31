#! /usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from lib_metrics_spaces import lp, plot

ti = time.time()

N = 1_000 # number of points to sample
nn = np.arange(1,201) # dimensionality
pp = [i/10. for i in range(1,11)]+[2, 3, 5, 10, 20] # metric's parameter
DD = np.empty((N, nn.size)) # array to store distances
print(pp)
## Sampling points

np.random.seed(0)

X = np.random.rand(N, nn.size)

for p in pp:

    for n in nn:

        DD[:, n-1] = lp(X=X[:, :n+1], p=p) # the origin [0,..., 0]

    Dmin = np.min(DD, axis=0)
    Dmax = np.max(DD, axis=0)
    D_diff = Dmax-Dmin

    plot(x=nn, y=D_diff, fname=f'contrast_p_{p}')

tf = time.time()

print(f'Running time: {tf-ti:.2f} seconds')
