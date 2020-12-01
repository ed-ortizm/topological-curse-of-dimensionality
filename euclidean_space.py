#! /usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from lib_metrics_spaces import lp, plot

ti = time.time()

path_plot = './plots/euclidean_spaces'

N = 1_000 # number of points to sample
nn = np.arange(1,201) # dimensionality
pp = [i/10. for i in range(1,11)]+[2, 3, 5, 10, 20] # metric's parameter
D_mm = np.empty(nn.size) # array to store distances

## Sampling points

np.random.seed(0)
X = np.random.rand(N, nn.size)

for p in pp:

    for n in nn:

        d = lp(X=X[:, :n+1], p=p)
        D_mm[n-1] = np.max(d) - np.min(d)

    plot(x=nn, y=D_mm, fname=f'contrast_euclidean_p_{p}', path=path_plot,
    title='Distance behavior in the n-d Euclidean Space', metric='d(x,y)=L_p(x,y)', euclidean=True, p=p)

tf = time.time()

print(f'Running time: {tf-ti:.2f} seconds')
