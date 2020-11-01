#! /usr/bin/env python3

import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from lib_metrics_spaces import arccos_metric, plot

ti = time.time()

path_plot = './plots/spheres'

np.random.seed(0)

N = 1_000 # number of points
nn = np.arange(1,201) # number of dimensions
D_mm = np.empty(nn.size)

## uniform sampling of points in the n-phere
# Marsaglia
X = np.random.normal(loc=0, scale=1, size=(N, nn.size))
# X = 1.*(np.random.random(size=(N, nn.size)) - 0.5)
# both work, notheless I keep the first one, after I check the paper, I'll see.

for n in nn:

    # uniform distributed points in the n-sphere
    r = np.sqrt(np.sum(X[:, :n]*X[:, :n], axis=1))
    S = X[:, :n]*(1/r[:, np.newaxis])

    d = arccos_metric(S)
    D_mm[n-1] = np.max(d) - np.min(d)
# print(D_mm)
plot(x=nn, y=D_mm, fname=f'contrast_sphere', path=path_plot)

## 2-sphere
# uniform distributed points in the n-sphere
n = 2
r = np.sqrt(np.sum(X[:, :n]*X[:, :n], axis=1))
S = X[:, :n]*(1/r[:, np.newaxis])

plot(x=S[:, 0], y=S[:, 1], fname=f'n_{n}_sphere', path=path_plot, figsize=(10,10))

# ## 3-sphere
# # uniform distributed points in the n-sphere
# n = 3
# r = np.sqrt(np.sum(X[:, :n]*X[:, :n], axis=1))
# S = X[:, :n]*(1/r[:, np.newaxis])
#
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ax.scatter(S[:,0], S[:,1], S[:,2])
# plt.show()
# plt.close()

tf = time.time()
print(f'Running time: {tf-ti:.2f} seconds')
