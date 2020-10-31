#! /usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def dist(x, y=None, p=1):
    ## https://en.wikipedia.org/wiki/Hyperbolic_space#:~:text=Hyperbolic%20space%20is%20a%20space,also%20called%20the%20hyperbolic%20plane.
    if y==None:
        y = np.zeros(x.shape)
        y[:,-1] = 1

    B_xy = (x[:,-1]*y[:,-1])**p - np.sum((x[:, :-1]*y[:, :-1])**p, axis=1)
    B_xy **= 1/p

    return np.arccosh(B_xy)

np.random.seed(0)


N = 1_000 # number of points to generate in the n-sphere
nn = np.arange(1,101)
pp = np.array([1/3, 1/2, 2/3, 1, 2, 3, 5, 10])
D_mm = np.empty(100)

for p in pp:

    for n in nn:

        x = np.empty((N, n+1 )) # embedded in Euclidean space

        for idx in range(N):
            # x[idx, 0:n] = np.random.normal(loc=0, scale=1, size=n)
            x[idx, 0:n] = np.random.random(size=n) - 0.5
            # both work, notheless I keep the first one, after I check the paper, I'll see.

        x[:, n] = np.sqrt(1 + np.sum(x[:, :n]**2, axis=1))

        d = dist(x, p=p)
        D_mm[n-1] = np.max(d) - np.min(d)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.scatter(nn, D_mm)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
    # plt.close()
    fig.savefig(f'./hyperbolic/hyperbola_distance_dimensions_p_{p:.2f}.png')


## 1-hyperbola
# N = 100
# n=1
# x = np.empty((N, n+1 )) # embedded in Euclidean space
#
# for idx in range(N):
#     x[idx, 0:n] = np.random.normal(loc=0, scale=1, size=n)
#     # x[idx, :] = np.random.random(size=nn[1]) - 0.5
#     # both work, notheless I keep the first one, after I check the paper, I'll see.
#
# x[:, n] = np.sqrt(1 + np.sum(x[:, :n]**2, axis=1))
#
#
# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.scatter(x[:, 0], x[:, 1])
# ax.set_aspect('equal', adjustable='box')
# # plt.show()
# # plt.close()
# fig.savefig(f'./hyperbolic/{n}_hyperbola.png')
#
# ## 2-hyperbola
# N = 1_000
# n=2
# x = np.empty((N, n+1 )) # embedded in Euclidean space
#
# for idx in range(N):
#     # x[idx, 0:n] = np.random.normal(loc=0, scale=1, size=n)
#     x[idx, 0:n] = np.random.random(size=n) - 0.5
#     # both work, notheless I keep the first one, after I check the paper, I'll see.
#
# x[:, n] = np.sqrt(1 + np.sum(x[:, :n]**2, axis=1))
#
#
# fig, tmp = plt.subplots(figsize=(10,10))
# ax = Axes3D(fig)
#
# ax.scatter(x[:, 0], x[:, 1], x[:, 2])
# # plt.show()
# # plt.close()
# fig.savefig(f'./hyperbolic/{n}_hyperbola.png')
