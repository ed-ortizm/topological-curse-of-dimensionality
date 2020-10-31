#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

################################################################################
def dist(x, y=None):

    if y==None:
        # For this case, I set the origin in:
        # y = np.zeros(shape=x.shape)
        # y[:, -1] = 1, therefore
        d = np.arccos(x[:, -1])
        return d

    d = np.arccos(np.sum(x*y, axis=1))
    return d
################################################################################
np.random.seed(0)

nn = np.arange(1,101)
N = 1_000 # number of points to generate in the n-sphere

D_mm = np.empty(nn.size)

for n in nn:
    # uniform distributed points in the n-sphere
    # Marsaglia
    # x = np.random.normal(loc=0, scale=1, size=(N, n))
    x = 2*(np.random.random(size=(N, n)) - 0.5)
    # both work, notheless I keep the first one, after I check the paper, I'll see.

    r = np.sqrt(np.sum(x*x, axis=1))
    x *= (1/r[:, np.newaxis])

    d = dist(x)
    D_mm[n-1] = np.max(d) - np.min(d)

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(nn, D_mm)
# ax.set_aspect('equal', adjustable='box')
# plt.show()
# plt.close()
# ax.set_aspect('equal', adjustable='box')
fig.savefig(f'./sphere/sphere_distance_dimensions.png')
print(D_mm[0:5], '\n', nn[:5])

## 2-sphere

# x = np.empty((N, nn[1]))
#
# for idx in range(N):
#     x[idx, :] = np.random.normal(loc=0, scale=1, size=nn[1])
#     # x[idx, :] = np.random.random(size=nn[1]) - 0.5
#     # both work, notheless I keep the first one, after I check the paper, I'll see.
#
# r = np.sqrt(np.sum(x*x, axis=1))
#
# s = x*(1/r[:, np.newaxis])

# fig, ax = plt.subplots(figsize=(10,10))
#
# ax.scatter(s[:, 0], s[:, 1])
# ax.set_aspect('equal', adjustable='box')
# plt.show()
# plt.close()


## 3-sphere
# from mpl_toolkits.mplot3d import Axes3D
# x = np.empty((N, nn[2]))
#
# for idx in range(N):
#     # x[idx, :] = np.random.normal(loc=0, scale=1, size=nn[2])
#     x[idx, :] = np.random.random(size=nn[2]) - 0.5
#     # both work, notheless I keep the first one, after I check the paper, I'll see.
#
# r = np.sqrt(np.sum(x*x, axis=1))
#
# s = x*(1/r[:, np.newaxis])
#
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#
#
# ax.scatter(s[:,0], s[:,1], s[:,2])
# plt.show()
# plt.close()
