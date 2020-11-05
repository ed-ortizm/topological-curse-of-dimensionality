#! /usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from lib_metrics_spaces import hyp_dist

## Wikipedia page for Hyperbolic space

# Euclid's parallel postulate is no longer assumed to hold. Instead, the parallel
# postulate is replaced by the following alternative (in two dimensions):
#
#  *Given any line L and point P not on L, there are at least two distinct lines
#   passing through P which do not intersect L.
#
# There are several important models of hyperbolic space: the Klein model, the
# hyperboloid model, the Poincaré ball model and the Poincaré half space model.
# These all model the same geometry in the sense that any two of them can be
# related by a transformation that preserves all the geometrical properties of
# the space, including isometry (though not with respect to the metric of a
# Euclidean embedding).

## Here I implement the Hyperboloid model, where the n-hyperbolic space is
# embedded in R^{n+1} --> x_o^2 - x_1^2 - ... - x_n^2 = 1, x_o > 0

# In this model a line (or geodesic) is the curve formed by the intersection
# of H^n with a plane through the origin in R^{n+1}.

np.random.seed(0)
ploth_path = './plots/hyperbolic_spaces'

H_scale = 10_000_000
N = 1_000 # number of points to generate in the n-sphere
nn = np.arange(1,201)
# pp = np.array([1/3, 1/2, 2/3, 1, 2, 3, 5, 10])
D_mm = np.empty(nn.size)

# origin = 'minima'
origin = 'random'

for n in nn:

    Hn = H_scale*(np.random.random(size=(N, n+1)) - 0.5) # embedded in Euclidean space
    Hn[:, 0] = np.sqrt(1 + np.sum(Hn[:, 1:]**2, axis=1))

    y = H_scale*(np.random.random(size=n+1) - 0.5)
    y[0] = np.sqrt(1 + np.sum(y[1:]**2))

    d = hyp_dist(Hn, y=y, origin=origin)

    D_mm[n-1] = np.max(d) - np.min(d)

fig, ax = plt.subplots(figsize=(10,5))

ax.scatter(nn[2:], D_mm[2:])
plt.tight_layout()

fig.savefig(f'{ploth_path}/hyperbic_distance.png')
fig.savefig(f'{ploth_path}/hyperbic_distance.pdf')

plt.close()


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
