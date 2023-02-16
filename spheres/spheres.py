"""
Reproduce results in spherical manifold from the paper:
Aggarwal, C. C., &amp; Yu, P. S.(2001, May).
Outlier detection for high dimensional data.
In Proceedings of the 2001 ACM SIGMOD international conference
on Management of data (pp. 37-46).
"""

import os
import time

import matplotlib.pyplot as plt
# import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from topocurse.metrics import arccosine_distance


def plot(
    x, y, fname, path, title, metric, euclidean=False, p=None, figsize=(10, 5)
):

    # creat path directory if not exists

    if os.path.exists(path) is False:
        os.makedirs(path)

    if euclidean is True:

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(
            f"{title}\n {metric} $ \\to p=$ {p:.1f}", fontsize="xx-large"
        )
        ax.set_xlabel("Number of Diemnsions", fontsize="xx-large")
        ax.set_ylabel("Contrast", fontsize="xx-large")

        ax.scatter(x, y)
        plt.tight_layout()
        fig.savefig(f"{path}/{fname}.png")
        fig.savefig(f"{path}/{fname}.pdf")
        plt.close()
    else:

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title}\n {metric}", fontsize="xx-large")
        ax.set_xlabel("Number of Diemnsions", fontsize="xx-large")
        ax.set_ylabel("Contrast", fontsize="xx-large")

        ax.scatter(x, y)
        plt.tight_layout()
        fig.savefig(f"{path}/{fname}.png")
        fig.savefig(f"{path}/{fname}.pdf")
        plt.close()


start_time = time.perf_counter()

path_plot = "/home/edgar/Downloads/topo/spheres"

np.random.seed(0)

N = 1_000  # number of points
nn = np.arange(1, 201)  # number of dimensions
D_mm = np.empty(nn.size)

## uniform sampling of points in the unit n-phere
# Marsaglia
X = np.random.normal(loc=0, scale=1, size=(N, nn.size))
# X = 1.*(np.random.random(size=(N, nn.size)) - 0.5)
# both work, notheless I keep the first one, after I check the paper, I'll see.

origin = "random"

for n in nn:

    # uniform distributed points in the n-sphere
    R = np.sqrt(np.sum(X[:, :n] * X[:, :n], axis=1))
    S = X[:, :n] * (1 / R[:, np.newaxis])

    if origin == "random":
        y = np.random.normal(loc=0, scale=1, size=n)
        # y = 1.*(np.random.random(size=nn.size) - 0.5)
        r = np.sqrt(np.sum(y * y))
        s = y * (1 / r)
    else:
        y = None
        origin = False

    d = arccosine_distance(X=S, Y=s, origin_at_angles_0=origin)
    D_mm[n - 1] = np.max(d) - np.min(d)
# print(D_mm)
plot(
    x=nn,
    y=D_mm,
    fname="contrast_sphere",
    path=path_plot,
    title="Distance behavior in the n-spheres",
    metric="d(x,y)=arccos(x $\cdot$ y)",
)

## 2-sphere
# uniform distributed points in the n-sphere
n = 2
r = np.sqrt(np.sum(X[:, :n] * X[:, :n], axis=1))
S = X[:, :n] * (1 / r[:, np.newaxis])

plot(
    x=S[:, 0],
    y=S[:, 1],
    fname=f"n_{n}_sphere",
    path=path_plot,
    title="S-1",
    metric="",
    figsize=(10, 10),
)

# ## 3-sphere
# # uniform distributed points in the n-sphere
n = 3
R = np.sqrt(np.sum(X[:, :n] * X[:, :n], axis=1))
S = X[:, :n] * (1 / R[:, np.newaxis])
fig, tmp = plt.subplots(figsize=(10, 10))
ax = Axes3D(fig)
ax.set_title("S-2", fontsize="xx-large")

ax.scatter(S[:, 0], S[:, 1], S[:, 2])
# plt.show()
fig.savefig(f"{path_plot}/S_2.png")
fig.savefig(f"{path_plot}/S_2.pdf")
plt.close()

finish_time = time.perf_counter()

print(f"Running time: {finish_time-start_time:.2f} [s]")
