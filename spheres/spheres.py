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
from topocurse.sampling import random_points_spheres
from topocurse.figures import contrast_plot


# def plot(
#     x, y, fname, path, title, metric, euclidean=False, p=None, figsize=(10, 5)
# ):

#     # creat path directory if not exists

#     if os.path.exists(path) is False:
#         os.makedirs(path)

#     if euclidean is True:

#         fig, ax = plt.subplots(figsize=figsize)
#         ax.set_title(
#             f"{title}\n {metric} $ \\to p=$ {p:.1f}", fontsize="xx-large"
#         )
#         ax.set_xlabel("Number of Diemnsions", fontsize="xx-large")
#         ax.set_ylabel("Contrast", fontsize="xx-large")

#         ax.scatter(x, y)
#         plt.tight_layout()
#         fig.savefig(f"{path}/{fname}.png")
#         fig.savefig(f"{path}/{fname}.pdf")
#         plt.close()
#     else:

#         fig, ax = plt.subplots(figsize=figsize)
#         ax.set_title(f"{title}\n {metric}", fontsize="xx-large")
#         ax.set_xlabel("Number of Diemnsions", fontsize="xx-large")
#         ax.set_ylabel("Contrast", fontsize="xx-large")

#         ax.scatter(x, y)
#         plt.tight_layout()
#         fig.savefig(f"{path}/{fname}.png")
#         fig.savefig(f"{path}/{fname}.pdf")
#         plt.close()


np.random.seed(0)

start_time = time.perf_counter()

path_plot = "/home/edgar/Downloads/topo/spheres"


N = 1_000  # number of points
nn = np.arange(1, 201)  # number of dimensions
D_mm = np.empty(nn.size)

origin = "random"

for n in nn:

    # uniform distributed points in the n-sphere
    # R = np.sqrt(np.sum(X[:, :n] * X[:, :n], axis=1))
    # S = X[:, :n] * (1 / R[:, np.newaxis])

    S = random_points_spheres(n=n, N=N)

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

# creat path directory if not exists

if os.path.exists(path_plot) is False:
    os.makedirs(path_plot)

fig, ax = contrast_plot(
    metric="d(x,y)=arccos(x $\cdot$ y)",
    title="Distance behavior in the n-spheres",
)

ax.scatter(nn, D_mm)
fig.savefig(f"{path_plot}/contrasts_sphere.png")
fig.savefig(f"{path_plot}/contrasts_sphere.pdf")
plt.close()
# plot(
#     x=nn,
#     y=D_mm,
#     fname="contrast_sphere",
#     path=path_plot,
#     title="Distance behavior in the n-spheres",
#     metric="d(x,y)=arccos(x $\cdot$ y)",
# )

## 2-sphere
# uniform distributed points in the n-sphere

S = random_points_spheres(n=2, N=N)

fig, ax = contrast_plot(
    metric=" ",
    title="S-1",
)

fname = f"n_{n}_sphere"
ax.scatter(S[:, 0], S[:, 1])
fig.savefig(f"{path_plot}/{fname}.png")
fig.savefig(f"{path_plot}/{fname}.pdf")
plt.close()
# 
# plot(
#     x=S[:, 0],
#     y=S[:, 1],
#     fname=f"n_{n}_sphere",
#     path=path_plot,
#     title="S-1",
#     metric="",
#     figsize=(10, 10),
# )
# 
# n=3

S = random_points_spheres(n=3, N=N)

fig, tmp = plt.subplots(figsize=(10, 10))
ax = Axes3D(fig=fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title("S-2", fontsize="xx-large")

ax.scatter(S[:, 0], S[:, 1], S[:, 2])
# plt.show()
fig.savefig(f"{path_plot}/S_2.png")
fig.savefig(f"{path_plot}/S_2.pdf")
plt.close()

finish_time = time.perf_counter()

print(f"Running time: {finish_time-start_time:.2f} [s]")
