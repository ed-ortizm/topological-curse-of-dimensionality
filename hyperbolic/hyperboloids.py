"""
Hyperbolic spaces
"""

import os
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from topocurse.sampling import random_points_hyperbolic
from topocurse.metrics import hyperbolic_distance
from topocurse.figures import contrast_plot

start_time = time.perf_counter()

np.random.seed(0)
# creat path directory if not exists
path_plot = "/home/edgar/Downloads/topo/hyperboloids"

if os.path.exists(path_plot) is False:
    os.makedirs(path_plot)

H_scale = 1
N = 5_00  # number of points to generate in H^n
nn = np.arange(1, 201)
n_1_hyper = None  # data for 2D and 3D plot
n_2_hyper = None  # data for 2D and 3D plot
# pp = np.array([1/3, 1/2, 2/3, 1, 2, 3, 5, 10])
D_mm = np.empty(nn.size)

# origin = 'minima'
origin_at_minima = False

for n in nn:

    # Hn = H_scale * (
    #     np.random.random(size=(N, n + 1)) - 0.5
    # )  # embedded in Euclidean space
    # Hn[:, 0] = np.sqrt(1 + np.sum(Hn[:, 1:] ** 2, axis=1))

    Hn = random_points_hyperbolic(n=n, N=N, H_scale=H_scale)

    # data for 2D and 3D plot
    if n == 1:
        n_1_hyper = Hn.copy()

    if n == 2:
        n_2_hyper = Hn.copy()

    # Query point
    if origin_at_minima is False:

        Y = random_points_hyperbolic(n=n, N=1, H_scale=H_scale)

    else:

        Y = None

    d = hyperbolic_distance(
        X=Hn,
        Y=Y,
        origin_at_minima=origin_at_minima,
    )

    D_mm[n - 1] = np.max(d) - np.min(d)


# plot distance behavior in hyperbolic spaces
fig, ax = contrast_plot(
    metric="d(x,y)=arccosh($(Q(x+y)-2)/2 \\to Q(x) = x_0^2 - \cdots -x_n^2$",
    title="Distance behavior in hyperbolic spaces",
)

ax.scatter(nn, D_mm)
fig.savefig(f"{path_plot}/contrast_hyperbolic_space.png")
fig.savefig(f"{path_plot}/contrast_hyperbolic_space.pdf")
plt.close()

# plot 1-hyperbolic space
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("$H^1$", fontsize="xx-large")
ax.scatter(n_1_hyper[:, 1], n_1_hyper[:, 0])
fig.savefig(f"{path_plot}/n_2_hyperbolic_space.png")
fig.savefig(f"{path_plot}/n_2_hyperbolic_space.pdf")
plt.close()

# 2-hyperbola
fig, tmp = plt.subplots(figsize=(10, 10))
ax = Axes3D(fig=fig, auto_add_to_figure=False)
fig.add_axes(ax)

ax.set_title("$H^2$", fontsize="xx-large")

ax.scatter(n_2_hyper[:, 2], n_2_hyper[:, 1], n_2_hyper[:, 0])
fig.savefig(f"{path_plot}/n_3_hyperbolic_space.png")
fig.savefig(f"{path_plot}/n_3_hyperbolic_space.pdf")
plt.close()

finish_time = time.perf_counter()
print(f"Running time: {finish_time - start_time:.2f} [s]")
