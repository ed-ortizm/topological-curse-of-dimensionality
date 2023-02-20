import os
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from lib_metrics_spaces import hyp_dist, plot
from topocurse.sampling import random_points_hyperbolic
from topocurse.metrics import hyperbolic_distance
# Wikipedia page for Hyperbolic space

# Euclid's parallel postulate is no longer assumed to hold. Instead, the 
# parallel postulate is replaced by the following alternative
# (in two dimensions):
#
# *Given any line L and point P not on L, there are at least two 
# distinct lines passing through P which do not intersect L.
#
# There are several important models of hyperbolic space: the Klein 
# model, the hyperboloid model, the Poincaré ball model and the Poincaré
#  half space model.
# These all model the same geometry in the sense that any two of them 
# can be related by a transformation that preserves all the geometrical 
# properties of the space, including isometry
# (though not with respect to the metric of a Euclidean embedding).

# Here I implement the Hyperboloid model, where the n-hyperbolic space is
# embedded in R^{n+1} --> x_o^2 - x_1^2 - ... - x_n^2 = 1, x_o > 0

# In this model a line (or geodesic) is the curve formed by the 
# intersection of H^n with a plane through the origin in R^{n+1}.

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

plot(
    x=nn,
    y=D_mm,
    fname="contrast_hyperbolic_space",
    path=path_plot,
    title="Distance behavior in hyperbolic spaces",
    metric="d(x,y)=arccosh($(Q(x+y)-2)/2 \\to Q(x) = x_0^2 - \cdots -x_n^2$",
)

# fig, ax = plt.subplots(figsize=(10,5))
# ax.scatter(nn[2:], D_mm[2:])
# plt.tight_layout()
#
# fig.savefig(f'{ploth_path}/hyperbic_distance.png')
# fig.savefig(f'{ploth_path}/hyperbic_distance.pdf')
#
# plt.close()

# plot 1-hyperbolic space
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("$H^1$", fontsize="xx-large")
ax.scatter(n_1_hyper[:, 1], n_1_hyper[:, 0])
# plt.show()
fig.savefig(f"{path_plot}/n_2_hyperbolic_space.png")
fig.savefig(f"{path_plot}/n_2_hyperbolic_space.pdf")
plt.close()

# 2-hyperbola
fig, tmp = plt.subplots(figsize=(10, 10))
ax = Axes3D(fig=fig, auto_add_to_figure=False)
fig.add_axes(ax)

ax.set_title("$H^2$", fontsize="xx-large")

ax.scatter(n_2_hyper[:, 2], n_2_hyper[:, 1], n_2_hyper[:, 0])
# plt.show()
fig.savefig(f"{path_plot}/n_3_hyperbolic_space.png")
fig.savefig(f"{path_plot}/n_3_hyperbolic_space.pdf")
plt.close()

finish_time = time.perf_counter()
print(f"Running time: {finish_time - start_time:.2f} [s]")
