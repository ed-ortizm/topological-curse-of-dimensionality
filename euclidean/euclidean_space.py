"blalalalal"

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from topocurse.sampling import random_points_euclidean
from topocurse.metrics import lp_distance
from topocurse.figures import contrast_plot

start_time = time.perf_counter()

path_plot = "/home/edgar/Downloads/topo/euclidean"

if os.path.exists(path_plot) is False:
    os.mkdir(path_plot)

# number of points to sample
N = 1_000
# dimensionality
nn = np.arange(1, 201)
# metric's parameter
pp = [i / 10.0 for i in range(1, 11)] + [2, 3, 5, 10, 20]
# array to store distances
D_mm = np.empty(nn.size)

# Sampling points
np.random.seed(0)
X = random_points_euclidean(n=nn.size, N=N)

for p in pp:

    for n in nn:

        d = lp_distance(X=X[:, : n + 1], p=p)
        D_mm[n - 1] = np.max(d) - np.min(d)

    fig, ax = contrast_plot(
        metric="d(x,y)=L_p(x,y)",
        title="Distance behavior in the n-d Euclidean Space",
        euclidean=True,
        p=p,
    )

    ax.scatter(nn, D_mm)
    fig.savefig(f"{path_plot}/contrast_euclidean_p_{p}.png")
    fig.savefig(f"{path_plot}/contrast_euclidean_p_{p}.pdf")
    plt.close()

finish_time = time.time()

print(f"Running time: {finish_time - start_time:.2f} seconds")
