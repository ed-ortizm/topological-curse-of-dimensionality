import matplotlib.pyplot as plt
import numpy as np


def lp(X, Y=None, p=2):

    if Y == None:
        Y = np.zeros(X.shape)

    return np.sum((X - Y) ** p, axis=1) ** (1 / p)


def arccos_metric(X, y, origin="1...0"):

    if origin == "1...0":
        # For this case, I set the origin in:
        # y = np.zeros(X.shape[1])
        # y[0] = 1, therefore

        return np.arccos(X[:, 0])

    return np.arccos(np.sum(X * y, axis=1))


def Q(X):
    return X[:, 0] ** 2 - np.sum(X[:, 1:] ** 2, axis=1)


def hyp_dist(X, y, origin="minima"):
    ## https://en.wikipedia.org/wiki/Hyperbolic_space#:~:text=Hyperbolic%20space
    ## %20is%20a%20space,also%20called%20the%20hyperbolic%20plane.

    if origin == "minima":
        # y = np.zeros(X.shape[1])
        # y[0] = 1. # reference point
        X[:, 0] += 1  # --> cuase B_xy = (Q(x+y)-Q(x)-Q(y))/2
        B_Xy = 0.5 * (Q(X) - 2)

        return np.arccosh(B_Xy)

    B_Xy = 0.5 * (Q(X + y) - 2)

    return np.arccosh(B_Xy)


def plot(
    x, y, fname, path, title, metric, euclidean=False, p=None, figsize=(10, 5)
):

    if euclidean:

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
