import matplotlib.pyplot as plt
import numpy as np

################################################################################
def lp(X, Y=None, p=2):

    if Y==None:
        Y = np.zeros(X.shape)

    return np.sum((X-Y)**p, axis=1)**(1/p)
################################################################################
def arccos_metric(X, Y=None):

    if Y==None:
        # For this case, I set the origin in:
        # Y = np.zeros(shape=X.shape)
        # Y[:, -1] = 1, therefore

        return np.arccos(X[:, -1])

    return np.arccos(np.sum(X*Y, axis=1))
################################################################################
def Q(X):
    return X[:, 0]**2 - np.sum(X[:, 1:]**2, axis=1)

def hyp_dist(X, Y, origin='default'):
    ## https://en.wikipedia.org/wiki/Hyperbolic_space#:~:text=Hyperbolic%20space
    ## %20is%20a%20space,also%20called%20the%20hyperbolic%20plane.

    if origin=='default':
        # Y = np.zeros(X.shape)
        # Y[:,0] = 1. # reference point
        X[:, 0] += 1
        B_xy = 0.5*( Q(X) - 2)

        return np.arccosh(B_xy)

    B_xy = 0.5*( Q(X+Y) -2 )

    return np.arccosh(B_xy)
################################################################################

def plot(x, y, fname, path, figsize=(10,5)):

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y)
    plt.tight_layout()
    fig.savefig(f'{path}/{fname}.png')
    fig.savefig(f'{path}/{fname}.pdf')
    plt.close()
################################################################################
