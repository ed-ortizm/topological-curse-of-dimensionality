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

def plot(x, y, fname, path, figsize=(10,5)):

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y)
    plt.tight_layout()
    fig.savefig(f'{path}/{fname}.png')
    fig.savefig(f'{path}/{fname}.pdf')
    plt.close()
################################################################################
