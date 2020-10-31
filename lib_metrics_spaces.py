import matplotlib.pyplot as plt
import numpy as np

################################################################################
def lp(X, Y=None, p=2):

    if Y==None:
        Y = np.zeros(X.shape)

    return np.sum((X-Y)**p, axis=1)**(1/p)
################################################################################
def plot(x, y, fname, figsize=(10,5)):

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y)
    plt.tight_layout()
    fig.savefig(f'{fname}.png')
    fig.savefig(f'{fname}.png')
    plt.close()

################################################################################
