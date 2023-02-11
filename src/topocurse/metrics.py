"""
Metrics to measure distance in spaces with possitive, null and negative
curvature. In plain English, measure the distance between two points in
a sphere, euclidean space and a hyperbolic manifold.
"""

import numpy as np


def lp_distance(
    X: np.array,
    Y: np.array = None,
    p: float = 2.
) -> np.array:

    """
    Lp metric to compute distance in  the Euclidean space

    INPUT:
    X: Random sample of points in The Euclidean space. Shape (N, d)
    Y: Origing of Euclidean space. if None, set to zeros. Shape (N, d)
    p: parameter of the metric. p=2 is the Euclidean metric
        d is the number of dimensions of the space

    OUTPUT:
    distance: distance between X and Y. Shape (N, )
    """

    if Y is None:

        Y = np.zeros(X.shape)

    distance = np.sum((X-Y)**p, axis=1)**(1/p)

    return distance


def arccosine_distance(
    X: np.array,
    Y: np.array = None,
    origin_at_angles_0: bool = True
) -> np.array:

    """
    Arccosine metric to compute distance in the sphere

    INPUT:
    X: Random sample of points in The sphere. Shape (N, d)
    y: reference point in the sphere. Shape (d, )
    origin_at_angles_0: reference point in the sphere. if True, set Y to
        zeros with first entry set to 1. Shape (d, )

    OUTPUT:
    distance: distance between X and y. Shape (N, )
    """

    if origin_at_angles_0 is True:

        # For this case, I set the origin in:
        Y = np.zeros(X.shape[1])
        Y[0] = 1.

        distance = np.arccos(X[:, 0])

        return distance

    distance = np.arccos(np.sum(X*Y, axis=1))

    return distance


def hyperbolic_distance(
    X: np.array,
    Y: np.array = None,
    origin_at_minima: bool = True
) -> np.array:

    """
    Hyperbolic metric to compute distance in the hyperbolic space.
    ref: https://en.wikipedia.org/wiki/Hyperbolic_space

    INPUT:
    X: Random sample of points in The hyperbolic space. Shape (N, d)
    y: reference point in the hyperbolic space. Shape (d, )
    origin_at_minima: reference point in the hyperbolic space. if True
        set Y to zeros with first entry set to 1. Shape (d, )

    OUTPUT:

    distance: distance between X and y. Shape (N, )
    """

    def Q(X):
        return X[:, 0]**2 - np.sum(X[:, 1:]**2, axis=1)

    if origin_at_minima is True:

        Y = np.zeros(X.shape[1])
        Y[0] = 1.
        B_XY = 0.5*(Q(X+Y) - Q(X) - Q(Y))

        return np.arccosh(B_XY)

    B_XY = 0.5*(Q(X+Y)-Q(X)-Q(Y))

    distance = np.arccosh(B_XY)

    return distance
