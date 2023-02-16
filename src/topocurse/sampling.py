"""
Module with functions to sample random points from different manifolds.
1. Spheres
2. Hyperbolic space
3. Euclidean space
"""

import numpy as np

np.random.seed(0)


def random_points_spheres(n: int, N: int) -> np.array:
    """
    Sample N points from the n-sphere

    INPUT:
    n: dimension of the sphere
    N: number of points to sample

    OUTPUT:
    S: Random sample of points in the n-sphere. Shape (N, n)
    """

    X = np.random.normal(loc=0, scale=1, size=(N, n))

    R = np.sqrt(np.sum(X * X, axis=1))
    S = X * (1 / R[:, np.newaxis])

    return S
