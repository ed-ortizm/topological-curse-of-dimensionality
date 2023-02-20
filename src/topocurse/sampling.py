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


def random_points_hyperbolic(n: int, N: int, H_scale: float = 1.0) -> np.array:
    """
    Sample N points from the n-hyoperbolic space.
    Here I implement the Hyperboloid model, where the n-hyperbolic space
    is embedded in:
    R^{n+1} --> x_o^2 - x_1^2 - ... - x_n^2 = 1, x_o > 0

    INPUT:
    n: dimension of the hyperbolic space
    N: number of points to sample
    H_scale: scale of the hyperbolic space

    OUTPUT:
    H_n: Random sample of points in the n-hyperbola embbeded
    in an n+1 Euclidena space. Shape (N, n+1)
    """

    # embedded in (n+1)-Euclidean space
    H_n = H_scale * (
        np.random.random(size=(N, n + 1)) - 0.5
    )

    H_n[:, 0] = np.sqrt(1 + np.sum(H_n[:, 1:] ** 2, axis=1))

    return H_n
