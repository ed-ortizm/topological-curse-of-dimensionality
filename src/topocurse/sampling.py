"""
Module with functions to sample random points from different manifolds.
1. Spheres
2. Hyperbolic space
3. Euclidean space

Wikipedia page for Hyperbolic space

Euclid's parallel postulate is no longer assumed to hold. Instead, the
parallel postulate is replaced by the following alternative
(in two dimensions):

*Given any line L and point P not on L, there are at least two
distinct lines passing through P which do not intersect L.

There are several important models of hyperbolic space: the Klein
model, the hyperboloid model, the Poincaré ball model and the Poincaré
 half space model.
These all model the same geometry in the sense that any two of them
can be related by a transformation that preserves all the geometrical
properties of the space, including isometry
(though not with respect to the metric of a Euclidean embedding).

Here I implement the Hyperboloid model, where the n-hyperbolic space is
embedded in R^{n+1} --> x_0^2 - x_1^2 - ... - x_n^2 = 1, x_0 > 0

In this model a line (or geodesic) is the curve formed by the
intersection of H^n with a plane through the origin in R^{n+1}.
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
    R^{n+1} --> x_0^2 - x_1^2 - ... - x_n^2 = 1, x_0 > 0

    INPUT:
    n: dimension of the hyperbolic space
    N: number of points to sample
    H_scale: scale of the hyperbolic space

    OUTPUT:
    H_n: Random sample of points in the n-hyperbola embbeded
    in an n+1 Euclidena space. Shape (N, n+1)
    """

    # embedded in (n+1)-Euclidean space
    H_n = H_scale * (np.random.random(size=(N, n + 1)) - 0.5)

    H_n[:, 0] = np.sqrt(1 + np.sum(H_n[:, 1:] ** 2, axis=1))

    return H_n


def random_points_euclidean(n: int, N: int) -> np.array:
    """
    Sample N points from the n-Euclidean space.

    INPUT:
    n: dimension of the Euclidean space
    N: number of points to sample

    OUTPUT:
    E_n: Random sample of points in the n-Euclidean space. Shape (N, n)
    """

    E_n = np.random.random(size=(N, n))

    return E_n
