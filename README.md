# Metrics' behavior in high dimensional spaces

This repository is aimed at exploring the behavior of metrics in different spaces as is the case of spherical, hyperbolic and Euclidean spaces. 
After uniformly sampling a given number of points in the space under consideration, the difference between the distance of the farthest away 
point minus the closest one to a given reference will be inspected against the number of dimensions of the space. One concrete goal is to understand 
the curse of dimensionality in manifolds with different geometrical properties and try to design 'metrics' that are more suitable to explore distances
in these spaces.

We consider the following spaces:

* The n-dimensional Euclidean space
* The *n-sphere*.
* The *n* dimensional *hyperbolic space*, constructed using the hyperboloid model, embedded in an n+1 dimensional space.
<!-- 
This repository builds upon the work in  **Aggarwal, C. C., &amp; Yu, P. S.** (2001, May). **Outlier detection for high dimensional data.** In Proceedings of the 2001 ACM SIGMOD international conference on Management of data (pp. 37-46). -->
``` citation
Aggarwal CC, Hinneburg A, Keim DA. On the surprising behavior of distance metrics in high dimensional space. InDatabase Theory—ICDT 2001: 8th International Conference London, UK, January 4–6, 2001 Proceedings 8 2001 (pp. 420-434). Springer Berlin Heidelberg.
```
# Install

```bash
python3 setup.py develop --user
```