"""Setup script for topocurse package."""

from setuptools import setup, find_packages

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="topocurse",
    version="1.0.0",
    author="Edgar Ortiz",
    author_email="ed.ortizm@gmail.com",
    packages=find_packages(where="src", include=["[a-z]*"], exclude=[]),
    package_dir={"": "src"},
    description="Explore the curse of dimensionality in different manifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ed-ortizm/topological-curse-of-dimensionality",
    license="MIT",
    keywords="curse of dimensionality, euclidean space, hyperbolic space, sphere, manifold,topology",
)
