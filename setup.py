"""Setup script for topocurse package."""


from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="topocurse",
    version="0.1.0",
    author="Edgar Ortiz",
    author_email="ed.ortizm@gmail.com",
    packages=find_packages(where="src", include=["[a-z]*"], exclude=[]),
    package_dir={"": "src"},
    description=(
        "Python Code to explore the curse of dimensionality in"
        "spaces different to the Euclidean space"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ed-ortizm/topological-curse-of-dimensionality",
    license="MIT",
    keywords=(
        "AI, no-free-lunch-theorem, curse-of-dimensionality,"
        "euclidean-space, hyperbolic-space, sphere, manifold,"
        "topology, geometry",
    ),
)
