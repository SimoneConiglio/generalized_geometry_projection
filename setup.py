from setuptools import setup, find_packages

setup(
    name="ggp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "gemseo",
    ],
    author="Simone Coniglio",
    description="Generalized Geometry Projection for Topology Optimization in Additive Manufacturing",
)
