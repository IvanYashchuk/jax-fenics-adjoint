from setuptools import setup

import sys

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

setup(
    name="jaxfenics_adjoint",
    version="1.0.0",
    description="JAX-FEniCS interface using dolfin-adjoint",
    url="https://github.com/IvanYashchuk/jax-fenics",
    author="Ivan Yashchuk",
    license="MIT",
    packages=["jaxfenics_adjoint"],
    install_requires=["jax", "fdm", "fecr"],
)
