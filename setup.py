from distutils.core import setup

import sys

if sys.version_info[0] < 3:
    raise Exception(
        "The JAX-FEniCS-Adjoint only supports Python3. Did you run $python setup.py <option>.? Try running $python3 setup.py <option>."
    )

setup(
    name="jaxfenics_adjoint",
    description="JAX-FEniCS interface using dolfin-adjoint",
    url="https://github.com/IvanYashchuk/jax-fenics",
    author="Ivan Yashchuk",
    license="MIT",
    packages=["jaxfenics_adjoint"],
    install_requires=["jax", "fenics", "pyadjoint", "fdm", "scipy"],
)
