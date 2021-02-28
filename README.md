# jax-fenics-adjoint &middot; [![Build FEniCS](https://github.com/ivanyashchuk/jax-fenics-adjoint/workflows/FEniCS/badge.svg)](https://github.com/ivanyashchuk/jax-fenics-adjoint/actions?query=workflow%3AFEniCS+branch%3Amaster) [![Build Firedrake](https://github.com/ivanyashchuk/jax-fenics-adjoint/workflows/Firedrake/badge.svg)](https://github.com/ivanyashchuk/jax-fenics-adjoint/actions?query=workflow%3AFiredrake+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/jax-fenics-adjoint/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/jax-fenics-adjoint?branch=master)

This package enables use of [FEniCS](https://fenicsproject.org/) or [Firedrake](https://firedrakeproject.org/) for solving differentiable variational problems in [JAX](https://github.com/google/jax).

Automatic tangent linear and adjoint solvers for FEniCS/Firedrake programs are derived with [dolfin-adjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/).
These solvers make it possible to use JAX's forward and reverse Automatic Differentiation with FEniCS/Firedrake.

For using JAX-FEniCS without dolfin-adjoint (still differentiable with automatic tangent and adjoint solvers using [UFL](https://github.com/FEniCS/ufl)) check out [jax-fenics](https://github.com/IvanYashchuk/jax-fenics).

Current limitations:
* Composition of forward and reverse modes for higher-order derivatives is not implemented yet.
* Differentiation wrt Dirichlet boundary conditions and mesh coordinates is not implemented yet.

## Example
Here is the demonstration of solving the [Poisson's PDE](https://en.wikipedia.org/wiki/Poisson%27s_equation)
on 2D square domain and calculating the solution Jacobian matrix (_du/df_) using the reverse (adjoint) mode Automatic Differentiation.
```python
import jax
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

import fenics
import fenics_adjoint
import ufl

from jaxfenics_adjoint import build_jax_fem_eval
from fecr import from_numpy

# Create mesh for the unit square domain
n = 10
mesh = fenics_adjoint.UnitSquareMesh(n, n)

# Define discrete function spaces and functions
V = fenics.FunctionSpace(mesh, "CG", 1)
W = fenics.FunctionSpace(mesh, "DG", 0)

# Define FEniCS template representation of JAX input
templates = (fenics_adjoint.Function(W),)

@build_jax_fem_eval(templates)
def fenics_solve(f):
    # This function inside should be traceable by fenics_adjoint
    u = fenics_adjoint.Function(V, name="PDE Solution")
    v = fenics.TestFunction(V)
    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F = (inner(grad(u), grad(v)) - f * v) * dx
    bcs = [fenics_adjoint.DirichletBC(V, 0.0, "on_boundary")]
    fenics_adjoint.solve(F == 0, u, bcs)
    return u

# build_jax_fem_eval is a wrapper decorator that registers `fenics_solve` for JAX

# Let's create a vector of ones with size equal to the number of cells in the mesh
f = np.ones(W.dim())
u = fenics_solve(f) # u is JAX's array
u_fenics = from_numpy(u, fenics.Function(V)) # we need to explicitly provide template function for conversion

# now we can calculate vector-Jacobian product with `jax.vjp`
jvp_result = jax.vjp(fenics_solve, f)[1](np.ones_like(u))

# or the full (dense) Jacobian matrix du/df with `jax.jacrev`
dudf = jax.jacrev(fenics_solve)(f)

# function `fenics_solve` maps R^200 (dimension of W) to R^121 (dimension of V)
# therefore the Jacobian matrix dimension is dim V x dim W
assert dudf.shape == (V.dim(), W.dim())
```
Check `examples/` or `tests/` folders for the additional examples.

## Installation
First install [FEniCS](https://fenicsproject.org/download/) or [Firedrake](https://firedrakeproject.org/download.html).
Then install [pyadjoint](http://www.dolfin-adjoint.org/en/latest/) with:

    python -m pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@master

Then install [fecr](https://github.com/IvanYashchuk/fecr) with:

    python -m pip install git+https://github.com/IvanYashchuk/fecr@master

Then install [JAX](https://github.com/google/jax) with:

    python -m pip install --upgrade jax jaxlib  # CPU-only version

After that install jax-fenics-adjoint with:

    python -m pip install git+https://github.com/IvanYashchuk/jax-fenics-adjoint.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/jax-fenics-adjoint/issues/new

## Asking questions and general discussion

If you have a question or anything else, create a new [discussion]. Using issues is also fine!

[discussion]: https://github.com/IvanYashchuk/jax-fenics-adjoint/discussions/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/jax-fenics-adjoint.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/fenics  # or pytest tests/firedrake

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/jax-fenics-adjoint/pulls
