# jax-fenics-adjoint &middot; [![Build](https://github.com/ivanyashchuk/jax-fenics-adjoint/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/jax-fenics-adjoint/actions?query=workflow%3ACI+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/jax-fenics-adjoint/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/jax-fenics-adjoint?branch=master)

This package enables use of [FEniCS](http://fenicsproject.org) for solving differentiable variational problems in [JAX](https://github.com/google/jax).

Automatic tangent linear and adjoint solvers for FEniCS programs are derived with [dolfin-adjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/).
These solvers make it possible to use JAX's forward and reverse Automatic Differentiation with FEniCS.

For using JAX-FEniCS without dolfin-adjoint (still differentiable with automatic tangent and adjoint solvers using [UFL](https://github.com/FEniCS/ufl)) check out [jax-fenics](https://github.com/IvanYashchuk/jax-fenics).

For JAX-Firedrake interface see [jax-firedrake](https://github.com/IvanYashchuk/jax-firedrake).

## Installation
First install [FEniCS](http://fenicsproject.org).
Then install [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) with:

    python -m pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@master

Then install [JAX](https://github.com/google/jax) with:

    python -m pip install pip install --upgrade jax jaxlib  # CPU-only version

After that install the jax-fenics-adjoint with:

    python -m pip install git+https://github.com/IvanYashchuk/jax-fenics-adjoint.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/jax-fenics-adjoint/issues/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/jax-fenics-adjoint.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/jax-fenics-adjoint/pulls
