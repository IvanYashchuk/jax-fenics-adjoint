from pytest_check import check
import pytest
import fdm
import jax
from jax.config import config
import jax.numpy as np

import firedrake
import firedrake_adjoint
import ufl

from jaxfenics_adjoint import build_jax_fem_eval_fwd

config.update("jax_enable_x64", True)

mesh = firedrake.UnitSquareMesh(3, 2)
V = firedrake.FunctionSpace(mesh, "P", 1)


def solve_firedrake(q, kappa0, kappa1):

    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    u = firedrake.Function(V)
    bcs = [firedrake.DirichletBC(V, kappa1, "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - q * kappa1 * f * u * dx
    v = firedrake.TestFunction(V)
    F = firedrake.derivative(JJ, u, v)
    firedrake.solve(F == 0, u, bcs=bcs)
    return u


templates = (firedrake.Function(V), firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1), np.ones(1) * 1.2)
jax_fem_eval = build_jax_fem_eval_fwd(templates)(solve_firedrake)

# multivariate output function
ff = lambda x, y, z: np.sin(np.cos(jax_fem_eval(x, np.sqrt(y**3), z)))  # noqa: E731
ff0 = lambda x: ff(x, inputs[1], inputs[2])  # noqa: E731
ff1 = lambda y: ff(inputs[0], y, inputs[2])  # noqa: E731
ff2 = lambda z: ff(inputs[0], inputs[1], z)  # noqa: E731


def test_vmap():
    bdim = 2
    vinputs = (
        np.ones((bdim, V.dim())),
        np.ones((bdim, 1)) * 0.5,
        np.ones((bdim, 1)) * 0.6,
    )
    out = jax.vmap(ff)(*vinputs)
    with check:
        assert out.shape[0] == bdim
    with check:
        assert np.all(out == out[0])


def test_jvp():
    for func, inp in zip((ff0, ff1, ff2), inputs):
        dir_v = 0.432543 * np.ones_like(inp)
        fdm_jvp = fdm.jvp(func, dir_v)(inp)
        jax_jvp = jax.jvp(func, (inp,), (dir_v,))[1]
        with check:
            assert np.allclose(fdm_jvp, jax_jvp)


# TODO: jax.jacfwd outputs zero
# JAX Tracer goes inside fenics wrappers and zero array is returned
# because fenics numpy conversion works only for concrete arrays
@pytest.mark.xfail
def test_jacobian():
    # skipping ff0 as it is expensive with fdm
    for func, inp in zip((ff1, ff2), (inputs[1], inputs[2])):
        jax_jac = jax.jacfwd(func)(inp)
        fdm_jac = fdm.jacobian(func)(inp)
        with check:
            assert np.allclose(jax_jac, fdm_jac)
