from pytest_check import check
import fdm
import jax
from jax.config import config
import jax.numpy as np

import firedrake
import firedrake_adjoint
import ufl

from jaxfenics_adjoint import build_jax_fem_eval

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
jax_solve_eval = build_jax_fem_eval(templates)(solve_firedrake)

# multivariate output function
ff = lambda x, y, z: np.sqrt(  # noqa: E731
    np.square(jax_solve_eval(x, np.sqrt(y**3), z))
)
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


def test_jacobian_and_vjp():
    rngkey = jax.random.PRNGKey(0)
    v = jax.random.normal(rngkey, shape=(V.dim(),), dtype="float64")
    # skipping ff0 as it is expensive with fdm
    for func, inp in zip((ff1, ff2), (inputs[1], inputs[2])):
        fdm_jac = fdm.jacobian(func)(inp)
        jax_jac = jax.jacrev(func)(inp)

        with check:
            assert np.allclose(fdm_jac, jax_jac)

        fdm_vjp = v @ fdm_jac
        # jax.vjp now returns a tuple
        jax_vjp = jax.vjp(func, inp)[1](v)[0]

        with check:
            assert np.allclose(fdm_vjp, jax_vjp)


# scalar output function
f_scalar = lambda x, y, z: np.sqrt(  # noqa: E731
    np.sum(np.square(jax_solve_eval(x, np.sqrt(y**3), z)))
)
h_scalar = lambda y: f_scalar(inputs[0], y, inputs[2])  # noqa: E731
fs0 = lambda x: f_scalar(x, inputs[1], inputs[2])  # noqa: E731
fs1 = lambda y: f_scalar(inputs[0], y, inputs[2])  # noqa: E731
fs2 = lambda z: f_scalar(inputs[0], inputs[1], z)  # noqa: E731


def test_grad():
    fdm_grads = []
    for func, inp in zip((fs0, fs1, fs2), inputs):
        fdm_grad = fdm.gradient(func)(inp)
        fdm_grads.append(fdm_grad)
        jax_grad = jax.grad(func)(inp)

        with check:
            assert np.allclose(fdm_grad, jax_grad)

    jax_grads = jax.grad(f_scalar, (0, 1, 2))(*inputs)
    for i, fdm_grad in enumerate(fdm_grads):
        with check:
            assert np.allclose(fdm_grad, jax_grads[i])
