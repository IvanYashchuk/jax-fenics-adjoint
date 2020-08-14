from pytest_check import check
import jax
from jax.config import config
import jax.numpy as np
import numpy as onp

import fenics
import fenics_adjoint as fa
import ufl

import fdm

from jaxfenics_adjoint import build_jax_fem_eval

config.update("jax_enable_x64", True)

mesh = fa.UnitSquareMesh(3, 2)
V = fenics.FunctionSpace(mesh, "P", 1)


def assemble_fenics(u, kappa0, kappa1):

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = fa.assemble(J_form)
    return J


templates = (fa.Function(V), fa.Constant(0.0), fa.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1) * 0.5, np.ones(1) * 0.6)

hh = build_jax_fem_eval(templates)(assemble_fenics)
hh0 = lambda x: hh(x, inputs[1], inputs[2])  # noqa: E731
hh1 = lambda y: hh(inputs[0], y, inputs[2])  # noqa: E731
hh2 = lambda z: hh(inputs[0], inputs[1], z)  # noqa: E731


def test_jacobian_and_vjp():
    fdm_jac0 = fdm.jacobian(hh0)(inputs[0])
    jax_jac0 = jax.jacrev(hh0)(inputs[0])
    with check:
        assert np.allclose(fdm_jac0, jax_jac0)
    v0 = np.asarray(onp.random.normal(size=(1,)))
    fdm_vjp0 = v0 @ fdm_jac0
    jax_vjp0 = jax.vjp(hh0, inputs[0])[1](v0)
    with check:
        assert np.allclose(fdm_vjp0, jax_vjp0)

    fdm_jac1 = fdm.jacobian(hh1)(inputs[1])
    jax_jac1 = jax.jacrev(hh1)(inputs[1])
    with check:
        assert np.allclose(fdm_jac1, jax_jac1)
    v1 = np.asarray(onp.random.normal(size=(1,)))
    fdm_vjp1 = v1 @ fdm_jac1
    jax_vjp1 = jax.vjp(hh1, inputs[1])[1](v1)
    with check:
        assert np.allclose(fdm_vjp1, jax_vjp1)

    fdm_jac2 = fdm.jacobian(hh2)(inputs[2])
    jax_jac2 = jax.jacrev(hh2)(inputs[2])
    with check:
        assert np.allclose(fdm_jac2, jax_jac2)
    v2 = np.asarray(onp.random.normal(size=(1,)))
    fdm_vjp2 = v2 @ fdm_jac2
    jax_vjp2 = jax.vjp(hh2, inputs[2])[1](v2)
    with check:
        assert np.allclose(fdm_vjp2, jax_vjp2)


def test_gradient():
    jax_grads = jax.grad(hh, (0, 1, 2))(*inputs)
    fdm_grad0 = fdm.gradient(hh0)(inputs[0])
    fdm_grad1 = fdm.gradient(hh1)(inputs[1])
    fdm_grad2 = fdm.gradient(hh2)(inputs[2])

    with check:
        assert np.allclose(fdm_grad0, jax_grads[0])
    with check:
        assert np.allclose(fdm_grad1, jax_grads[1])
    with check:
        assert np.allclose(fdm_grad2, jax_grads[2])
