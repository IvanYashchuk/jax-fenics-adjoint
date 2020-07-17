from pytest_check import check
import jax
from jax.config import config
import jax.numpy as np
import numpy as onp

import fenics
import fenics_adjoint as fa
import ufl

import fdm

# from jaxfenics_adjoint import fem_eval, vjp_fem_eval, jvp_fem_eval
from jaxfenics_adjoint import build_jax_fem_eval

# from jaxfenics_adjoint import fenics_to_numpy, numpy_to_fenics

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
# ff = lambda *args: fem_eval(assemble_fenics, templates, *args)[0]  # noqa: E731
# ff0 = lambda x: ff(x, inputs[1], inputs[2])  # noqa: E731
# ff1 = lambda y: ff(inputs[0], y, inputs[2])  # noqa: E731
# ff2 = lambda z: ff(inputs[0], inputs[1], z)  # noqa: E731


# def test_fenics_forward():
#     numpy_output, _, _, _, = fem_eval(assemble_fenics, templates, *inputs)
#     u1 = fa.interpolate(fa.Constant(1.0), V)
#     J = assemble_fenics(u1, fa.Constant(0.5), fa.Constant(0.6))
#     assert np.isclose(numpy_output, J)


# def test_vjp_assemble_eval():
#     numpy_output, vjp_fun = vjp_fem_eval(assemble_fenics, templates, *inputs)
#     g = np.ones_like(numpy_output)
#     vjp_out = vjp_fun(g)

#     fdm_jac0 = fdm.jacobian(ff0)(inputs[0])
#     fdm_jac1 = fdm.jacobian(ff1)(inputs[1])
#     fdm_jac2 = fdm.jacobian(ff2)(inputs[2])

#     check1 = np.allclose(vjp_out[0], fdm_jac0)
#     check2 = np.allclose(vjp_out[1], fdm_jac1)
#     check3 = np.allclose(vjp_out[2], fdm_jac2)
#     assert check1 and check2 and check3


# def test_jvp_assemble_eval():
#     primals = inputs
#     tangent0 = np.asarray(onp.random.normal(size=(V.dim(),)))
#     tangent1 = np.asarray(onp.random.normal(size=(1,)))
#     tangent2 = np.asarray(onp.random.normal(size=(1,)))
#     tangents = (tangent0, tangent1, tangent2)

#     fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
#     fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])
#     fdm_jvp2 = fdm.jvp(ff2, tangents[2])(primals[2])

#     _, out_tangent = jvp_fem_eval(assemble_fenics, templates, primals, tangents)

#     assert np.allclose(fdm_jvp0 + fdm_jvp1 + fdm_jvp2, out_tangent)


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
