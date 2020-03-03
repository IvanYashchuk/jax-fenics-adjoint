from pytest_check import check
import fdm
import jax
from jax.config import config
import jax.numpy as np

import fenics
import fenics_adjoint as fa
import ufl

from jaxfenics_adjoint import build_jax_fem_eval

config.update("jax_enable_x64", True)
fenics.parameters["std_out_all_processes"] = False
fenics.set_log_level(fenics.LogLevel.ERROR)

mesh = fa.UnitSquareMesh(3, 2)
V = fenics.FunctionSpace(mesh, "P", 1)


def solve_fenics(kappa0, kappa1):

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fa.Function(V)
    bcs = [fa.DirichletBC(V, fa.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fa.solve(F == 0, u, bcs=bcs)
    return u


templates = (fa.Constant(0.0), fa.Constant(0.0))
jax_solve_eval = build_jax_fem_eval(templates)(solve_fenics)

# multivariate output function
ff = lambda x, y: np.sqrt(np.square(jax_solve_eval(np.sqrt(x ** 3), y)))  # noqa: E731
x_input = np.ones(1)
y_input = 1.2 * np.ones(1)

# multivariate output function of the first argument
hh = lambda x: ff(x, y_input)  # noqa: E731
# multivariate output function of the second argument
gg = lambda y: ff(x_input, y)  # noqa: E731


def test_jacobian():
    fdm_jac0 = fdm.jacobian(hh)(x_input)
    jax_jac0 = jax.jacrev(hh)(x_input)

    with check:
        assert np.allclose(fdm_jac0, jax_jac0)

    rngkey = jax.random.PRNGKey(0)
    v = jax.random.normal(rngkey, shape=(V.dim(),), dtype="float64")

    fdm_vjp0 = v @ fdm_jac0
    jax_vjp0 = jax.vjp(hh, x_input)[1](v)

    with check:
        assert np.allclose(fdm_vjp0, jax_vjp0)

    fdm_jac1 = fdm.jacobian(gg)(y_input)
    jax_jac1 = jax.jacrev(gg)(y_input)

    with check:
        assert np.allclose(fdm_jac1, jax_jac1)

    rngkey = jax.random.PRNGKey(1)
    v = jax.random.normal(rngkey, shape=(V.dim(),), dtype="float64")

    fdm_vjp1 = v @ fdm_jac1
    jax_vjp1 = jax.vjp(gg, y_input)[1](v)

    with check:
        assert np.allclose(fdm_vjp1, jax_vjp1)


# scalar output function
f_scalar = lambda x, y: np.sqrt(  # noqa: E731
    np.sum(np.square(jax_solve_eval(np.sqrt(x ** 3), y)))
)
h_scalar = lambda x: f_scalar(x, y_input)  # noqa: E731


def test_grad():
    fdm_grad = fdm.gradient(h_scalar)(x_input)
    jax_grad = jax.grad(h_scalar)(x_input)

    with check:
        assert np.allclose(fdm_grad, jax_grad)

    jax_grads = jax.grad(f_scalar, (0, 1))(x_input, y_input)
    fdm_grad0 = fdm_grad
    fdm_grad1 = fdm.gradient(lambda y: f_scalar(x_input, y))(y_input)  # noqa: E731

    with check:
        assert np.allclose(fdm_grad0, jax_grads[0])
    with check:
        assert np.allclose(fdm_grad1, jax_grads[1])
