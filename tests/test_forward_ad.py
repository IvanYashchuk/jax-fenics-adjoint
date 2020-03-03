from pytest_check import check
import fdm
import jax
from jax.config import config
import jax.numpy as np

import fenics
import fenics_adjoint as fa
import ufl

from jaxfenics_adjoint import build_jax_fem_eval_fwd

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
jax_fem_eval = build_jax_fem_eval_fwd(templates)(solve_fenics)

# multivariate output function
ff = lambda x, y: np.sqrt(np.square(jax_fem_eval(np.sqrt(x ** 3), y)))  # noqa: E731
x_input = np.ones(1)
y_input = 1.2 * np.ones(1)

# multivariate output function of the first argument
hh = lambda x: ff(x, y_input)  # noqa: E731
# multivariate output function of the second argument
gg = lambda y: ff(x_input, y)  # noqa: E731


def test_jvp():
    dir_v = 0.432543 * np.ones_like(x_input)
    fdm_jvp = fdm.jvp(hh, dir_v)(x_input)
    jax_jvp = jax.jvp(hh, (x_input,), (dir_v,))[1]

    assert np.allclose(fdm_jvp, jax_jvp)


# for input of size one we can get full jacobian using jvp
def test_jacobian():
    jax_fwd_jac = jax.jvp(
        ff, (x_input, y_input), (np.ones_like(x_input), np.ones_like(y_input))
    )[1]
    fdm_jac0 = fdm.jacobian(hh)(x_input)
    fdm_jac1 = fdm.jacobian(gg)(y_input)
    assert np.allclose(fdm_jac0 + fdm_jac1, jax_fwd_jac[:, None])
