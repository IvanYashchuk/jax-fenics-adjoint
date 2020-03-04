# Example is based on
# http://www.dolfin-adjoint.org/en/latest/documentation/stokes-topology/stokes-topology.html

import jax
from jax.config import config

import jax.numpy as np
import numpy as onp

from scipy.optimize import minimize
from scipy.optimize import SR1
from scipy.optimize import NonlinearConstraint

import fenics
import fenics_adjoint
import ufl

import logging

from jaxfenics_adjoint import build_jax_fem_eval
from jaxfenics_adjoint import numpy_to_fenics, fenics_to_numpy

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

fenics.set_log_level(fenics.LogLevel.ERROR)
logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)

mu = fenics_adjoint.Constant(1.0)  # viscosity
alphaunderbar = 2.5 * mu / (100 ** 2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01 ** 2)  # parameter for \alpha
q = fenics_adjoint.Constant(
    0.01
)  # q value that controls difficulty/discrete-valuedness of solution


def alpha(rho):
    """Inverse permeability as a function of rho"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)


N = 20
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = (
    fenics_adjoint.Constant(1.0 / 3) * delta
)  # want the fluid to occupy 1/3 of the domain

mesh = fenics_adjoint.Mesh(
    fenics.RectangleMesh(fenics.Point(0.0, 0.0), fenics.Point(delta, 1.0), N, N)
)
A = fenics.FunctionSpace(mesh, "CG", 1)  # control function space

U_h = fenics.VectorElement("CG", mesh.ufl_cell(), 2)
P_h = fenics.FiniteElement("CG", mesh.ufl_cell(), 1)
W = fenics.FunctionSpace(mesh, U_h * P_h)  # mixed Taylor-Hood function space

# Define the boundary condition on velocity
(x, y) = ufl.SpatialCoordinate(mesh)
l = 1.0 / 6.0  # noqa: E741
gbar = 1.0
cond1 = ufl.And(ufl.gt(y, (1.0 / 4 - l / 2)), ufl.lt(y, (1.0 / 4 + l / 2)))
val1 = gbar * (1 - (2 * (y - 0.25) / l) ** 2)
cond2 = ufl.And(ufl.gt(y, (3.0 / 4 - l / 2)), ufl.lt(y, (3.0 / 4 + l / 2)))
val2 = gbar * (1 - (2 * (y - 0.75) / l) ** 2)
inflow_outflow = ufl.conditional(cond1, val1, ufl.conditional(cond2, val2, 0))
inflow_outflow_bc = fenics_adjoint.project(inflow_outflow, W.sub(0).sub(0).collapse())

solve_templates = (fenics_adjoint.Function(A),)
assemble_templates = (fenics_adjoint.Function(W), fenics_adjoint.Function(A))


@build_jax_fem_eval(solve_templates)
def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = fenics_adjoint.Function(W)
    (u, p) = fenics.split(w)
    (v, q) = fenics.TestFunctions(W)

    inner, grad, dx, div = ufl.inner, ufl.grad, ufl.dx, ufl.div
    F = (
        alpha(rho) * inner(u, v) * dx
        + inner(grad(u), grad(v)) * dx
        + inner(grad(p), v) * dx
        + inner(div(u), q) * dx
    )
    bcs = [
        fenics_adjoint.DirichletBC(W.sub(0).sub(1), 0, "on_boundary"),
        fenics_adjoint.DirichletBC(W.sub(0).sub(0), inflow_outflow_bc, "on_boundary"),
    ]
    fenics_adjoint.solve(F == 0, w, bcs=bcs)
    return w


@build_jax_fem_eval(assemble_templates)
def eval_cost_fem(w, rho):
    u, _ = fenics.split(w)
    J_form = (
        0.5 * ufl.inner(alpha(rho) * u, u) * ufl.dx
        + mu * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
    )
    J = fenics_adjoint.assemble(J_form)
    return J


@build_jax_fem_eval(solve_templates)
def eval_volume_fem(rho):
    # We want V - \int rho dx >= 0, so write this as \int V/delta - rho dx >= 0
    J_form = (V / delta - rho) * ufl.dx
    J = fenics_adjoint.assemble(J_form)
    return J


def obj_function(rho):
    w = forward(rho)
    cost = eval_cost_fem(w, rho)
    return cost


def min_f(x):
    value, grad = jax.value_and_grad(obj_function)(x)
    return onp.array(value), onp.array(grad)


fun_ncl = lambda x: eval_volume_fem(x)  # noqa: E731
volume_constraint = NonlinearConstraint(
    fun_ncl, 0.0, np.inf, jac=lambda x: jax.grad(fun_ncl)(x), hess=SR1()
)

x0 = np.ones(A.dim())
res = minimize(
    min_f,
    x0,
    method="trust-constr",
    jac=True,
    hessp=SR1(),
    tol=1e-7,
    constraints=volume_constraint,
    bounds=((0, 1.0),) * A.dim(),
    options={"verbose": 3, "gtol": 1e-7, "maxiter": 20},
)

q.assign(0.1)
res = minimize(
    min_f,
    res.x,
    method="trust-constr",
    jac=True,
    hessp=SR1(),
    tol=1e-7,
    constraints=volume_constraint,
    bounds=((0, 1.0),) * A.dim(),
    options={"verbose": 3, "gtol": 1e-7, "maxiter": 100},
)

rho_opt_final = numpy_to_fenics(res.x, fenics.Function(A))

c = fenics.plot(rho_opt_final)
plt.colorbar(c)
plt.show()

rho_opt_file = fenics.XDMFFile(
    fenics.MPI.comm_world, "output/control_solution_final.xdmf"
)
rho_opt_file.write(rho_opt_final)
