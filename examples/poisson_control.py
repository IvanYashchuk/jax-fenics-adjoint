r""" Solves a optimal control problem constrained by the Poisson equation:
    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2
    subject to
    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega
    This examples is based on:
    http://www.dolfin-adjoint.org/en/latest/documentation/poisson-mother/poisson-mother.html
"""

import jax
from jax.config import config

import jax.numpy as np
import numpy as onp

from scipy.optimize import minimize

import fenics as fn
import fenics_adjoint as fa
import ufl

from jaxfenics_adjoint import build_jax_fem_eval
from jaxfenics_adjoint import from_numpy

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
fn.set_log_level(fn.LogLevel.ERROR)

# Create mesh, refined in the center
n = 64
mesh = fn.UnitSquareMesh(n, n)

cf = fn.MeshFunction("bool", mesh, mesh.geometry().dim())
subdomain = fn.CompiledSubDomain(
    "std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25"
)
subdomain.mark(cf, True)
mesh = fa.Mesh(fn.refine(mesh, cf))

# Define discrete function spaces and functions
V = fn.FunctionSpace(mesh, "CG", 1)
W = fn.FunctionSpace(mesh, "DG", 0)

solve_templates = (fa.Function(W),)
assemble_templates = (fa.Function(V), fa.Function(W))

# Define and solve the Poisson equation
@build_jax_fem_eval(solve_templates)
def fenics_solve(f):
    u = fa.Function(V, name="State")
    v = fn.TestFunction(V)
    F = (ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * ufl.dx
    bcs = [fa.DirichletBC(V, 0.0, "on_boundary")]
    fa.solve(F == 0, u, bcs)
    return u


# Define functional of interest and the reduced functional
@build_jax_fem_eval(assemble_templates)
def fenics_cost(u, f):
    x = ufl.SpatialCoordinate(mesh)
    w = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    d = 1 / (2 * ufl.pi ** 2) * w
    alpha = fa.Constant(1e-6)
    J_form = (0.5 * ufl.inner(u - d, u - d)) * ufl.dx + alpha / 2 * f ** 2 * ufl.dx
    J = fa.assemble(J_form)
    return J


def obj_function(x):
    u = fenics_solve(x)
    cost = fenics_cost(u, x)
    return cost


def min_f(x):
    value, grad = jax.value_and_grad(obj_function)(x)
    return onp.array(value), onp.array(grad)


x0 = np.ones(W.dim())

res = minimize(
    min_f,
    x0,
    method="L-BFGS-B",
    jac=True,
    tol=1e-9,
    bounds=((0, 0.8),) * W.dim(),
    options={"gtol": 1e-10, "ftol": 0, "disp": True, "maxiter": 50},
)

# Define the expressions of the analytical solution
alpha = 1e-6
x = ufl.SpatialCoordinate(mesh)
w = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
f_analytic = 1 / (1 + alpha * 4 * ufl.pi ** 4) * w
u_analytic = 1 / (2 * ufl.pi ** 2) * f_analytic

f_opt = from_numpy(res.x, fn.Function(W))
u = fn.Function(V)
v = fn.TestFunction(V)
F = (ufl.inner(ufl.grad(u), ufl.grad(v)) - f_opt * v) * ufl.dx
bc = fn.DirichletBC(V, 0.0, "on_boundary")
fn.solve(F == 0, u, bc)
print(f"norm of f_opt is {fn.norm(f_opt)}")

# interpolatation of UFL forms does not work in FEniCS, hence projection
CG3 = fn.FunctionSpace(mesh, "CG", 3)
control_error = fn.errornorm(fn.project(f_analytic, CG3), f_opt)
state_error = fn.errornorm(fn.project(u_analytic, CG3), u)
print("h(min):           %e." % mesh.hmin())
print("Error in state:   %e." % state_error)
print("Error in control: %e." % control_error)

# Write solutions to XDMFFile, can be visualized with paraview
# First time step is approximated solution, second timestep is analytic
# solution

out_f = fn.XDMFFile("output/f_jax_scipy.xdmf")
out_f.write_checkpoint(f_opt, "f", 0.0, fn.XDMFFile.Encoding.HDF5, True)
out_f.write_checkpoint(
    fn.project(f_analytic, W), "f", 1.0, fn.XDMFFile.Encoding.HDF5, True
)

out_u = fn.XDMFFile("output/u_jax_scipy.xdmf")
out_u.write_checkpoint(u, "u", 0.0, fn.XDMFFile.Encoding.HDF5, True)
out_u.write_checkpoint(
    fn.project(u_analytic, V), "u", 1.0, fn.XDMFFile.Encoding.HDF5, True
)
