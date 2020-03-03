import jax
from jax.config import config

import jax.numpy as np
import numpy as onp

# https://github.com/matthias-k/cyipopt
from ipopt import minimize_ipopt

import fenics as fn
import fenics_adjoint as fa
import ufl

from jaxfenics_adjoint import build_jax_fem_eval
from jaxfenics_adjoint import numpy_to_fenics, fenics_to_numpy

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

fn.set_log_level(fn.LogLevel.ERROR)

tr, sym, grad, Identity = ufl.tr, ufl.sym, ufl.grad, ufl.Identity
inner, dot, dx = ufl.inner, ufl.dot, ufl.dx

# Geometry and elasticity
t, h, L = 2.0, 1.0, 5.0  # Thickness, height and length
E, nu = 210e3, 0.3  # Young Modulus
G = E / (2.0 * (1.0 + nu))  # Shear Modulus
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Lambda


def simp(x):
    return eps + (1 - eps) * x ** p


max_volume = 0.4 * L * h  # Volume constraint
p = 4  # Exponent
eps = fa.Constant(1.0e-6)  # Epsilon for SIMP

# Mesh, Control and Solution Spaces
nelx = 192
nely = 64
mesh = fa.RectangleMesh.create(
    [fn.Point(0.0, 0.0), fn.Point(L, h)], [nelx, nely], fn.CellType.Type.triangle
)

V = fn.VectorFunctionSpace(mesh, "CG", 1)  # Displacements
C = fn.FunctionSpace(mesh, "CG", 1)  # Control

# Volumetric Load
q = -10.0 / t
b = fa.Constant((0.0, q))


def Left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < fn.DOLFIN_EPS


u_L = fa.Constant((0.0, 0.0))
bcs = [fa.DirichletBC(V, u_L, Left_boundary)]


@build_jax_fem_eval((fa.Function(C),))
def forward(x):
    u = fn.TrialFunction(V)
    w = fn.TestFunction(V)
    sigma = lmbda * tr(sym(grad(u))) * Identity(2) + 2 * G * sym(grad(u))  # Stress
    R = simp(x) * inner(sigma, grad(w)) * dx - dot(b, w) * dx
    a, L = ufl.lhs(R), ufl.rhs(R)
    u = fa.Function(V)
    fa.solve(a == L, u, bcs)
    return u


@build_jax_fem_eval((fa.Function(V), fa.Function(C)))
def eval_cost(u, x):
    J_form = dot(b, u) * dx + fa.Constant(1.0e-8) * dot(grad(x), grad(x)) * dx
    J = fa.assemble(J_form)
    return J


@build_jax_fem_eval((fa.Function(C),))
def eval_volume(rho):
    J_form = rho * ufl.dx
    J = fa.assemble(J_form)
    return J


def obj_function(x):
    u = forward(x)
    cost = eval_cost(u, x)
    return cost


def min_f(x):
    value, grad = jax.value_and_grad(obj_function)(x)
    return onp.array(value), onp.array(grad)


def volume_inequality_fun(rho):
    """Enforce the volume constraint g(rho) = V - rho*dx >= 0."""
    return max_volume - eval_volume(rho)


constraints = [
    {
        "type": "ineq",
        "fun": volume_inequality_fun,
        "jac": lambda x: jax.grad(volume_inequality_fun)(x),
    }
]

x0 = np.ones(C.dim()) * max_volume / (L * h)
res = minimize_ipopt(
    min_f,
    x0,
    jac=True,
    bounds=((0.0, 1.0),) * C.dim(),
    constraints=constraints,
    options={"print_level": 5, "max_iter": 100},
)

rho_opt_final = numpy_to_fenics(res.x, fa.Function(C))

c = fn.plot(rho_opt_final)
plt.colorbar(c)
plt.show()

# Save optimal solution for visualizing with ParaView
# with XDMFFile("1_dist_load/control_solution_1.xdmf") as f:
#     f.write(rho_opt)
