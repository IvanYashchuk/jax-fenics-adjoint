import pytest

import fenics
import numpy

# from fenics_numpy import fenics_to_numpy, numpy_to_fenics
from jaxfenics_adjoint import jax_to_fenics

import jax
from jax.config import config
import jax.numpy

config.update("jax_enable_x64", True)

# Test JAX specific conversions here
make_shaped_array = jax.abstract_arrays.make_shaped_array
get_aval = jax.core.get_aval


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (make_shaped_array(jax.numpy.ones(1)), fenics.Constant(0.0)),
        (make_shaped_array(jax.numpy.ones(2)), fenics.Constant([0.0, 0.0])),
        (get_aval(jax.numpy.asarray(0.66)), fenics.Constant(0.66)),
        (get_aval(jax.numpy.asarray([0.5, 0.66])), fenics.Constant([0.5, 0.66]),),
        (jax.ad_util.Zero(get_aval(jax.numpy.asarray(0.0))), fenics.Constant(0.0)),
    ],
)
def test_jax_to_fenics_constant(test_input, expected):
    fenics_test_input = jax_to_fenics(test_input, fenics.Constant(0.0))
    assert numpy.allclose(fenics_test_input.values(), expected.values())


@pytest.mark.parametrize(
    "test_input,expected_expr",
    [
        (make_shaped_array(jax.numpy.ones(10)), "0.0"),
        (jax.ad_util.Zero(get_aval(jax.numpy.asarray(0.0))), "0.0"),
        (get_aval(jax.numpy.linspace(0.05, 0.95, num=10)), "x[0]"),
    ],
)
def test_jax_to_fenics_function(test_input, expected_expr):
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    template = fenics.Function(V)
    fenics_test_input = jax_to_fenics(test_input, template)
    expected = fenics.interpolate(fenics.Expression(expected_expr, degree=1), V)
    assert numpy.allclose(
        fenics_test_input.vector().get_local(), expected.vector().get_local()
    )
