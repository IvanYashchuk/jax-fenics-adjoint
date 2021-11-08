import pytest

import firedrake
import numpy

from jaxfenics_adjoint import from_jax

import jax
from jax._src import ad_util
from jax.config import config
import jax.numpy

config.update("jax_enable_x64", True)

# Test JAX specific conversions here
make_shaped_array = jax.abstract_arrays.make_shaped_array
get_aval = jax.core.get_aval


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (make_shaped_array(jax.numpy.ones(1)), firedrake.Constant(0.0)),
        (make_shaped_array(jax.numpy.ones(2)), firedrake.Constant([0.0, 0.0])),
        (get_aval(jax.numpy.asarray(0.66)), firedrake.Constant(0.66)),
        (get_aval(jax.numpy.asarray([0.5, 0.66])), firedrake.Constant([0.5, 0.66]),),
        (ad_util.Zero(get_aval(jax.numpy.asarray(0.0))), firedrake.Constant(0.0)),
    ],
)
def test_from_jax_constant(test_input, expected):
    fenics_test_input = from_jax(test_input, firedrake.Constant(0.0))
    assert numpy.allclose(fenics_test_input.values(), expected.values())


def _x0(mesh):
    x = firedrake.SpatialCoordinate(mesh)
    return x[0]


@pytest.mark.parametrize(
    "test_input,expected_expr",
    [
        (make_shaped_array(jax.numpy.ones(10)), lambda mesh: firedrake.Constant(0.0)),
        (
            ad_util.Zero(get_aval(jax.numpy.asarray(0.0))),
            lambda mesh: firedrake.Constant(0.0),
        ),
        (get_aval(jax.numpy.linspace(0.05, 0.95, num=10)), _x0),
    ],
)
def test_from_jax_function(test_input, expected_expr):
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    template = firedrake.Function(V)
    fenics_test_input = from_jax(test_input, template)
    expected = firedrake.interpolate(expected_expr(mesh), V)
    assert numpy.allclose(
        fenics_test_input.vector().get_local(), expected.vector().get_local()
    )
