import pytest

import fenics
import numpy

from jaxfenics_adjoint import from_jax

import jax
from jax._src import ad_util
from jax.config import config
from jax.core import get_aval
from jax._src import ad_util
from jax._src.abstract_arrays import make_shaped_array
import jax.numpy

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (make_shaped_array(jax.numpy.ones(1)), fenics.Constant(0.0)),
        (make_shaped_array(jax.numpy.ones(2)), fenics.Constant([0.0, 0.0])),
        (get_aval(jax.numpy.asarray(0.66)), fenics.Constant(0.66)),
        (
            get_aval(jax.numpy.asarray([0.5, 0.66])),
            fenics.Constant([0.5, 0.66]),
        ),
        (ad_util.Zero(get_aval(jax.numpy.asarray(0.0))), fenics.Constant(0.0)),
    ],
)
def test_from_jax_constant(test_input, expected):
    fenics_test_input = from_jax(test_input, fenics.Constant(0.0))
    assert numpy.allclose(fenics_test_input.values(), expected.values())


@pytest.mark.parametrize(
    "test_input,expected_expr",
    [
        (make_shaped_array(jax.numpy.ones(10)), "0.0"),
        (ad_util.Zero(get_aval(jax.numpy.asarray(0.0))), "0.0"),
        (get_aval(jax.numpy.linspace(0.05, 0.95, num=10)), "x[0]"),
    ],
)
def test_from_jax_function(test_input, expected_expr):
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    template = fenics.Function(V)
    fenics_test_input = from_jax(test_input, template)
    expected = fenics.interpolate(fenics.Expression(expected_expr, degree=1), V)
    assert numpy.allclose(
        fenics_test_input.vector().get_local(), expected.vector().get_local()
    )
