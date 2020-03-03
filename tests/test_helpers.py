import pytest

import fenics
import numpy
from jaxfenics_adjoint import fenics_to_numpy, numpy_to_fenics

import jax
from jax.config import config
import jax.numpy

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (fenics.Constant(0.66), numpy.asarray(0.66)),
        (fenics.Constant([0.5, 0.66]), numpy.asarray([0.5, 0.66])),
    ],
)
def test_fenics_to_numpy_constant(test_input, expected):
    assert numpy.allclose(fenics_to_numpy(test_input), expected)


def test_fenics_to_numpy_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    test_input = fenics.interpolate(fenics.Expression("x[0]", degree=1), V)
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(fenics_to_numpy(test_input), expected)


def test_fenics_to_numpy_mixed_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = fenics.UnitIntervalMesh(10)
    vec_dim = 4
    V = fenics.VectorFunctionSpace(mesh, "DG", 0, dim=vec_dim)
    test_input = fenics.interpolate(
        fenics.Expression(vec_dim * ("x[0]",), element=V.ufl_element()), V
    )
    expected = numpy.linspace(0.05, 0.95, num=10)
    expected = numpy.reshape(numpy.tile(expected, (4, 1)).T, V.dim())
    assert numpy.allclose(fenics_to_numpy(test_input), expected)


def test_fenics_to_numpy_vector():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    test_input = fenics.interpolate(fenics.Expression("x[0]", degree=1), V)
    test_input_vector = test_input.vector()
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(fenics_to_numpy(test_input_vector), expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (numpy.asarray(0.66), fenics.Constant(0.66)),
        (numpy.asarray([0.5, 0.66]), fenics.Constant([0.5, 0.66])),
    ],
)
def test_numpy_to_fenics_constant(test_input, expected):
    fenics_test_input = numpy_to_fenics(test_input, fenics.Constant(0.0))
    assert numpy.allclose(fenics_test_input.values(), expected.values())


def test_numpy_to_fenics_function():
    test_input = numpy.linspace(0.05, 0.95, num=10)
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    template = fenics.Function(V)
    fenics_test_input = numpy_to_fenics(test_input, template)
    expected = fenics.interpolate(fenics.Expression("x[0]", degree=1), V)
    assert numpy.allclose(
        fenics_test_input.vector().get_local(), expected.vector().get_local()
    )


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
        (jax.ad_util.Zero(), fenics.Constant(0.0)),
    ],
)
def test_jax_to_fenics_constant(test_input, expected):
    fenics_test_input = numpy_to_fenics(test_input, fenics.Constant(0.0))
    assert numpy.allclose(fenics_test_input.values(), expected.values())


@pytest.mark.parametrize(
    "test_input,expected_expr",
    [
        (make_shaped_array(jax.numpy.ones(10)), "0.0"),
        (jax.ad_util.Zero(), "0.0"),
        (get_aval(jax.numpy.linspace(0.05, 0.95, num=10)), "x[0]"),
    ],
)
def test_jax_to_fenics_function(test_input, expected_expr):
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "DG", 0)
    template = fenics.Function(V)
    fenics_test_input = numpy_to_fenics(test_input, template)
    expected = fenics.interpolate(fenics.Expression(expected_expr, degree=1), V)
    assert numpy.allclose(
        fenics_test_input.vector().get_local(), expected.vector().get_local()
    )
