import jax
from jax._src import ad_util
from jax._src.abstract_arrays import ConcreteArray, ShapedArray
import numpy as np

import fecr
from fecr import from_numpy
from jax._src import ad_util

import warnings

from typing import Type, List, Union, Iterable, Callable, Tuple

# Union[fenics.Constant, fenics.Function, firedrake.Constant, firedrake.Function, pyadjoint.AdjFloat]
BackendVariable = fecr._backends.BackendVariable
JAXArray = Union[jax.numpy.array, np.array]


def jax_to_fenics_numpy(jax_array: JAXArray, fem_variable: BackendVariable) -> np.array:
    """Convert JAX symbolic variables to concrete NumPy array compatible with FEniCS/Firedrake"""

    fem_backend = fecr._backends.get_backend(fem_variable)

    # JAX tracer specific part. Here we return zero values if tracer is not ConcreteArray type.
    if isinstance(jax_array, ad_util.Zero):
        if isinstance(fem_variable, fem_backend.lib.Constant):
            numpy_array = np.zeros_like(fem_variable.values())
            return numpy_array
        elif isinstance(fem_variable, fem_backend.lib.Function):
            numpy_array = np.zeros(fem_variable.vector().size())
            return numpy_array

    elif isinstance(jax_array, (jax.core.Tracer,)):
        numpy_array = jax.core.get_aval(jax_array)
        return numpy_array

    elif isinstance(jax_array, (ShapedArray,)):
        if not isinstance(jax_array, (ConcreteArray,)):
            warnings.warn(
                "Got JAX tracer type to convert to FEniCS/Firedrake. Returning zero."
            )
            numpy_array = np.zeros(jax_array.shape)
            return numpy_array

        elif isinstance(jax_array, (ConcreteArray,)):
            numpy_array = jax_array.val
            return numpy_array

    else:
        numpy_array = np.asarray(jax_array)
        return numpy_array


def from_jax(
    jax_array: JAXArray, fem_variable: BackendVariable
) -> BackendVariable:  # noqa: C901
    """Convert numpy/jax array to FEniCS/Firedrake/pyadjoint variable"""
    return from_numpy(jax_to_fenics_numpy(jax_array, fem_variable), fem_variable)
