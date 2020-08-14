import fenics
import pyadjoint
import jax
import numpy as np

from fenics_numpy import numpy_to_fenics

import warnings

from typing import Type, List, Union, Iterable, Callable, Tuple

FenicsVariable = Union[fenics.Constant, fenics.Function]
JAXArray = Union[jax.numpy.array, np.array]


def jax_to_fenics_numpy(
    jax_array: JAXArray, fenics_var_template: FenicsVariable
) -> np.array:
    """Convert JAX symbolic variables to concrete NumPy array compatible with FEniCS"""

    # JAX tracer specific part. Here we return zero values if tracer is not ConcreteArray type.
    if isinstance(jax_array, jax.ad_util.Zero):
        if isinstance(fenics_var_template, fenics.Constant):
            numpy_array = np.zeros_like(fenics_var_template.values())
            return numpy_array
        elif isinstance(fenics_var_template, fenics.Function):
            numpy_array = np.zeros(fenics_var_template.vector().size())
            return numpy_array

    elif isinstance(jax_array, (jax.core.Tracer,)):
        numpy_array = jax.core.get_aval(jax_array)
        return numpy_array

    elif isinstance(jax_array, (jax.abstract_arrays.ShapedArray,)):
        if not isinstance(jax_array, (jax.abstract_arrays.ConcreteArray,)):
            warnings.warn("Got JAX tracer type to convert to FEniCS. Returning zero.")
            numpy_array = np.zeros(jax_array.shape)
            return numpy_array

        elif isinstance(jax_array, (jax.abstract_arrays.ConcreteArray,)):
            numpy_array = jax_array.val
            return numpy_array

    else:
        numpy_array = np.asarray(jax_array)
        return numpy_array


def jax_to_fenics(
    jax_array: JAXArray, fenics_var_template: FenicsVariable
) -> FenicsVariable:  # noqa: C901
    """Convert numpy/jax array to FEniCS variable"""
    return numpy_to_fenics(
        jax_to_fenics_numpy(jax_array, fenics_var_template), fenics_var_template
    )
