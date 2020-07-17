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

    # if not isinstance(jax_array, np.ndarray):
    #     return _jax_to_fenics(jax_array, fenics_var_template)
    #     numpy_array = jax_to_fenics_numpy(jax_array, fenics_var_template)
    #     return numpy_to_fenics(numpy_array, fenics_var_template)
    # else:
    return numpy_to_fenics(
        jax_to_fenics_numpy(jax_array, fenics_var_template), fenics_var_template
    )


def _jax_to_fenics(numpy_array, fenics_var_template):  # noqa: C901
    """Convert numpy/jax array to FEniCS variable"""

    if isinstance(fenics_var_template, fenics.Constant):

        # JAX tracer specific part. Here we return zero values if tracer is not ConcreteArray type.
        if isinstance(numpy_array, (jax.core.Tracer,)):
            numpy_array = jax.core.get_aval(numpy_array)

        if isinstance(numpy_array, (jax.abstract_arrays.ShapedArray,)):
            if not isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
                warnings.warn(
                    "Got JAX tracer type to convert to FEniCS. Returning zero."
                )
                if numpy_array.shape == (1,):
                    return type(fenics_var_template)(0.0)
                else:
                    return type(fenics_var_template)(
                        np.zeros_like(fenics_var_template.values())
                    )

        if isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
            numpy_array = numpy_array.val

        if isinstance(numpy_array, jax.ad_util.Zero):
            numpy_array = np.zeros_like(fenics_var_template.values())

        if numpy_array.shape == (1,):
            return type(fenics_var_template)(numpy_array[0])
        else:
            return type(fenics_var_template)(numpy_array)

    if isinstance(fenics_var_template, fenics.Function):
        function_space = fenics_var_template.function_space()

        u = type(fenics_var_template)(function_space)

        if isinstance(numpy_array, jax.ad_util.Zero):
            return u

        # assume that given numpy/jax array is global array that needs to be distrubuted across processes
        # when FEniCS function is created
        fenics_size = u.vector().size()
        np_size = numpy_array.size

        if np_size != fenics_size:
            err_msg = (
                f"Cannot convert numpy array to Function:"
                f"Wrong size {numpy_array.size} vs {u.vector().size()}"
            )
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, "
                "but got {numpy_array.dtype}"
            )
            raise ValueError(err_msg)

        if isinstance(numpy_array, (jax.core.Tracer,)):
            numpy_array = jax.core.get_aval(numpy_array)

        if isinstance(numpy_array, (jax.abstract_arrays.ShapedArray,)):
            if not isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
                warnings.warn(
                    "Got JAX tracer type to convert to FEniCS. Returning zero."
                )
                return u

        if isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
            numpy_array = numpy_array.val

        range_begin, range_end = u.vector().local_range()
        # up to this point `numpy_array` could be JAX array
        # get NumPy array instead of JAX because the following slicing and reshaping is extremely slow for JAX arrays
        numpy_array = np.asarray(numpy_array)
        local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
        u.vector().set_local(local_array)
        u.vector().apply("insert")
        return u

    if isinstance(fenics_var_template, pyadjoint.AdjFloat):
        if isinstance(numpy_array, (jax.core.Tracer,)):
            numpy_array = jax.core.get_aval(numpy_array)
        if isinstance(numpy_array, (jax.abstract_arrays.ShapedArray,)):
            if not isinstance(numpy_array, (jax.abstract_arrays.ConcreteArray,)):
                warnings.warn(
                    "Got JAX tracer type to convert to FEniCS. Returning zero."
                )
                return 0.0
        return float(numpy_array)

    err_msg = f"Cannot convert numpy/jax array to {fenics_var_template}"
    raise ValueError(err_msg)
