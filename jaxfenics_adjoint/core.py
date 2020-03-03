import fenics
import fenics_adjoint
import pyadjoint
import ufl

import jax
import jax.numpy as np

from jax.core import Primitive
from jax.interpreters.ad import defvjp, defvjp_all
from jax.api import defjvp_all

import functools
import itertools

from .helpers import (
    numpy_to_fenics,
    fenics_to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_fenics,
)
from .helpers import FenicsVariable

from typing import Type, List, Union, Iterable, Callable, Tuple


def fem_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: np.array,
) -> Tuple[np.array, FenicsVariable, Tuple[FenicsVariable], pyadjoint.Tape]:
    """Computes the output of a fenics_function and saves a corresponding gradient tape
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        fenics_templates (iterable of FenicsVariable): Templates for converting arrays to FEniCS types
        args (tuple): jax array representation of the input to fenics_function
    Output:
        numpy_output (np.array): JAX array representation of the output from fenics_function(*fenics_inputs)
        residual_form (ufl.Form): UFL Form for the residual used to solve the problem with fenics.solve(F==0, ...)
        fenics_inputs (list of FenicsVariable): FEniCS representation of the input args
    """

    check_input(fenics_templates, *args)
    fenics_inputs = convert_all_to_fenics(fenics_templates, *args)

    # Create tape associated with this forward pass
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    fenics_output = fenics_function(*fenics_inputs)

    if isinstance(fenics_output, tuple):
        raise ValueError("Only single output from FEniCS function is supported.")

    numpy_output = np.asarray(fenics_to_numpy(fenics_output))
    return numpy_output, fenics_output, fenics_inputs, tape


def vjp_fem_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: np.array,
) -> Tuple[np.array, Callable]:
    """Computes the gradients of the output with respect to the input
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        args (tuple): jax array representation of the input to fenics_function
    Output:
        A pair where the first element is the value of fun applied to the arguments and the second element
        is a Python callable representing the VJP map from output cotangents to input cotangents.
        The returned VJP function must accept a value with the same shape as the value of fun applied
        to the arguments and must return a tuple with length equal to the number of positional arguments to fun.
    """

    numpy_output, fenics_output, fenics_inputs, tape = fem_eval(
        fenics_function, fenics_templates, *args
    )

    # @trace("vjp_fun1")
    def vjp_fun1(g):
        return vjp_fun1_p.bind(g)

    vjp_fun1_p = Primitive("vjp_fun1")
    vjp_fun1_p.multiple_results = True
    vjp_fun1_p.def_impl(
        lambda g: tuple(
            vjp if vjp is not None else jax.ad_util.zeros_like_jaxval(args[i])
            for i, vjp in enumerate(
                vjp_fem_eval_impl(g, fenics_output, fenics_inputs, tape)
            )
        )
    )

    # @trace("vjp_fun1_abstract_eval")
    def vjp_fun1_abstract_eval(g):
        if len(args) > 1:
            return tuple(
                (jax.abstract_arrays.ShapedArray(arg.shape, arg.dtype) for arg in args)
            )
        else:
            return (
                jax.abstract_arrays.ShapedArray((1, *args[0].shape), args[0].dtype),
            )

    vjp_fun1_p.def_abstract_eval(vjp_fun1_abstract_eval)

    # @trace("vjp_fun1_batch")
    def vjp_fun1_batch(vector_arg_values, batch_axes):
        """Computes the batched version of the primitive.

        This must be a JAX-traceable function.

        Args:
            vector_arg_values: a tuple of arguments, each being a tensor of matching
            shape.
            batch_axes: the axes that are being batched. See vmap documentation.
        Returns:
            a tuple of the result, and the result axis that was batched.
        """
        # _trace("Using vjp_fun1 to compute the batch:")
        assert (
            batch_axes[0] == 0
        )  # assert that batch axis is zero, need to rewrite for a general case?
        # compute function row-by-row
        res = [
            vjp_fun1(vector_arg_values[0][i])
            for i in range(vector_arg_values[0].shape[0])
        ]
        # transpose resulting list
        res_T = list(itertools.zip_longest(*res))
        return tuple(map(np.vstack, res_T)), (batch_axes[0],) * len(args)

    jax.batching.primitive_batchers[vjp_fun1_p] = vjp_fun1_batch

    return numpy_output, vjp_fun1


# @trace("vjp_fem_eval_impl")
def vjp_fem_eval_impl(
    g: np.array,
    fenics_output: FenicsVariable,
    fenics_inputs: Iterable[FenicsVariable],
    tape: pyadjoint.Tape,
) -> Tuple[np.array]:
    """Computes the gradients of the output with respect to the inputs."""
    # Convert tangent covector (adjoint) to a FEniCS variable
    adj_value = numpy_to_fenics(g, fenics_output)
    if isinstance(adj_value, (fenics.Function, fenics_adjoint.Function)):
        adj_value = adj_value.vector()

    tape.reset_variables()
    fenics_output.block_variable.adj_value = adj_value
    with tape.marked_nodes(fenics_inputs):
        tape.evaluate_adj(markings=True)
    fenics_grads = [fi.block_variable.adj_value for fi in fenics_inputs]

    # Convert FEniCS gradients to jax array representation
    jax_grads = (
        None if fg is None else np.asarray(fenics_to_numpy(fg)) for fg in fenics_grads
    )

    jax_grad_tuple = tuple(jax_grads)

    return jax_grad_tuple


def jvp_fem_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    primals: Tuple[np.array],
    tangents: Tuple[np.array],
) -> Tuple[np.array]:
    """Computes the tangent linear model
    """

    numpy_output_primal, fenics_output_primal, fenics_primals, tape = fem_eval(
        fenics_function, fenics_templates, *primals
    )

    # Now tangent evaluation!
    tape.reset_variables()

    fenics_tangents = convert_all_to_fenics(fenics_primals, *tangents)
    for fp, ft in zip(fenics_primals, fenics_tangents):
        fp.block_variable.tlm_value = ft

    tape.evaluate_tlm()

    fenics_output_tangent = fenics_output_primal.block_variable.tlm_value
    jax_output_tangent = np.asarray(fenics_to_numpy(fenics_output_tangent))

    return numpy_output_primal, jax_output_tangent


def build_jax_fem_eval(fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))`.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))` with
    the VJP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_jax_fem_eval(*args)(ofunc(*args))`
    """

    def decorator(fenics_function: Callable) -> Callable:
        @functools.wraps(fenics_function)
        def jax_fem_eval(*args):
            return jax_fem_eval_p.bind(*args)

        jax_fem_eval_p = Primitive("jax_fem_eval")
        jax_fem_eval_p.def_impl(
            lambda *args: fem_eval(fenics_function, fenics_templates, *args)[0]
        )

        jax_fem_eval_p.def_abstract_eval(
            lambda *args: jax.abstract_arrays.make_shaped_array(
                fem_eval(fenics_function, fenics_templates, *args)[0]
            )
        )

        def jax_fem_eval_batch(vector_arg_values, batch_axes):
            assert len(set(batch_axes)) == 1  # assert that all batch axes are same
            assert (
                batch_axes[0] == 0
            )  # assert that batch axis is zero, need to rewrite for a general case?
            # compute function row-by-row
            res = np.asarray(
                [
                    jax_fem_eval(
                        *(vector_arg_values[j][i] for j in range(len(batch_axes)))
                    )
                    for i in range(vector_arg_values[0].shape[0])
                ]
            )
            return res, batch_axes[0]

        jax.batching.primitive_batchers[jax_fem_eval_p] = jax_fem_eval_batch

        # @trace("djax_fem_eval")
        def djax_fem_eval(*args):
            return djax_fem_eval_p.bind(*args)

        djax_fem_eval_p = Primitive("djax_fem_eval")
        # djax_fem_eval_p.multiple_results = True
        djax_fem_eval_p.def_impl(
            lambda *args: vjp_fem_eval(fenics_function, fenics_templates, *args)
        )

        defvjp_all(jax_fem_eval_p, djax_fem_eval)
        return jax_fem_eval

    return decorator


# it seems that it is not possible to define custom vjp and jvp rules simultaneusly
# at least I did not figure out how to do this
# they override each other
# therefore here I create a separate wrapped function
def build_jax_fem_eval_fwd(fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))`. This is forward mode AD.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))` with
    the JVP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_jax_fem_eval(*args)(ofunc(*args))`
    """

    def decorator(fenics_function: Callable) -> Callable:
        @functools.wraps(fenics_function)
        def jax_fem_eval(*args):
            return jax_fem_eval_p.bind(*args)

        jax_fem_eval_p = Primitive("jax_fem_eval")
        jax_fem_eval_p.def_impl(
            lambda *args: fem_eval(fenics_function, fenics_templates, *args)[0]
        )

        jax_fem_eval_p.def_abstract_eval(
            lambda *args: jax.abstract_arrays.make_shaped_array(
                fem_eval(fenics_function, fenics_templates, *args)[0]
            )
        )

        def jax_fem_eval_batch(vector_arg_values, batch_axes):
            assert len(set(batch_axes)) == 1  # assert that all batch axes are same
            assert (
                batch_axes[0] == 0
            )  # assert that batch axis is zero, need to rewrite for a general case?
            # compute function row-by-row
            res = np.asarray(
                [
                    jax_fem_eval(
                        *(vector_arg_values[j][i] for j in range(len(batch_axes)))
                    )
                    for i in range(vector_arg_values[0].shape[0])
                ]
            )
            return res, batch_axes[0]

        jax.batching.primitive_batchers[jax_fem_eval_p] = jax_fem_eval_batch

        # @trace("jvp_jax_fem_eval")
        def jvp_jax_fem_eval(ps, ts):
            return jvp_jax_fem_eval_p.bind(ps, ts)

        jvp_jax_fem_eval_p = Primitive("jvp_jax_fem_eval")
        jvp_jax_fem_eval_p.multiple_results = True
        jvp_jax_fem_eval_p.def_impl(
            lambda ps, ts: jvp_fem_eval(fenics_function, fenics_templates, ps, ts)
        )

        jax.interpreters.ad.primitive_jvps[jax_fem_eval_p] = jvp_jax_fem_eval

        # TODO: JAX Tracer goes inside fenics wrappers and zero array is returned
        # because fenics numpy conversion works only for concrete arrays
        vjp_jax_fem_eval_p = Primitive("vjp_jax_fem_eval")
        vjp_jax_fem_eval_p.def_impl(
            lambda ct, *args: vjp_fem_eval(fenics_function, fenics_templates, *args)[1](
                ct
            )
        )

        jax.interpreters.ad.primitive_transposes[jax_fem_eval_p] = vjp_jax_fem_eval_p

        return jax_fem_eval

    return decorator
