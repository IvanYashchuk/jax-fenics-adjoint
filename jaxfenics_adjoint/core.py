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

from fenics_numpy import evaluate_primal, evaluate_vjp, evaluate_jvp

from .helpers import FenicsVariable
from .helpers import jax_to_fenics_numpy

from typing import Type, List, Union, Iterable, Callable, Tuple


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

    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
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
            for i, vjp in enumerate(evaluate_vjp(g, fenics_output, fenics_inputs, tape))
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
        # apply function row-by-row
        res = list(map(vjp_fun1, *vector_arg_values))
        # transpose resulting list
        res_T = list(itertools.zip_longest(*res))
        return tuple(map(np.vstack, res_T)), (batch_axes[0],) * len(args)

    jax.interpreters.batching.primitive_batchers[vjp_fun1_p] = vjp_fun1_batch

    return numpy_output, vjp_fun1


def build_jax_fem_eval(fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))`.
    This is for reverse mode AD.
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
            lambda *args: evaluate_primal(fenics_function, fenics_templates, *args)[0]
        )

        jax_fem_eval_p.def_abstract_eval(
            lambda *args: jax.abstract_arrays.make_shaped_array(
                evaluate_primal(fenics_function, fenics_templates, *args)[0]
            )
        )

        def jax_fem_eval_batch(vector_arg_values, batch_axes):
            assert len(set(batch_axes)) == 1  # assert that all batch axes are same
            assert (
                batch_axes[0] == 0
            )  # assert that batch axis is zero, need to rewrite for a general case?
            res = list(map(jax_fem_eval, *vector_arg_values))
            res = np.asarray(res)
            return res, batch_axes[0]

        jax.interpreters.batching.primitive_batchers[
            jax_fem_eval_p
        ] = jax_fem_eval_batch

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
    """Return `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))`.
    This is for forward mode AD.
    Given the FEniCS-side function ofunc(*args), return the function
    `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))` with
    the JVP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = build_jax_fem_eval_fwd(*args)(ofunc(*args))`
    """

    def decorator(fenics_function: Callable) -> Callable:
        @functools.wraps(fenics_function)
        def jax_fem_eval(*args):
            return jax_fem_eval_p.bind(*args)

        jax_fem_eval_p = Primitive("jax_fem_eval")

        def jax_fem_eval_p_impl(*args):
            args = (
                jax_to_fenics_numpy(arg, ft) for arg, ft in zip(args, fenics_templates)
            )
            return evaluate_primal(fenics_function, fenics_templates, *args)[0]

        jax_fem_eval_p.def_impl(jax_fem_eval_p_impl)

        def jax_fem_eval_p_abstract_eval(*args):
            args = (
                jax_to_fenics_numpy(arg, ft) for arg, ft in zip(args, fenics_templates)
            )
            return jax.abstract_arrays.make_shaped_array(
                evaluate_primal(fenics_function, fenics_templates, *args)[0]
            )

        jax_fem_eval_p.def_abstract_eval(jax_fem_eval_p_abstract_eval)

        def jax_fem_eval_batch(vector_arg_values, batch_axes):
            assert len(set(batch_axes)) == 1  # assert that all batch axes are same
            assert (
                batch_axes[0] == 0
            )  # assert that batch axis is zero, need to rewrite for a general case?
            res = list(map(jax_fem_eval, *vector_arg_values))
            res = np.asarray(res)
            return res, batch_axes[0]

        jax.interpreters.batching.primitive_batchers[
            jax_fem_eval_p
        ] = jax_fem_eval_batch

        # @trace("jvp_jax_fem_eval")
        def jvp_jax_fem_eval(ps, ts):
            return jvp_jax_fem_eval_p.bind(ps, ts)

        jvp_jax_fem_eval_p = Primitive("jvp_jax_fem_eval")
        jvp_jax_fem_eval_p.multiple_results = True

        def jvp_jax_fem_eval_impl(ps, ts):
            ps = (jax_to_fenics_numpy(p, ft) for p, ft in zip(ps, fenics_templates))
            ts = (jax_to_fenics_numpy(t, ft) for t, ft in zip(ts, fenics_templates))
            return evaluate_jvp(fenics_function, fenics_templates, ps, ts)

        jvp_jax_fem_eval_p.def_impl(jvp_jax_fem_eval_impl)

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
