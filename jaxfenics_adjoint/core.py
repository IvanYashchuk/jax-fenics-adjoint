import jax
import jax.numpy as np
from jax._src import ad_util

from jax.core import Primitive
from jax.custom_derivatives import custom_vjp
from jax._src import ad_util

import dataclasses
import functools
import itertools

from fecr import evaluate_primal, evaluate_pullback, evaluate_pushforward
from pyadjoint.tape import Tape

from .helpers import BackendVariable
from .helpers import jax_to_fenics_numpy

from typing import Collection, Callable, Tuple


@dataclasses.dataclass
class PyadjointMetadata:
    fenics_output: BackendVariable
    fenics_inputs: Collection[BackendVariable]
    tape: Tape


def flatten_pyadjoint_metadata(pyadjoint_metadata):
    return (
        tuple(),
        (
            pyadjoint_metadata.fenics_output,
            pyadjoint_metadata.fenics_inputs,
            pyadjoint_metadata.tape,
        ),
    )


def unflatten_pyadjoint_metadata(aux_data, _):
    return PyadjointMetadata(*aux_data)


jax.tree_util.register_pytree_node(
    PyadjointMetadata, flatten_pyadjoint_metadata, unflatten_pyadjoint_metadata
)


def get_pullback_function(
    fenics_function: Callable, fenics_templates: Collection[BackendVariable]
) -> Callable:
    """Computes the gradients of the output with respect to the input
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
    Output:
        A Python callable representing the VJP map from output cotangents to input cotangents.
        The returned VJP function must accept a value with the same shape as the value of fun applied
        to the arguments and must return a tuple with length equal to the number of positional arguments to fun.
    """

    # @trace("vjp_fun1")
    def vjp_fun1(aux_args, g):
        return vjp_fun1_p.bind(aux_args, g)

    def vjp_fun1_p_impl(aux_args, g):
        fe_aux, args = aux_args
        fenics_output, fenics_inputs, tape = (
            fe_aux.fenics_output,
            fe_aux.fenics_inputs,
            fe_aux.tape,
        )
        return tuple(
            vjp if vjp is not None else ad_util.zeros_like_jaxval(args[i])
            for i, vjp in enumerate(
                evaluate_pullback(fenics_output, fenics_inputs, tape, g)
            )
        )

    vjp_fun1_p = Primitive("vjp_fun1")
    vjp_fun1_p.multiple_results = True
    vjp_fun1_p.def_impl(vjp_fun1_p_impl)

    # @trace("vjp_fun1_abstract_eval")
    def vjp_fun1_abstract_eval(aux_args, g):
        _, args = aux_args
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
        _, args = vector_arg_values[0]
        assert (
            batch_axes[0] is None
        )  # assert that batch axis is None, need to rewrite for a general case?
        assert (
            batch_axes[1] == 0
        )  # assert that batch axis is zero, need to rewrite for a general case?
        # apply function row-by-row
        vjp_fun1_partial = functools.partial(vjp_fun1, vector_arg_values[0])
        res = list(map(vjp_fun1_partial, *(vector_arg_values[1],)))
        # transpose resulting list
        res_T = list(itertools.zip_longest(*res))
        return tuple(map(np.vstack, res_T)), (batch_axes[1],) * len(args)

    jax.interpreters.batching.primitive_batchers[vjp_fun1_p] = vjp_fun1_batch

    return vjp_fun1


def build_jax_fem_eval(fenics_templates: BackendVariable) -> Callable:
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
        @custom_vjp
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

        def primal(*args):
            numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
                fenics_function, fenics_templates, *args
            )
            return (
                numpy_output,
                (PyadjointMetadata(fenics_output, fenics_inputs, tape), args),
            )

        def pullback(aux_args, g):
            pb_fn = get_pullback_function(fenics_function, fenics_templates)
            # for some reason output of get_pullback_function is a list but we need tuple
            return tuple(pb_fn(aux_args, g))

        jax_fem_eval.defvjp(primal, pullback)
        return jax_fem_eval

    return decorator


# it seems that it is not possible to define custom vjp and jvp rules simultaneously
# at least I did not figure out how to do this
# they override each other
# therefore here I create a separate wrapped function
def build_jax_fem_eval_fwd(fenics_templates: BackendVariable) -> Callable:
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

        def jvp_jax_fem_eval_impl(primals, tangents):
            primals = (
                jax_to_fenics_numpy(p, ft) for p, ft in zip(primals, fenics_templates)
            )
            numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
                fenics_function, fenics_templates, *primals
            )

            tangents = (
                jax_to_fenics_numpy(t, ft) for t, ft in zip(tangents, fenics_templates)
            )
            dnumpy_output = evaluate_pushforward(
                fenics_output, fenics_inputs, tape, tangents
            )
            return numpy_output, dnumpy_output

        jvp_jax_fem_eval_p.def_impl(jvp_jax_fem_eval_impl)

        jax.interpreters.ad.primitive_jvps[jax_fem_eval_p] = jvp_jax_fem_eval

        # TODO: JAX Tracer goes inside fenics wrappers and zero array is returned
        # because fenics numpy conversion works only for concrete arrays
        # vjp_jax_fem_eval_p = Primitive("vjp_jax_fem_eval")
        # vjp_jax_fem_eval_p.def_impl(
        #     lambda ct, *args: vjp_fem_eval(fenics_function, fenics_templates, *args)[1](
        #         ct
        #     )
        # )

        # jax.interpreters.ad.primitive_transposes[jax_fem_eval_p] = vjp_jax_fem_eval_p

        return jax_fem_eval

    return decorator
