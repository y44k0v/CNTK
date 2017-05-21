# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
from __future__ import print_function
import numpy as np
import cntk as C
from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as auto_select
from cntk.default_options import default_override_or, get_default_override
from cntk.internal.swig_helper import typemap


@typemap
def uniform(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),),
            low=0.0, high=1.0, seed=auto_select, name=''):
    """uniform(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),), low=0.0, high=1.0, seed=auto_select, name='')
    Generates samples from the uniform distribution in the interval [`low`,`high`).

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        dynamic_axes (list or tuple): a list or tuple of dynamic axes (e.g., batch axis, time axis)
        low (float): lower end of the range of the random numbers
        high (float): upper end of the range of the random numbers
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    from cntk.cntk_py import uniform_random_variable
    from cntk.internal import sanitize_shape, sanitize_dtype_cntk, sanitize_dynamic_axes

    shape = sanitize_shape(shape)
    dtype = get_default_override(None, dtype=dtype)
    if dtype is None:
        dtype = np.float32
    dtype = sanitize_dtype_cntk(dtype)
    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)
    return uniform_random_variable(shape, dtype, dynamic_axes, low, high, seed, name)


@typemap
def normal(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),),
           mean=0.0, scale=1.0, seed=auto_select, name=''):
    """normal(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),), mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the normal distribution with mean `mean` and standard deviation `scale`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        dynamic_axes (list or tuple): a list or tuple of dynamic axes (e.g., batch axis, time axis)
        mean (float): mean of the distribution
        scale (float): scale (standard deviation) of the distribution
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    from cntk.cntk_py import normal_random_variable
    from cntk.internal import sanitize_shape, sanitize_dtype_cntk, sanitize_dynamic_axes

    shape = sanitize_shape(shape)
    dtype = get_default_override(None, dtype=dtype)
    if dtype is None:
        dtype = np.float32
    dtype = sanitize_dtype_cntk(dtype)
    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)
    return normal_random_variable(shape, dtype, dynamic_axes, mean, scale, seed, name)


@typemap
def gumbel(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),),
           loc=0.0, scale=1.0, seed=auto_select, name=''):
    """gumbel(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),), loc=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the Gumbel distribution with location `loc` and scale `scale`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        dynamic_axes (list or tuple): a list or tuple of dynamic axes (e.g., batch axis, time axis)
        loc (float): location of the distribution
        scale (float): scale of the distribution
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        `The Gumbel-Max Trick<https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>`_.

    """
    from cntk.cntk_py import gumbel_random_variable
    from cntk.internal import sanitize_shape, sanitize_dtype_cntk, sanitize_dynamic_axes

    shape = sanitize_shape(shape)
    dtype = get_default_override(None, dtype=dtype)
    if dtype is None:
        dtype = np.float32
    dtype = sanitize_dtype_cntk(dtype)
    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)
    return gumbel_random_variable(shape, dtype, dynamic_axes, loc, scale, seed, name)


@typemap
def bernoulli(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),),
              mean=0.5, seed=auto_select, name=''):
    """bernoulli(shape, dtype=default_override_or(np.float32), dynamic_axes=(C.Axis.default_batch_axis(),), mean=0.5, seed=auto_select, name='')
    Generates samples from the Bernoulli distribution with success probability `mean`.

    Args:
        shape (tuple): shape of the output (entries are independent random draws)
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        dynamic_axes (list or tuple): a list or tuple of dynamic axes (e.g., batch axis, time axis)
        mean (float): success probability
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    from cntk.cntk_py import bernoulli_random_variable
    from cntk.internal import sanitize_shape, sanitize_dtype_cntk, sanitize_dynamic_axes

    shape = sanitize_shape(shape)
    dtype = get_default_override(None, dtype=dtype)
    if dtype is None:
        dtype = np.float32
    dtype = sanitize_dtype_cntk(dtype)
    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)
    return bernoulli_random_variable(shape, dtype, dynamic_axes, mean, seed, name)


@typemap
def uniform_like(variable, low=0.0, high=1.0, seed=auto_select, name=''):
    """uniform_like(variable, low=0.0, high=1.0, seed=auto_select, name='')
    Generates samples from the uniform distribution in the interval [`low`,`high`).

    Args:
        variable: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        low (float): lower end of the range of the random numbers
        high (float): upper end of the range of the random numbers
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    from cntk.cntk_py import uniform_random_variable_like
    return uniform_random_variable_like(variable, low, high, seed, name)


@typemap
def normal_like(variable, mean=0.0, scale=1.0, seed=auto_select, name=''):
    """normal_like(variable, mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the normal distribution with mean `mean` and standard deviation `scale`.

    Args:
        variable: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        mean (float): mean of the distribution
        scale (float): scale (standard deviation) of the distribution
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    from cntk.cntk_py import normal_random_variable_like
    return normal_random_variable_like(variable.shape, variable.dtype, variable.dynamic_axes, mean, scale, seed, name)


@typemap
def gumbel_like(variable, loc=0.0, scale=1.0, seed=auto_select, name=''):
    """gumbel_like(variable, mean=0.0, scale=1.0, seed=auto_select, name='')
    Generates samples from the Gumbel distribution with location `loc` and scale `scale`.

    Args:
        variable: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        loc (float): location of the distribution
        scale (float): scale of the distribution
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        `The Gumbel-Max Trick<https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>`_.

    """
    from cntk.cntk_py import gumbel_random_variable_like
    return gumbel_random_variable_like(variable, loc, scale, seed, name)


@typemap
def bernoulli_like(variable, mean=0.5, seed=auto_select, name=''):
    """bernoulli_like(variable, mean=0.5, seed=auto_select, name='')
    Generates samples from the Bernoulli distribution with success probability `mean`.

    Args:
        variable: cntk variable (input, output, parameter, or constant) from which to copy the shape, data type, and dynamic axes.
        mean (float): success probability
        seed (int): pseudo random number generator seed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    """
    from cntk.cntk_py import bernoulli_random_variable_like
    return bernoulli_random_variable_like(variable, mean, seed, name)


