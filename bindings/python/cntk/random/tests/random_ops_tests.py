# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for random number generation operations; forward only of course
"""

from __future__ import division
import numpy as np
import pytest
import cntk as C
from cntk.tests.test_utils import precision, PRECISION_TO_TYPE
from cntk.ops.tests.ops_test_utils import cntk_device

DIST_PARAMS = [
    (0.1, 1),
    (0.5, 1),
    (0.9, 1),
    (0.1, 2),
    (0.5, 2),
    (0.9, 2)
]


@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_randomlike_moments(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    from cntk import random as cr

    x = C.input_variable(1, dtype=dt)
    N = 100000
    B = 50.0 / np.sqrt(N) # about 1.5 larger than the largest value ever observed
    x0 = np.zeros((N, 1), dtype=dt)
    eg = np.euler_gamma

    #                  op             mean,                   variance
    ops1 = [(cr.bernoulli_like, lambda a: a           , lambda a: a*(1-a))]

    #                  op             mean,                   variance
    ops2 = [(cr.uniform_like,   lambda a, b: (b+a)*0.5, lambda a,b: (b-a)**2/12.0   ),
            (cr.normal_like,    lambda a, b: a        , lambda a,b: b**2            ),
            (cr.gumbel_like,    lambda a, b: a+b*eg   , lambda a,b: (np.pi*b)**2/6.0)]

    for op, fmean, fvar in ops1:
        input_op = op(x, arg0, seed=98052)
        value = input_op.eval({x: x0}, device=dev)
        assert np.abs(np.mean(value) - fmean(arg0)) < B
        assert np.abs(np.var(value) - fvar(arg0)) < B

    for op, fmean, fvar in ops2:
        input_op = op(x, arg0, arg1, seed=98052)
        value = input_op.eval({x: x0}, device=dev)
        assert np.abs(np.mean(value) - fmean(arg0, arg1)) < B
        assert np.abs(np.var(value) - fvar(arg0, arg1)) < B