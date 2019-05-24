# Copyright (c) 2017, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import division, print_function, absolute_import

import numpy as np

from .calculus import definite_integral


def mean_rotor_in_chordal_metric(R, t=None):
    """Return rotor that is closest to all R in the least-squares sense

    This can be done (quasi-)analytically because of the simplicity of
    the chordal metric function.  The only approximation is the simple
    2nd-order discrete formula for the definite integral of the input
    rotor function.

    Note that the `t` argument is optional.  If it is present, the
    times are used to weight the corresponding integral.  If it is not
    present, a simple sum is used instead (which may be slightly
    faster).

    """
    if t is None:
        return np.sum(R).normalized()
    mean = definite_integral(R, t)
    return mean.normalized()


def optimal_alignment_in_chordal_metric(Ra, Rb, t=None):
    """Return Rd such that Rd*Rb is as close to Ra as possible

    This function simply encapsulates the mean rotor of Ra/Rb.

    As in the `mean_rotor_in_chordal_metric` function, the `t`
    argument is optional.  If it is present, the times are used to
    weight the corresponding integral.  If it is not present, a simple
    sum is used instead (which may be slightly faster).

    """
    return mean_rotor_in_chordal_metric(Ra / Rb, t)


def mean_rotor_in_intrinsic_metric(R, t=None):
    raise NotImplementedError()
