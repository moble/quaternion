# Copyright (c) 2016, Michael Boyle
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
    if not t:
        return np.quaternion(*(np.sum(as_float_array(R)))).normalized()
    mean = np.empty((4,), dtype=float)
    definite_integral(as_float_array(R), t, mean)
    return np.quaternion(*mean).normalized()


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
    pass
