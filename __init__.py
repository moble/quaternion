# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import division, print_function, absolute_import

import numpy as np

from .numpy_quaternion import (quaternion,
                               from_spherical_coords, from_euler_angles,
                               rotor_intrinsic_distance, rotor_chordal_distance,
                               rotation_intrinsic_distance, rotation_chordal_distance,
                               slerp, squad_evaluate)
from .quaternion_time_series import squad
from .calculus import derivative, definite_integral, indefinite_integral
from ._version import __version__

__doc_title__ = "Quaternion dtype for NumPy"
__doc__ = "Adds a quaternion dtype to NumPy."

__all__ = ['quaternion', 'from_spherical_coords', 'from_euler_angles',
           'rotor_intrinsic_distance', 'rotor_chordal_distance',
           'rotation_intrinsic_distance', 'rotation_chordal_distance',
           'slerp', 'squad_evaluate',
           'zero', 'one', 'x', 'y', 'z',
           'as_float_array', 'as_quat_array', 'as_spinor_array',
           'squad', 'derivative', 'definite_integral', 'indefinite_integral']

if 'quaternion' in np.__dict__:
    raise RuntimeError('The NumPy package already has a quaternion type')

np.quaternion = quaternion
np.typeDict['quaternion'] = np.dtype(quaternion)

zero = np.quaternion(0, 0, 0, 0)
one = np.quaternion(1, 0, 0, 0)
x = np.quaternion(0, 1, 0, 0)
y = np.quaternion(0, 0, 1, 0)
z = np.quaternion(0, 0, 0, 1)


def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    assert a.dtype == np.dtype(np.quaternion)
    av = a.view(np.float)
    av = av.reshape(a.shape + (4,))
    return av
    # return a.view(np.float).reshape(a.shape+(4,))


def as_quat_array(a):
    """View a float array as an array of quaternions

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The input array must have a final dimension whose size is
    divisible by four (or better yet *is* 4).

    We will not convert back from a spinor array because there is no
    unique convention for the spinors, so I don't want to mess with
    that.  Also, we want to discourage users from the slow,
    memory-copying process of swapping columns required for useful
    definitions of the spinors.

    """
    assert a.dtype == np.dtype(np.float)
    av = a.view(np.quaternion)
    if a.shape[-1] == 4:
        av = av.reshape(a.shape[:-1])
        # return a.view(np.quaternion).reshape(a.shape[:-1])
    else:
        av = av.reshape(a.shape[:-1] + (a.shape[-1] // 4,))
        # return a.view(np.quaternion).reshape(a.shape[:-1]+(a.shape[-1]//4,))
    return av


def as_spinor_array(a):
    """View a quaternion array as spinors in two-complex representation

    This function is relatively slow and scales poorly, because memory
    copying is apparently involved -- I think it's due to the
    "advanced indexing" required to swap the columns.

    """
    assert a.dtype == np.dtype(np.quaternion)
    # I'm not sure why it has to be so complicated, but all of these steps
    # appear to be necessary in this case.
    return a.view(np.float).reshape(a.shape + (4,))[..., [0, 3, 2, 1]].ravel().view(np.complex).reshape(a.shape + (2,))


def allclose(a, b, rtol=4*np.finfo(float).eps, atol=0.0, verbose=False):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    This function is essentially a copy of the `numpy.allclose` function,
    with different default tolerances, minor changes necessary to deal
    correctly with quaternions, and the verbose option.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    If either array contains one or more NaNs, False is returned.
    Infs are treated as equal if they are in the same place and of the same
    sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).  Default 4*eps.
    atol : float
        The absolute tolerance parameter (see Notes).  Default 0.0.

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    numpy.allclose

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.
     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.

    Examples
    --------
    >>> import numpy as np
    >>> import quaternion
    >>> q1 = quaternion.quaternion(1e10, 0, 0, 0)
    >>> q2 = quaternion.quaternion(1.00001e10, 0, 0, 0)
    >>> q3 = quaternion.quaternion(1.0001e10, 0, 0, 0)
    >>> q4 = quaternion.quaternion(1e-7, 0, 0, 0)
    >>> q5 = quaternion.quaternion(1e-8, 0, 0, 0)
    >>> q6 = quaternion.quaternion(1e-9, 0, 0, 0)
    >>> q7 = quaternion.quaternion(np.nan, 0, 0, 0)
    >>> quaternion.allclose([q1, q4], [q2, q5], rtol=1.e-5, atol=1.e-8)
    False
    >>> quaternion.allclose([q1, q5], [q2, q6], rtol=1.e-5, atol=1.e-8)
    True
    >>> quaternion.allclose([q1, q5], [q3, q6], rtol=1.e-5, atol=1.e-8)
    False
    >>> quaternion.allclose([quaternion.one, q7], [quaternion.one, q7], rtol=1.e-5, atol=1.e-8)
    False
    """
    x = np.array(a, copy=False, ndmin=1)
    y = np.array(b, copy=False, ndmin=1)

    xinf = np.isinf(x)
    yinf = np.isinf(y)
    if any(xinf) or any(yinf):
        # Check that x and y have inf's only in the same positions
        if not all(xinf == yinf):
            if verbose:
                print('not all(xinf == yinf)')
                equal = (xinf == yinf)
                for i, val in enumerate(equal):
                    if not val:
                        print('\nx[{0}]={1}\ny[{0}]={2}'.format(i, x[i], y[i]))
            return False
        # Check that sign of inf's in x and y is the same
        if not all(x[xinf] == y[xinf]):
            if verbose:
                print('not all(x[xinf] == y[xinf])')
                equal = (x[xinf] == y[xinf])
                for i, val in enumerate(equal):
                    if not val:
                        print('\nx[{0}]={1}\ny[{0}]={2}'.format(i, x[xinf][i], y[xinf][i]))
            return False

        x = x[~xinf]
        y = y[~xinf]

    # ignore invalid fpe's
    with np.errstate(invalid='ignore'):
        r = all(np.less_equal(abs(x - y), atol + rtol * abs(y)))
        if verbose and not r:
            lessequal = np.less_equal(abs(x - y), atol + rtol * abs(y))
            for i, val in enumerate(lessequal):
                if not val:
                    print('\nx[{0}]={1}\ny[{0}]={2}'.format(i, x[i], y[i])
                          + '\n{0} > {1} + {2} * {3} = {4}'.format(abs(x[i] - y[i]), atol, rtol, abs(y[i]),
                                                                   atol + rtol * abs(y[i])))

    return r
