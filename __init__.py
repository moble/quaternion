# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import division, print_function, absolute_import

import numpy as np

from .numpy_quaternion import (quaternion, _eps,
                               from_spherical_coords, from_euler_angles,
                               rotor_intrinsic_distance, rotor_chordal_distance,
                               rotation_intrinsic_distance, rotation_chordal_distance,
                               slerp_evaluate, squad_evaluate,
                               # slerp_vectorized, squad_vectorized,
                               # slerp, squad,
                               )
from .quaternion_time_series import slerp, squad
from .calculus import derivative, definite_integral, indefinite_integral
from ._version import __version__

__doc_title__ = "Quaternion dtype for NumPy"
__doc__ = "Adds a quaternion dtype to NumPy."

__all__ = ['quaternion', 'from_spherical_coords', 'from_euler_angles',
           'rotor_intrinsic_distance', 'rotor_chordal_distance',
           'rotation_intrinsic_distance', 'rotation_chordal_distance',
           'slerp_evaluate', 'squad_evaluate',
           'zero', 'one', 'x', 'y', 'z',
           'as_float_array', 'as_quat_array', 'as_spinor_array',
           'squad', 'slerp', 'derivative', 'definite_integral', 'indefinite_integral']

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
    a = np.atleast_1d(a)
    assert a.dtype == np.dtype(np.quaternion)
    if a.shape == ():
        return a.components
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
    if a.shape == (4,):
        return quaternion(a[0], a[1], a[2], a[3])
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
    a = np.atleast_1d(a)
    assert a.dtype == np.dtype(np.quaternion)
    # I'm not sure why it has to be so complicated, but all of these steps
    # appear to be necessary in this case.
    return a.view(np.float).reshape(a.shape + (4,))[..., [0, 3, 2, 1]].ravel().view(np.complex).reshape(a.shape + (2,))


def as_rotation_matrix(q):
    """Convert input quaternion to 3x3 rotation matrix

    Parameters
    ----------
    q: quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must all be nonzero

    Returns
    -------
    rot: float array
        Output shape is q.shape+(3,3).  This matrix should multiply (from
        the left) a column vector to produce the rotated column vector.

    Raises
    ------
    ZeroDivisionError
        If any of the input quaternions have norm 0.0.

    """
    if q.shape == ():  # This is just a single quaternion
        n = q.norm()
        if n == 0.0:
            raise ZeroDivisionError("Input to `as_rotation_matrix({0})` has zero norm".format(q))
        elif abs(n-1.0) < _eps:  # Input q is basically normalized
            return np.array([
                [1 - 2*(q.y**2 + q.z**2),   2*(q.x*q.y - q.z*q.w),      2*(q.x*q.z + q.y*q.w)],
                [2*(q.x*q.y + q.z*q.w),     1 - 2*(q.x**2 + q.z**2),    2*(q.y*q.z - q.x*q.w)],
                [2*(q.x*q.z - q.y*q.w),     2*(q.y*q.z + q.x*q.w),      1 - 2*(q.x**2 + q.y**2)]
            ])
        else:  # Input q is not normalized
            return np.array([
                [1 - 2*(q.y**2 + q.z**2)/n,   2*(q.x*q.y - q.z*q.w)/n,      2*(q.x*q.z + q.y*q.w)/n],
                [2*(q.x*q.y + q.z*q.w)/n,     1 - 2*(q.x**2 + q.z**2)/n,    2*(q.y*q.z - q.x*q.w)/n],
                [2*(q.x*q.z - q.y*q.w)/n,     2*(q.y*q.z + q.x*q.w)/n,      1 - 2*(q.x**2 + q.y**2)/n]
            ])
    else:  # This is an array of quaternions
        n = np.norm(q)
        if np.any(n == 0.0):
            raise ZeroDivisionError("Array input to `as_rotation_matrix` has at least one element with zero norm")
        else:  # Assume input q is not normalized
            m = np.empty(q.shape + (3, 3))
            q = as_float_array(q)
            m[..., 0, 0] = 1.0 - 2*(q[..., 2]**2 + q[..., 3]**2)/n
            m[..., 0, 1] = 2*(q[..., 1]*q[..., 2] - q[..., 3]*q[..., 0])/n
            m[..., 0, 2] = 2*(q[..., 1]*q[..., 3] + q[..., 2]*q[..., 0])/n
            m[..., 1, 0] = 2*(q[..., 1]*q[..., 2] + q[..., 3]*q[..., 0])/n
            m[..., 1, 1] = 1.0 - 2*(q[..., 1]**2 + q[..., 3]**2)/n
            m[..., 1, 2] = 2*(q[..., 2]*q[..., 3] - q[..., 1]*q[..., 0])/n
            m[..., 2, 0] = 2*(q[..., 1]*q[..., 3] - q[..., 2]*q[..., 0])/n
            m[..., 2, 1] = 2*(q[..., 2]*q[..., 3] + q[..., 1]*q[..., 0])/n
            m[..., 2, 2] = 1.0 - 2*(q[..., 1]**2 + q[..., 2]**2)/n
            return m


def from_rotation_matrix(rot, accurate=True):
    """Convert input 3x3 rotation matrix to unit quaternion

    By default, if scipy.linalg is available, this function uses
    Bar-Itzhack's algorithm to allow for non-orthogonal matrices.
    [J. Guidance, Vol. 23, No. 6, p. 1085 <http://dx.doi.org/10.2514/2.4654>]
    This will almost certainly be quite a bit slower than simpler versions,
    though it will be more robust to numerical errors in the rotation matrix.
    Also note that Bar-Itzhack uses some pretty weird conventions.  The last
    component of the quaternion appears to represent the scalar, and the
    quaternion itself is conjugated relative to the convention used
    throughout this module.

    If scipy.linalg is not available or if the optional `accurate` parameter
    is set to `False`, this function falls back to the possibly faster, but
    less accurate, algorithm of Shoemake [ACM SIGGRAPH Comp. Graph., Vol. 19,
    No. 3, p. 245 (1985), <http://dx.doi.org/10.1145/325165.325242>].

    Parameters
    ----------
    rot: (Nx3x3) float array
        Each 3x3 matrix represents a rotation by multiplying (from the left)
        a column vector to produce a rotated column vector.  Note that this
        input may actually have ndims>3; it is just assumed that the last
        two dimensions have size 3, representing the matrix.
    accurate: bool, optional
        If scipy.linalg is available, use the more accurate algorithm of
        Bar-Itzhack.  Default value is True.

    Returns
    -------
    q: array of quaternions
        Unit quaternions resulting in rotations corresponding to input
        rotations.  Output shape is rot.shape[:-2].

    Raises
    ------
    LinAlgError
        If any of the eigenvalue solutions does not converge

    """
    try:
        from scipy import linalg
    except ImportError:
        linalg = False

    rot = np.array(rot, copy=False)
    shape = rot.shape[:-2]

    if linalg and accurate:
        from operator import mul
        from functools import reduce

        K3 = np.empty(shape+(4, 4))
        K3[..., 0, 0] = (rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2])/3.0
        K3[..., 0, 1] = (rot[..., 1, 0] + rot[..., 0, 1])/3.0
        K3[..., 0, 2] = (rot[..., 2, 0] + rot[..., 0, 2])/3.0
        K3[..., 0, 3] = (rot[..., 1, 2] - rot[..., 2, 1])/3.0
        K3[..., 1, 0] = K3[..., 0, 1]
        K3[..., 1, 1] = (rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2])/3.0
        K3[..., 1, 2] = (rot[..., 2, 1] + rot[..., 1, 2])/3.0
        K3[..., 1, 3] = (rot[..., 2, 0] - rot[..., 0, 2])/3.0
        K3[..., 2, 0] = K3[..., 0, 2]
        K3[..., 2, 1] = K3[..., 1, 2]
        K3[..., 2, 2] = (rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1])/3.0
        K3[..., 2, 3] = (rot[..., 0, 1] - rot[..., 1, 0])/3.0
        K3[..., 3, 0] = K3[..., 0, 3]
        K3[..., 3, 1] = K3[..., 1, 3]
        K3[..., 3, 2] = K3[..., 2, 3]
        K3[..., 3, 3] = (rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2])/3.0

        if not shape:
            q = zero.copy()
            eigvals, eigvecs = linalg.eigh(K3.T, eigvals=(3, 3))
            q.components[0] = eigvecs[-1]
            q.components[1:] = -eigvecs[:-1].flatten()
            return q
        else:
            q = np.empty(shape+(4,), dtype=np.float)
            for flat_index in range(reduce(mul, shape)):
                multi_index = np.unravel_index(flat_index, shape)
                eigvals, eigvecs = linalg.eigh(K3[multi_index], eigvals=(3, 3))
                q[multi_index, 0] = eigvecs[-1]
                q[multi_index, 1:] = -eigvecs[:-1].flatten()
            return as_quat_array(q)

    else:  # No scipy.linalg or not `accurate`
        if not shape:
            import math
            q = np.empty(shape+(4,), dtype=np.float)
            q[0] = (1 + rot[0, 0] + rot[1, 1] + rot[2, 2]) / 4
            if q[0] > _eps:
                q[0] = math.sqrt(q[0])
                q[1] = (rot[1, 2] - rot[2, 1]) / (-4*q[0])
                q[2] = (rot[2, 0] - rot[0, 2]) / (-4*q[0])
                q[3] = (rot[0, 1] - rot[1, 0]) / (-4*q[0])
            else:
                q[0] = 0.0
                q[1] = (rot[1, 1] + rot[2, 2]) / -2.0
                if q[1] > _eps:
                    q[1] = math.sqrt(q[1])
                    q[2] = rot[0, 1] / (2*q[1])
                    q[3] = rot[0, 2] / (2*q[1])
                else:
                    q[1] = 0.0
                    q[2] = (1.0 - rot[2, 2]) / 2.0
                    if q[2] > _eps:
                        q[2] = math.sqrt(q[2])
                        q[3] = rot[1, 2] / (2*q[2])
                    else:
                        q[2] = 0.0
                        q[3] = 1.0
        else:
            q = np.empty(shape+(4,), dtype=np.float)
            q[..., 0] = (1 + rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]) / 4
            w2_gtr_eps = (q[..., 0] > _eps)
            # w**2 > eps
            q[w2_gtr_eps, 0] = np.sqrt(q[w2_gtr_eps, 0])
            q[w2_gtr_eps, 1] = (rot[w2_gtr_eps, 1, 2] - rot[w2_gtr_eps, 2, 1]) / (-4*q[w2_gtr_eps, 0])
            q[w2_gtr_eps, 2] = (rot[w2_gtr_eps, 2, 0] - rot[w2_gtr_eps, 0, 2]) / (-4*q[w2_gtr_eps, 0])
            q[w2_gtr_eps, 3] = (rot[w2_gtr_eps, 0, 1] - rot[w2_gtr_eps, 1, 0]) / (-4*q[w2_gtr_eps, 0])
            # w**2 <= eps
            q[~w2_gtr_eps, 0] = 0.0
            q[~w2_gtr_eps, 1] = (rot[~w2_gtr_eps, 1, 1] + rot[~w2_gtr_eps, 2, 2]) / -2.0
            x2_gtr_eps = np.logical_and(~w2_gtr_eps, q[..., 1] > _eps)
            x2_leq_eps = np.logical_and(~w2_gtr_eps, q[..., 1] <= _eps)
            # w**2 <= eps and x**2 > eps
            q[x2_gtr_eps, 1] = np.sqrt(q[x2_gtr_eps, 1])
            q[x2_gtr_eps, 2] = rot[x2_gtr_eps, 0, 1] / (2*q[x2_gtr_eps, 1])
            q[x2_gtr_eps, 3] = rot[x2_gtr_eps, 0, 2] / (2*q[x2_gtr_eps, 1])
            # w**2 <= eps and x**2 <= eps
            q[x2_leq_eps, 1] = 0.0
            q[x2_leq_eps, 2] = (1.0 - rot[x2_leq_eps, 2, 2]) / 2.0
            y2_gtr_eps = np.logical_and(x2_leq_eps, q[..., 2] > _eps)
            y2_leq_eps = np.logical_and(x2_leq_eps, q[..., 2] <= _eps)
            # w**2 <= eps and x**2 <= eps and y**2 > eps
            q[y2_gtr_eps, 2] = np.sqrt(q[y2_gtr_eps, 2])
            q[y2_gtr_eps, 3] = rot[y2_gtr_eps, 1, 2] / (2*q[y2_gtr_eps, 2])
            # w**2 <= eps and x**2 <= eps and y**2 <= eps
            q[y2_leq_eps, 2] = 0.0
            q[y2_leq_eps, 3] = 1.0
        return as_quat_array(q)



def as_rotation_vector(q):
    """Convert input quaternion to the axis-angle representation

    Note that if any of the input quaternions has norm zero, no error is
    raised, but NaNs will appear in the output.

    Parameters
    ----------
    q: quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must all be nonzero

    Returns
    -------
    rot: float array
        Output shape is q.shape+(3,).  Each vector represents the axis of
        the rotation, with norm proportional to the angle of the rotation.

    """
    return as_float_array(2*np.log(np.normalized(q)))[..., 1:]


def from_rotation_vector(rot):
    """Convert input 3-vector in axis-angle representation to unit quaternion

    Parameters
    ----------
    rot: (Nx3) float array
        Each vector represents the axis of the rotation, with norm
        proportional to the angle of the rotation.

    Returns
    -------
    q: array of quaternions
        Unit quaternions resulting in rotations corresponding to input
        rotations.  Output shape is rot.shape[:-1].

    """
    rot = np.array(rot, copy=False)
    quats = np.zeros(rot.shape[:-1]+(4,))
    quats[..., 1:] = rot[...]/2
    quats = as_quat_array(quats)
    return np.exp(quats)


def as_euler_angles(q):
    """Open Pandora's Box

    If somebody is trying to make you use Euler angles, tell them no,
    and walk away, and go and tell your mum.

    You don't want to use Euler angles.  They are awful.  Stay away.
    It's one thing to convert from Euler angles to quaternions; at least
    you're moving in the right direction.  But to go the other way?!  It's
    just not right.

    Parameters
    ----------
    q: quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must all be nonzero

    Returns
    -------
    alpha_beta_gamma: float array
        Output shape is q.shape+(3,).  These represent the angles
        (alpha, beta, gamma), where the normalized input quaternion
        represents `exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)`.

    Raises
    ------
    AllHell
        If you try to actually use Euler angles, when you could have been
        using quaternions like a sensible person.

    """
    alpha_beta_gamma = np.empty(q.shape + (3,), dtype=np.float)
    n = np.norm(q)
    q = as_float_array(q)
    alpha_beta_gamma[..., 0] = np.arctan2(q[..., 3], q[..., 0]) + np.arctan2(-q[..., 1], q[..., 2])
    alpha_beta_gamma[..., 1] = 2*np.arccos(np.sqrt((q[..., 0]**2 + q[..., 3]**2)/n))
    alpha_beta_gamma[..., 2] = np.arctan2(q[..., 3], q[..., 0]) - np.arctan2(-q[..., 1], q[..., 2])
    return alpha_beta_gamma


def as_spherical_coords(q):
    """Return the spherical coordinates corresponding to this quaternion

    Obviously, spherical coordinates do not contain as much information as a quaternion, so this function does lose
    some information.  However, the returned spherical coordinates will represent the point(s) on the sphere to which
    the input quaternion(s) rotate the z axis.

    Parameters
    ----------
    q: quaternion or array of quaternions
        The quaternion(s) need not be normalized, but must be nonzero

    Returns
    -------
    vartheta_varphi: float array
        Output shape is q.shape+(2,).  These represent the angles
        (vartheta, varphi), where the normalized input quaternion
        represents `exp(varphi*z/2) * exp(vartheta*y/2)`, up to an
        arbitrary inital rotation about `z`.

    """
    return as_euler_angles(q)[..., 1::-1]


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
    if np.any(xinf) or np.any(yinf):
        # Check that x and y have inf's only in the same positions
        if not np.all(xinf == yinf):
            if verbose:
                print('not all(xinf == yinf)')
                equal = (xinf == yinf)
                for i, val in enumerate(equal.flatten()):
                    if not val:
                        print('\nx[{0}]={1}\ny[{0}]={2}'.format(i, x.flatten()[i], y.flatten()[i]))
            return False
        # Check that sign of inf's in x and y is the same
        if not np.all(x[xinf] == y[xinf]):
            if verbose:
                print('not all(x[xinf] == y[xinf])')
                equal = (x[xinf] == y[xinf])
                for i, val in enumerate(equal.flatten()):
                    if not val:
                        print('\nx[{0}]={1}\ny[{0}]={2}'.format(i, x[xinf].flatten()[i], y[xinf].flatten()[i]))
            return False

        x = x[~xinf]
        y = y[~xinf]

    # ignore invalid fpe's
    with np.errstate(invalid='ignore'):
        r = np.all(np.less_equal(abs(x - y), atol + rtol * abs(y)))
        if verbose and not r:
            lessequal = np.less_equal(abs(x - y), atol + rtol * abs(y))
            for i, val in enumerate(lessequal.flatten()):
                if not val:
                    print('\nx[{0}]={1}\ny[{0}]={2}'.format(i, x.flatten()[i], y.flatten()[i])
                          + '\n{0} > {1} + {2} * {3} = {4}'.format(abs(x.flatten()[i] - y.flatten()[i]),
                                                                   atol, rtol, abs(y.flatten()[i]),
                                                                   atol + rtol * abs(y.flatten()[i])))

    return r
