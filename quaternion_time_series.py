# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
from quaternion.numba_wrapper import njit


def slerp(R1, R2, t1, t2, t_out):
    """Spherical linear interpolation of rotors

    This function uses a simpler interface than the more fundamental
    `slerp_evaluate` and `slerp_vectorized` functions.  The latter
    are fast, being implemented at the C level, but take input `tau`
    instead of time.  This function adjusts the time accordingly.

    Parameters
    ----------
    R1: quaternion
        Quaternion at beginning of interpolation
    R2: quaternion
        Quaternion at end of interpolation
    t1: float
        Time corresponding to R1
    t2: float
        Time corresponding to R2
    t_out: float or array of floats
        Times to which the rotors should be interpolated


    """
    tau = (t_out-t1)/(t2-t1)
    return np.slerp_vectorized(R1, R2, tau)


def squad(R_in, t_in, t_out):
    """Spherical "quadrangular" interpolation of rotors with a cubic spline

    This is the best way to interpolate rotations.  It uses the analog
    of a cubic spline, except that the interpolant is confined to the
    rotor manifold in a natural way.  Alternative methods involving
    interpolation of other coordinates on the rotation group or
    normalization of interpolated values give bad results.  The
    results from this method are as natural as any, and are continuous
    in first and second derivatives.

    The input `R_in` rotors are assumed to be reasonably continuous
    (no sign flips), and the input `t` arrays are assumed to be
    sorted.  No checking is done for either case, and you may get
    silently bad results if these conditions are violated.

    This function simplifies the calling, compared to `squad_evaluate`
    (which takes a set of four quaternions forming the edges of the
    "quadrangle", and the normalized time `tau`) and `squad_vectorized`
    (which takes the same arguments, but in array form, and efficiently
    loops over them).

    Parameters
    ----------
    R_in: array of quaternions
        A time-series of rotors (unit quaternions) to be interpolated
    t_in: array of float
        The times corresponding to R_in
    t_out: array of float
        The times to which R_in should be interpolated

    """
    if R_in.size == 0 or t_out.size == 0:
        return np.array((), dtype=np.quaternion)

    # This list contains an index for each `t_out` such that
    # t_in[i-1] <= t_out < t_in[i]
    # Note that `side='right'` is much faster in my tests
    i_in_for_out = t_in.searchsorted(t_out, side='left')
    np.clip(i_in_for_out, 0, len(t_in) - 1, out=i_in_for_out)
    i_in = np.unique(i_in_for_out)

    # Now, for each index `i` in `i_in`, we need to compute the
    # interpolation "coefficients" (`A_i`, `B_ip1`).
    #
    # I previously tested an explicit version of the loops below,
    # comparing `stride_tricks.as_strided` with explicit
    # implementation via `roll` (as seen here).  I found that the
    # `roll` was significantly more efficient for simple calculations,
    # though the difference is probably totally washed out here.  In
    # any case, it might be useful to test again.
    #
    A = R_in * np.exp((- np.log((~R_in) * np.roll(R_in, -1))
                       + np.log((~np.roll(R_in, 1)) * R_in) * ((np.roll(t_in, -1) - t_in) / (t_in - np.roll(t_in, 1)))
                       ) * 0.25)
    B = np.roll(R_in, -1) * np.exp((np.log((~np.roll(R_in, -1)) * np.roll(R_in, -2))
                                    * ((np.roll(t_in, -1) - t_in) / (np.roll(t_in, -2) - np.roll(t_in, -1)))
                                    - np.log((~R_in) * np.roll(R_in, -1))) * -0.25)

    # Correct the first and last A time steps, and last two B time steps
    A[0] = R_in[0] * np.exp((np.log((~R_in[0])*R_in[1])
                             + np.log((~(R_in[0]*(~R_in[1])*R_in[0]))*R_in[0])
                             - 2 * np.log((~R_in[0]) * R_in[1])
                             ) * 0.25)
    A[-1] = R_in[-1]
    B[-2] = R_in[-1]
    #B[-1] = quaternion.one

    # Use the coefficients at the corresponding t_out indices to
    # compute the squad interpolant
    R_ip1 = np.array(np.roll(R_in, -1)[i_in_for_out])
    R_ip1[-1] = R_ip1[-2]*(~R_ip1[-3])*R_ip1[-2]
    tau = (t_out - t_in[i_in_for_out]) / ((np.roll(t_in, -1) - t_in)[i_in_for_out])
    R_out = np.squad_vectorized(tau, R_in[i_in_for_out], A[i_in_for_out], B[i_in_for_out], R_ip1)

    return R_out


@njit
def frame_from_angular_velocity_integrand(rfrak, Omega):
    import math
    from numpy import dot, cross
    from .numpy_quaternion import _eps
    rfrakMag = math.sqrt(rfrak[0] * rfrak[0] + rfrak[1] * rfrak[1] + rfrak[2] * rfrak[2])
    OmegaMag = math.sqrt(Omega[0] * Omega[0] + Omega[1] * Omega[1] + Omega[2] * Omega[2])
    # If the matrix is really close to the identity, return
    if rfrakMag < _eps * OmegaMag:
        return Omega[0] / 2.0, Omega[1] / 2.0, Omega[2] / 2.0
    # If the matrix is really close to singular, it's equivalent to the identity, so return
    if abs(math.sin(rfrakMag)) < _eps:
        return Omega[0] / 2.0, Omega[1] / 2.0, Omega[2] / 2.0

    OmegaOver2 = Omega[0] / 2.0, Omega[1] / 2.0, Omega[2] / 2.0
    rfrakHat = rfrak[0] / rfrakMag, rfrak[1] / rfrakMag, rfrak[2] / rfrakMag

    return ((OmegaOver2 - rfrakHat * dot(rfrakHat, OmegaOver2)) * (rfrakMag / math.tan(rfrakMag))
            + rfrakHat * dot(rfrakHat, OmegaOver2) + cross(OmegaOver2, rfrak))