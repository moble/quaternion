# Copyright (c) 2017, Michael Boyle
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
    # i_in_for_out = t_in.searchsorted(t_out, side='left')
    # np.clip(i_in_for_out, 0, len(t_in) - 1, out=i_in_for_out)
    i_in_for_out = t_in.searchsorted(t_out, side='right')-1

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

    # Correct the first and last A time steps, and last two B time steps.  We extend R_in with the following wrap-around
    # values:
    # R_in[0-1] = R_in[0]*(~R_in[1])*R_in[0]
    # R_in[n+0] = R_in[-1] * (~R_in[-2]) * R_in[-1]
    # R_in[n+1] = R_in[0] * (~R_in[-1]) * R_in[0]
    #           = R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1]
    #           = R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-2]) * R_in[-1]
    # A[i] = R_in[i] * np.exp((- np.log((~R_in[i]) * R_in[i+1])
    #                          + np.log((~R_in[i-1]) * R_in[i]) * ((t_in[i+1] - t_in[i]) / (t_in[i] - t_in[i-1]))
    #                          ) * 0.25)
    # A[0] = R_in[0] * np.exp((- np.log((~R_in[0]) * R_in[1]) + np.log((~R_in[0])*R_in[1]*(~R_in[0])) * R_in[0]) * 0.25)
    #      = R_in[0]
    A[0] = R_in[0]
    # A[-1] = R_in[-1] * np.exp((- np.log((~R_in[-1]) * R_in[n+0])
    #                          + np.log((~R_in[-2]) * R_in[-1]) * ((t_in[n+0] - t_in[-1]) / (t_in[-1] - t_in[-2]))
    #                          ) * 0.25)
    #       = R_in[-1] * np.exp((- np.log((~R_in[-1]) * R_in[n+0]) + np.log((~R_in[-2]) * R_in[-1])) * 0.25)
    #       = R_in[-1] * np.exp((- np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                           + np.log((~R_in[-2]) * R_in[-1])) * 0.25)
    #       = R_in[-1] * np.exp((- np.log((~R_in[-2]) * R_in[-1]) + np.log((~R_in[-2]) * R_in[-1])) * 0.25)
    #       = R_in[-1]
    A[-1] = R_in[-1]
    # B[i] = R_in[i+1] * np.exp((np.log((~R_in[i+1]) * R_in[i+2]) * ((t_in[i+1] - t_in[i]) / (t_in[i+2] - t_in[i+1]))
    #                            - np.log((~R_in[i]) * R_in[i+1])) * -0.25)
    # B[-2] = R_in[-1] * np.exp((np.log((~R_in[-1]) * R_in[0]) * ((t_in[-1] - t_in[-2]) / (t_in[0] - t_in[-1]))
    #                            - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * np.exp((np.log((~R_in[-1]) * R_in[0]) - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * np.exp((np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                            - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * np.exp((np.log((~R_in[-2]) * R_in[-1]) - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1]
    B[-2] = R_in[-1]
    # B[-1] = R_in[0]
    # B[-1] = R_in[0] * np.exp((np.log((~R_in[0]) * R_in[1]) - np.log((~R_in[-1]) * R_in[0])) * -0.25)
    #       = R_in[-1] * (~R_in[-2]) * R_in[-1]
    #         * np.exp((np.log((~(R_in[-1] * (~R_in[-2]) * R_in[-1])) * R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                  - np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * (~R_in[-2]) * R_in[-1]
    #         * np.exp((np.log(((~R_in[-1]) * R_in[-2] * (~R_in[-1])) * R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                  - np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])) * -0.25)
    #         * np.exp((np.log((~R_in[-2]) * R_in[-1])
    #                  - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    B[-1] = R_in[-1] * (~R_in[-2]) * R_in[-1]

    # Use the coefficients at the corresponding t_out indices to
    # compute the squad interpolant
    # R_ip1 = np.array(np.roll(R_in, -1)[i_in_for_out])
    # R_ip1[-1] = R_in[-1]*(~R_in[-2])*R_in[-1]
    R_ip1 = np.roll(R_in, -1)
    R_ip1[-1] = R_in[-1]*(~R_in[-2])*R_in[-1]
    R_ip1 = np.array(R_ip1[i_in_for_out])
    t_inp1 = np.roll(t_in, -1)
    t_inp1[-1] = t_in[-1] + (t_in[-1] - t_in[-2])
    tau = (t_out - t_in[i_in_for_out]) / ((t_inp1 - t_in)[i_in_for_out])
    # tau = (t_out - t_in[i_in_for_out]) / ((np.roll(t_in, -1) - t_in)[i_in_for_out])
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


class appending_array(object):
    def __init__(self, shape, dtype=np.float, initial_array=None):
        shape = list(shape)
        if shape[0] < 4:
            shape[0] = 4
        self._a = np.empty(shape, dtype=dtype)
        self.n = 0
        if initial_array is not None:
            assert initial_array.dtype == dtype
            assert initial_array.shape[1:] == shape[1:]
            assert initial_array.shape[0] <= shape[0]
            self.n = initial_array.shape[0]
            self._a[:self.n, ...] = initial_array[:]

    def append(self, row):
        self.n += 1
        if self.n > self._a.shape[0]:
            self._a = np.resize(self._a, (2*self._a.shape[0],)+self._a.shape[1:])
        self._a[self.n-1, ...] = row

    @property
    def a(self):
        return self._a[:self.n, ...]


def integrate_angular_velocity(Omega, t0, t1, R0=None, tolerance=1e-12):
    """Compute frame with given angular velocity

    Parameters
    ==========
    Omega: tuple or callable
        Angular velocity from which to compute frame.  Can be
          1) a 2-tuple of float arrays (t, v) giving the angular velocity vector at a series of times,
          2) a function of time that returns the 3-vector angular velocity, or
          3) a function of time and orientation (t, R) that returns the 3-vector angular velocity
        In case 1, the angular velocity will be interpolated to the required times.  Note that accuracy
        is poor in case 1.
    t0: float
        Initial time
    t1: float
        Final time
    R0: quaternion, optional
        Initial frame orientation.  Defaults to 1 (the identity orientation).
    tolerance: float, optional
        Absolute tolerance used in integration.  Defaults to 1e-12.

    Returns
    =======
    t: float array
    R: quaternion array

    """
    import warnings
    from scipy.integrate import ode

    if R0 is None:
        R0 = quaternion.one

    try:
        t_Omega, v = Omega
        from scipy.interpolate import InterpolatedUnivariateSpline
        Omega_x = InterpolatedUnivariateSpline(t_Omega, v[:, 0])
        Omega_y = InterpolatedUnivariateSpline(t_Omega, v[:, 1])
        Omega_z = InterpolatedUnivariateSpline(t_Omega, v[:, 2])
        def Omega_func(t, R):
            return [Omega_x(t), Omega_y(t), Omega_z(t)]
        Omega_func(t0, R0)
    except (TypeError, ValueError):
        def Omega_func(t, R):
            return Omega(t, R)
        try:
            Omega_func(t0, R0)
        except TypeError:
            def Omega_func(t, R):
                return Omega(t)
            Omega_func(t0, R0)

    def RHS(t, y):
        R = quaternion.quaternion(*y)
        return (0.5 * quaternion.quaternion(0.0, *Omega_func(t, R)) * R).components

    y0 = R0.components

    solver = ode(RHS)
    solver.set_integrator('dop853', nsteps=1, atol=tolerance, rtol=0.0)
    solver.set_initial_value(y0, t0)
    solver._integrator.iwork[2] = -1  # suppress Fortran-printed warning

    t = appending_array((int(t1-t0),))
    t.append(solver.t)
    R = appending_array((int(t1-t0), 4))
    R.append(solver.y)

    warnings.filterwarnings("ignore", category=UserWarning)
    t_last = solver.t
    while solver.t < t1:
        solver.integrate(t1, step=True)
        if solver.t > t_last:
            t.append(solver.t)
            R.append(solver.y)
            t_last = solver.t
    warnings.resetwarnings()

    t = t.a
    R = quaternion.as_quat_array(R.a)

    return t, R
