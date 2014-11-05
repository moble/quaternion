from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
from .numba import njit, xrange

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

    """

    # This list contains an index for each `t_out` such that
    #   t_in[i-1] <= t_out < t_in[i]
    # Note that `side='right'` is much faster in my tests
    i_in_for_out = t_in.searchsorted(t_out, side='left')
    np.clip(i_in_for_out, 0, len(t_in)-1, out=i_in_for_out)
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
    A = R_in * np.exp(( np.log((~R_in)*np.roll(R_in,-1))
                        + np.log((~np.roll(R_in,1))*R_in)*((np.roll(t_in,-1)-t_in)/(t_in-np.roll(t_in,1)))
                        - np.log((~R_in)*np.roll(R_in,-1))*2 )*0.25)
    B = np.roll(R_in,-1) * np.exp(( np.log((~np.roll(R_in,-1))*np.roll(R_in,-2))
                                    *((np.roll(t_in,-1)-t_in)/(np.roll(t_in,-2)-np.roll(t_in,-1)))
                                    + np.log((~R_in)*np.roll(R_in,-1))
                                    - np.log((~R_in)*np.roll(R_in,-1))*2 )*-0.25)

    # Use the coefficients at the corresponding t_out indices to
    # compute the squad interpolant
    tau = (t_out-t_in[i_in_for_out]) / ((np.roll(t_in,-1)-t_in)[i_in_for_out])
    R_out = quaternion.squad_loop(tau, R_in[i_in_for_out], A[i_in_for_out], B[i_in_for_out], np.roll(R_in,-1)[i_in_for_out])

    # Correct the first one and last two time steps


    return R_out


