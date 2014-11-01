from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion

## Allow the code to function without numba, but discourage it
## strongly.
try:
    from numbapro import njit, jit
except ImportError:
    try:
        from numba import njit, jit
    except ImportError:
        import warnings
        warning_text = \
            "\n\n" + "!"*53 + "\n" + \
            "Could not import from either numbapro or numba.\n" + \
            "This means that the code will run MUCH more slowly.\n" + \
            "You probably REALLY want to install numba / numbapro." + \
            "\n" + "!"*53 + "\n"
        warnings.warn(warning_text)
        def _identity_decorator_outer(*args, **kwargs):
            def _identity_decorator_inner(fn):
                return fn
            return _identity_decorator_inner
        njit = _identity_decorator_outer
        jit = _identity_decorator_outer

@njit
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
    i_in_for_out = t_in.searchsorted(t_out, side='right')
    np.clip(i_in_for_out, 0, len(t_in)-1, out=i_in_for_out)
    i_in = np.unique(i_in_for_out)

    # Now, for each index `i` in `i_in`, we need to compute the
    # interpolation "coefficients" (`A_i`, `B_ip1`).  Note that my
    # tests show that this ugly, obvious zip/roll technique is
    # actually about 20% faster (for simple calculations) than
    # "clever" things involving stride_tricks.as_strided.  Most of
    # that is probably washed out by the cost of the particularly
    # complicated calculations done in this loop, so I'm just saying
    # it's not worth changing the iteration.

    # AB = np.array([
    #     [R_i * (0.25*( (R_i.inverse()*R_ip1).log() +((t_ip1-t_i)/(t_i-t_im1))*(R_im1.inverse()*R_i).log() - 2*(R_i.inverse()*R_ip1).log() )).exp(),
    #      R_ip1 * (-0.25*( ((t_ip1-t_i)/(t_ip2-t_ip1))*(R_ip1.inverse()*R_ip2).log() + (R_i.inverse()*R_ip1).log() - 2*(R_i.inverse()*R_ip1).log() )).exp()]
    #     for t_im1,t_i,t_ip1,t_ip2,R_im1,R_i,R_ip1,R_ip2 in zip(np.roll(t_in,1),t_in,np.roll(t_in,-1),np.roll(t_in,-2),
    #                                                            np.roll(R_in,1),R_in,np.roll(R_in,-1),np.roll(R_in,-2))])

    AB = np.array(
        [R_in * exp(( log((~R_in)*np.roll(R_in,-1))
                      + np.log((~np.roll(R_in,1))*R_in)*((np.roll(t_in,-1)-t_in)/(t_in-np.roll(t_in,1)))
                      - log(R_in.inverse()*np.roll(R_in,-1))*2 )*0.25),
         np.roll(R_in,-1) * exp(( log((~np.roll(R_in,-1))*np.roll(R_in,-2))*((np.roll(t_in,-1)-t_in)/(np.roll(t_in,-2)-np.roll(t_in,-1)))
                                  + log((~R_in)*np.roll(R_in,-1))
                                  - log((~R_in)*np.roll(R_in,-1))*2 )*-0.25)])


    # Finally, we use the coefficients at the corresponding t_out
    # indices to compute the squad interpolant

    

    while(iOut<tOut.size() && tOut[iOut]<=tIn[iIn+1]):
        taui = (tOut[iOut]-tIn[iIn]) / Dti;
        ROut[iOut] = quaternion.slerp(2*taui*(1-taui),
                                      quaternion.slerp(taui, Qi, Qip1),
                                      quaternion.slerp(taui, Ai, Bip1))
        iOut += 1;

    return R_out
