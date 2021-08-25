# Copyright (c) 2017, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/main/LICENSE>

from __future__ import division, print_function, absolute_import



def mean_rotor_in_chordal_metric(R, t=None):
    """Return rotor that is closest to all R in the least-squares sense

    This can be done (quasi-)analytically because of the simplicity of
    the chordal metric function.  It is assumed that the input R values
    all are normalized (or at least have the same norm).

    Note that the `t` argument is optional.  If it is present, the
    times are used to weight the corresponding integral.  If it is not
    present, a simple sum is used instead (which may be slightly
    faster).  However, because a spline is used to do this integral,
    the number of input points must be at least 4 (one more than the
    degree of the spline).

    """
    import numpy as np
    from . import as_float_array
    from .calculus import definite_integral
    if t is None:
        return np.sum(R).normalized()
    if len(t) < 4 or len(R) < 4:
        raise ValueError('Input arguments must have length greater than 3; their lengths are {0} and {1}.'.format(len(R), len(t)))
    mean = definite_integral(as_float_array(R), t)
    return np.quaternion(*mean).normalized()


def optimal_alignment_in_chordal_metric(Ra, Rb, t=None):
    """Return Rd such that Rd*Rb is as close to Ra as possible

    This function simply encapsulates the mean rotor of Ra/Rb.

    As in the `mean_rotor_in_chordal_metric` function, the `t` argument is
    optional.  If it is present, the times are used to weight the corresponding
    integral.  If it is not present, a simple sum is used instead (which may be
    slightly faster).

    Notes
    =====
    The idea here is to find Rd such that

        ∫ |Rd*Rb - Ra|^2 dt

    is minimized.  [Note that the integrand is the distance in the chordal metric.]
    We can ensure that this quantity is minimized by multiplying Rd by an
    exponential, differentiating with respect to the argument of the exponential,
    and setting that argument to 0.  This derivative should be 0 at the minimum.
    We have

        ∂ᵢ ∫ |exp[vᵢ]*Rd*Rb-Ra|^2 dt  →  2 ⟨ eᵢ * Rd * ∫ Rb*R̄a dt ⟩₀

    where → denotes taking vᵢ→0, the symbol ⟨⟩₀ denotes taking the scalar part, and
    eᵢ is the unit quaternionic vector in the `i` direction.  The only way for this
    quantity to be zero for each choice of `i` is if

        Rd * ∫ Rb*R̄a dt

    is itself a pure scalar.  This, in turn, can only happen if either (1) the
    integral is 0 or (2) if Rd is proportional to the conjugate of the integral:

        Rd ∝ ∫ Ra*R̄b dt

    Now, since we want Rd to be a rotor, we simply define it to be the normalized
    integral.

    """
    return mean_rotor_in_chordal_metric(Ra / Rb, t)


def mean_rotor_in_intrinsic_metric(R, t=None):
    raise NotImplementedError()
