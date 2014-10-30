from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from .numpy_quaternion import (quaternion, from_spherical_coords, from_euler_angles,
                               rotor_intrinsic_distance, rotor_chordal_distance,
                               rotation_intrinsic_distance, rotation_chordal_distance,
                               slerp)

__doc_title__ = "Quaternion dtype for NumPy"
__doc__ = "Adds a quaternion dtype to NumPy."

__all__ = ['quaternion', 'from_spherical_coords', 'from_euler_angles',
           'one', 'x', 'y', 'z']

if 'quaternion' in np.__dict__:
    raise RuntimeError('The NumPy package already has a quaternion type')

np.quaternion = quaternion
np.typeDict['quaternion'] = np.dtype(quaternion)

one = np.quaternion(1,0,0,0)
x = np.quaternion(0,1,0,0)
y = np.quaternion(0,0,1,0)
z = np.quaternion(0,0,0,1)


def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    assert a.dtype == np.dtype(np.quaternion)
    av = a.view(np.float)
    av = av.reshape(a.shape+(4,))
    return av
    # return a.view(np.float).reshape(a.shape+(4,))
def as_quat_array(a):
    """View a float array as an array of quaternions

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The input array must have a final dimension whose size is
    divisible by four (or better yet *is* 4).

    """
    assert a.dtype == np.dtype(np.float)
    av = a.view(np.quaternion)
    if(a.shape[-1]==4) :
        av = av.reshape(a.shape[:-1])
        # return a.view(np.quaternion).reshape(a.shape[:-1])
    else :
        av = av.reshape(a.shape[:-1]+(a.shape[-1]//4,))
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
    return a.view(np.float).reshape(a.shape+(4,))[...,[0,3,2,1]].ravel().view(np.complex).reshape(a.shape+(2,))
