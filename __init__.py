from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from .numpy_quaternion import quaternion

__doc_title__ = "Quaternion dtype for NumPy"
__doc__ = "Adds a quaternion dtype to NumPy."

__all__ = ['quaternion']

if 'quaternion' in np.__dict__:
    raise RuntimeError('The NumPy package already has a quaternion type')

np.quaternion = quaternion
np.typeDict['quaternion'] = np.dtype(quaternion)

def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return a.view(np.float).reshape(a.shape+(4,))
def as_quat_array(a):
    """View a float array as an array of quaternions

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The input array must have a final dimension whose size is
    divisible by four (or better yet *is* 4).

    """
    if(a.shape[-1]==4) :
        return a.view(np.quaternion).reshape(a.shape[:-1])
    return a.view(np.quaternion).reshape(a.shape[:-1]+(a.shape[-1]//4,))
