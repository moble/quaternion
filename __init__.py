import numpy as np
from .numpy_quaternion import quaternion

__doc_title__ = "Quaternion dtype for NumPy"
__doc__ = "Adds a quaternion dtype to NumPy."

__all__ = ['quaternion']

if np.__dict__.get('quaternion') is not None:
    raise RuntimeError('The NumPy package already has a quaternion type')

np.quaternion = quaternion
np.typeDict['quaternion'] = np.dtype(quaternion)
