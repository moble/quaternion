from info import __doc__

__all__ = ['quaternion']

import numpy
from numpy_quaternion import quaternion

if numpy.__dict__.get('quaternion') is not None:
    raise RuntimeError('The NumPy package already has a quaternion type')

numpy.quaternion = quaternion
numpy.typeDict['quaternion'] = numpy.dtype(quaternion)
