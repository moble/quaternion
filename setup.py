#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    import numpy
    from distutils.errors import DistutilsError
    if numpy.__dict__.get('quaternion') is not None:
        raise DistutilsError('The target NumPy already has a quaternion type')
    from numpy.distutils.misc_util import Configuration
    config = Configuration('quaternion',parent_package,top_path)
    config.add_extension('numpy_quaternion',['quaternion.h','quaternion.c','numpy_quaternion.c'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
