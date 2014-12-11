#!/usr/bin/env python

# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    from distutils.errors import DistutilsError
    if numpy.__dict__.get('quaternion') is not None:
        raise DistutilsError('The target NumPy already has a quaternion type')
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    # if(os.environ.get('THIS_IS_TRAVIS') is not None):
    #     print("This appears to be Travis!")
    #     compile_args = ['-O3']
    # else:
    #     compile_args = ['-ffast-math', '-O3']
    compile_args = ['-O3']
    config = Configuration('quaternion',parent_package,top_path)
    config.add_extension('numpy_quaternion',
                         ['quaternion.c','numpy_quaternion.c'],
                         extra_compile_args=compile_args,)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
