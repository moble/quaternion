#!/usr/bin/env python

# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from auto_version import calculate_version, build_py_copy_version


def configuration(parent_package='', top_path=None):
    import numpy
    from distutils.errors import DistutilsError

    if numpy.__dict__.get('quaternion') is not None:
        raise DistutilsError('The target NumPy already has a quaternion type')
    from numpy.distutils.misc_util import Configuration
    compile_args = ['-ffast-math', '-O3']
    config = Configuration('quaternion', parent_package, top_path)
    config.add_extension('numpy_quaternion',
                         ['quaternion.c', 'numpy_quaternion.c'],
                         depends=['quaternion.c', 'quaternion.h', 'numpy_quaternion.c'],
                         extra_compile_args=compile_args, )
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='numpy-',
          configuration=configuration,
          version=calculate_version(),
          cmdclass={'build_py': build_py_copy_version},
          url='https://github.com/moble/quaternion',
          author='Michael Boyle',
          author_email='',
    )
