#!/usr/bin/env python

# Copyright (c) 2017, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

# Construct the version number from the date and time this python version was created.
from os import environ
if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        from subprocess import check_output
        version = check_output("""git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""", shell=use_shell).decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except:
        from time import strftime, gmtime
        version = strftime("%Y.%m.%d.%H.%M.%S", gmtime())
        print("Setup.py using strftime version='{0}'".format(version))
with open('_version.py', 'w') as f:
    f.write('__version__ = "{0}"'.format(version))


def configuration(parent_package='', top_path=None):
    import numpy
    from distutils.errors import DistutilsError
    if numpy.__dict__.get('quaternion') is not None:
        raise DistutilsError('The target NumPy already has a quaternion type')
    from numpy.distutils.misc_util import Configuration
    compile_args = ['-O3']
    config = Configuration('quaternion', parent_package, top_path)
    config.add_extension('numpy_quaternion',
                         ['quaternion.c', 'numpy_quaternion.c'],
                         depends=['quaternion.c', 'quaternion.h', 'numpy_quaternion.c'],
                         extra_compile_args=compile_args, )
    return config


if __name__ == "__main__":
    from os import getenv
    from numpy.distutils.core import setup
    setup(name='numpy-',
          configuration=configuration,
          version=version,
          url='https://github.com/moble/quaternion',
          author='Michael Boyle',
          author_email='mob22@cornell.edu',
    )
