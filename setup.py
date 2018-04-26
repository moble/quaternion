#!/usr/bin/env python

# Copyright (c) 2018, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

# Construct the version number from the date and time this python version was created.
from os import environ
if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        # For cases where this is being installed from git.  This gives the true version number.
        from sys import platform
        from subprocess import check_output
        on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())
        use_shell = not on_windows
        version = check_output("""git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=use_shell).decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except:
        # For cases where this isn't being installed from git.  This gives the wrong version number,
        # but at least it provides some information.
        try:
            from time import strftime, gmtime
            try:
                version = strftime("%Y.%-m.%-d.%-H.%-M.%-S", gmtime())
            except ValueError:  # because Windows
                version = strftime("%Y.%m.%d.%H.%M.%S", gmtime())
            print("Setup.py using strftime version='{0}'".format(version))
        except:
            version = '0.0.0'
            print("Setup.py failed to determine the version; using '{0}'".format(version))
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


long_description = """\
This package creates a quaternion type in python, and further enables numpy to create and manipulate arrays of
quaternions.  The usual algebraic operations (addition and multiplication) are available, along with numerous
properties like norm and various types of distance measures between two quaternions.  There are also
additional functions like "squad" and "slerp" interpolation, and conversions to and from axis-angle, matrix,
and Euler-angle representations of rotations.  The core of the code is written in C for speed.
"""


if __name__ == "__main__":
    from os import getenv
    from numpy.distutils.core import setup
    setup(name='numpy-',
          configuration=configuration,
          version=version,
          url='https://github.com/moble/quaternion',
          author='Michael Boyle',
          author_email='mob22@cornell.edu',
          description='Add built-in support for quaternions to numpy',
          long_description=long_description,
    )
