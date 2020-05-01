#!/usr/bin/env python

# Copyright (c) 2018, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

# Construct the version number from the date and time this python version was created.
from os import environ
from sys import platform
from warnings import warn
version = None
on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())
if "package_version" in environ:
    version = environ["package_version"]
    print("Setup.py using environment version='{0}'".format(version))
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        # For cases where this is being installed from git.  This gives the true version number.
        from subprocess import check_output
        if on_windows:
            version = check_output("""git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""", shell=False)
            version = version.decode('ascii').strip().replace('.0', '.').replace("'", "")
        else:
            try:
                from subprocess import DEVNULL as devnull
                version = check_output("""git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=True, stderr=devnull)
            except AttributeError:
                from os import devnull
                version = check_output("""git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=True, stderr=devnull)
            version = version.decode('ascii').rstrip()
        print("Setup.py using git log version='{0}'".format(version))
    except:
        pass
if version is not None:
    with open('_version.py', 'w') as f:
        f.write('__version__ = "{0}"'.format(version))
else:
    try:
        from ast import literal_eval
        with open('_version.py', 'r') as f:
            first_line = f.readline()
        version_string = first_line.split('=')[1].strip()
        version = literal_eval(version_string)
    except:
        pass




long_description = """\
This package creates a quaternion type in python, and further enables numpy to create and manipulate arrays of
quaternions.  The usual algebraic operations (addition and multiplication) are available, along with numerous
properties like norm and various types of distance measures between two quaternions.  There are also
additional functions like "squad" and "slerp" interpolation, and conversions to and from axis-angle, matrix,
and Euler-angle representations of rotations.  The core of the code is written in C for speed.
"""


if __name__ == "__main__":
    # Note: Because pip may try to install this package before numpy (on which it depends, and which
    # it needs to use *during* setup), we need to try to fail gracefully when numpy is not
    # installed.  The following mostly follows the strategy found in scipy's setup.py script, here:
    # https://github.com/scipy/scipy/blob/9ccc68475fc431c4a44c120693cf6878cc4c14a7/setup.py#L180
    
    import sys

    setup_metadata = dict(
        name='numpy-quaternion',  # Uploaded to pypi under this name
        packages=['quaternion'],  # This is the actual package name, as used in python
        package_dir={'quaternion': ''},
        url='https://github.com/moble/quaternion',
        author='Michael Boyle',
        author_email='mob22@cornell.edu',
        description='Add built-in support for quaternions to numpy',
        long_description=long_description,
    )

    if version is not None:
        setup_metadata['version'] = version

    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean')):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scipy when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
            setup_metadata['install_requires'] = ['numpy>=1.13',]
            setup_metadata['setup_requires'] = ['setuptools', 'wheel', 'numpy>=1.13',]
        except ImportError:
            from distutils.core import setup
    else:
        from setuptools import setup, Extension
        from setuptools.command.build_ext import build_ext as _build_ext
        setup_metadata['install_requires'] = ['numpy>=1.13',]
        setup_metadata['setup_requires'] = ['setuptools', 'wheel', 'numpy>=1.13',]
        extension = Extension(
            name='quaternion.numpy_quaternion',  # This is the name of the object file that will be compiled
            sources=['quaternion.c', 'numpy_quaternion.c'],
            extra_compile_args=['/O2' if on_windows else '-O3'],
            depends=['quaternion.c', 'quaternion.h', 'numpy_quaternion.c'],
        )
        setup_metadata['ext_modules'] = [extension]
        class build_ext(_build_ext):
            # This addition was suggested in https://stackoverflow.com/a/21621689/1194883
            def finalize_options(self):
                _build_ext.finalize_options(self)
                # Prevent numpy from thinking it is still in its setup process:
                try:
                    __builtins__.__NUMPY_SETUP__ = False
                except:
                    try:
                        # For python 3
                        import builtins
                        builtins.__NUMPY_SETUP__ = False
                    except:
                        warn("Skipping numpy hack; if installation fails, try installing numpy first")
                import numpy
                self.include_dirs.append(numpy.get_include())
                if numpy.__dict__.get('quaternion') is not None:
                    from distutils.errors import DistutilsError
                    raise DistutilsError('The target NumPy already has a quaternion type')
        setup_metadata['cmdclass'] = {'build_ext': build_ext}

    setup(**setup_metadata)
