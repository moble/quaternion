#!/usr/bin/env python

"""\
This package creates a quaternion type in python, and further enables numpy to
create and manipulate arrays of quaternions.  The usual algebraic operations
(addition and multiplication) are available, along with numerous properties
like norm and various types of distance measures between two quaternions.
There are also additional functions like "squad" and "slerp" interpolation, and
conversions to and from axis-angle, matrix, and Euler-angle representations of
rotations.  The core of the code is written in C for speed.
"""

from sys import platform
from setuptools import Extension, setup
import numpy as np


# Set this first for easier replacement
version = "2021.8.26.15.40.13"

if "win" in platform.lower() and not "darwin" in platform.lower():
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-w"]

extensions = [
    Extension(
        name="quaternion.numpy_quaternion",  # This is the name of the object file that will be compiled
        sources=[
            "src/quaternion.c",
            "src/numpy_quaternion.c"
        ],
        depends=[
            "src/quaternion.c",
            "src/quaternion.h",
            "src/numpy_quaternion.c"
        ],
        include_dirs=[
            np.get_include(),
            "src"
        ],
        extra_compile_args=extra_compile_args,
    ),
]

setup_metadata = dict(
    name="numpy-quaternion",  # Uploaded to pypi under this name
    packages=["quaternion"],  # This is the actual package name, as used in python
    package_dir = {'': 'src'},  # Remove `src/` from the package name
    url="https://github.com/moble/quaternion",
    author="Michael Boyle",
    author_email="mob22@cornell.edu",
    description="Add a quaternion dtype to NumPy",
    long_description=__doc__,
    ext_modules=extensions,
    install_requires=[
        "numpy>=1.13",
        # See also extras and :environment_marker specs below
    ],
    extras_require={
        "scipy": [
            "scipy",
        ],
        "numba:python_version < '3.6' and platform_python_implementation != 'PyPy'": [
            "numba<0.49.0",
            "llvmlite<0.32.0",
        ],
        "numba:python_version >= '3.6' and platform_python_implementation != 'PyPy'": [
            "numba",
        ],
        "docs":  [
            "mkdocs",
            "mktheapidocs[plugin]",
            "pymdown-extensions",
        ],
        "testing": [
            "pytest",
            "pytest-cov",
        ]
    },
    version=version,
)


def build(setup_kwargs):
    # For possible poetry support
    setup_kwargs.update({"ext_modules": extensions})


if __name__ == "__main__":
    setup(**setup_metadata)
