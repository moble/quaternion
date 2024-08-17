#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import platform
from setuptools import Extension, setup
from pathlib import Path
import numpy as np


# Set this first for easier replacement
version = "2023.1.0"

# read the contents of the README file into the PyPI description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Set appropriate optimization flags
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
            "src/numpy_quaternion.c",
            "src/npy_2_compat.h"
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=extensions,
    # install_requires=[
    #     "numpy>=1.13",
    #     # See also extras and :environment_marker specs below
    # ],
    # extras_require={
    #     "scipy": [
    #         "scipy",
    #     ],
    #     "numba:python_version < '3.6' and platform_python_implementation != 'PyPy'": [
    #         "numba<0.49.0",
    #         "llvmlite<0.32.0",
    #     ],
    #     "numba:python_version >= '3.6' and platform_python_implementation != 'PyPy'": [
    #         "numba",
    #     ],
    #     "docs":  [
    #         "mkdocs",
    #         "mktheapidocs[plugin]",
    #         "pymdown-extensions",
    #     ],
    #     "testing": [
    #         "pytest",
    #         "pytest-cov",
    #     ]
    # },
    version=version,
)


def build(setup_kwargs):
    # For possible poetry support
    setup_kwargs.update({"ext_modules": extensions})


if __name__ == "__main__":
    setup(**setup_metadata)
