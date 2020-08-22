from distutils.core import Extension
from os.path import join
from sys import platform
import numpy as np

on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())

extensions = [
    Extension(
        name='quaternion.numpy_quaternion',  # This is the name of the object file that will be compiled
        sources=[
            'quaternion/quaternion.c',
            'quaternion/numpy_quaternion.c'
        ],
        depends=[
            'quaternion/quaternion.c',
            'quaternion/quaternion.h',
            'quaternion/numpy_quaternion.c'
        ],
        include_dirs=[np.get_include()],
        extra_compile_args=['/O2' if on_windows else '-O3'],
    )
]


def build(setup_kwargs):
    setup_kwargs.update({'ext_modules': extensions})
