from distutils.core import Extension
from os.path import join
from sys import platform
import numpy as np

if 'win' in platform.lower() and not 'darwin' in platform.lower():
    extra_compile_args = ['/O2']
else:
    extra_compile_args = ['-O3', '-w']

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
        extra_compile_args=extra_compile_args,
    )
]


def build(setup_kwargs):
    setup_kwargs.update({'ext_modules': extensions})
