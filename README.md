[![Test Status](https://github.com/moble/quaternion/workflows/tests/badge.svg)](https://github.com/moble/quaternion/actions)
[![Documentation Status](https://readthedocs.org/projects/quaternion/badge/?version=latest)](https://quaternion.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/numpy-quaternion?color=)](https://pypi.org/project/numpy-quaternion/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/quaternion.svg?color=)](https://anaconda.org/conda-forge/quaternion)
[![MIT License](https://img.shields.io/github/license/moble/quaternion.svg)](https://github.com/moble/quaternion/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/27896013.svg)](https://zenodo.org/badge/latestdoi/27896013)


# Quaternions in numpy

This Python module adds a quaternion dtype to NumPy.

The code was originally based on [code by Martin
Ling](https://github.com/martinling/numpy_quaternion) (which he wrote
with help from Mark Wiebe), but was rewritten with ideas from
[rational](https://github.com/numpy/numpy-dtypes/tree/master/npytypes/rational)
to work with newer python versions (and to fix a few bugs), and
*greatly* expands the applications of quaternions.

See also the pure-python package
[quaternionic](https://github.com/moble/quaternionic).

## Quickstart

```sh
conda install -c conda-forge quaternion
```

or

```sh
python -m pip install --upgrade --force-reinstall numpy-quaternion
```

Optionally add `--user` after `install` in the second command if
you're not using a python environment — though you should start.


## Installation

Assuming you use `conda` to manage your python installation (which is
currently the preferred choice for science and engineering with
python), you can install this package simply as

```sh
conda install -c conda-forge quaternion
```

If you prefer to use `pip`, you can instead do

```sh
python -m pip install --upgrade --force-reinstall numpy-quaternion
```

(See [here](https://snarky.ca/why-you-should-use-python-m-pip/) for a
veteran python core contributor's explanation of why you should always
use `python -m pip` instead of just `pip` or `pip3`.)  The `--upgrade
--force-reinstall` options are not always necessary, but will ensure
that pip will update numpy if it has to.

If you refuse to use `conda`, you might want to install inside your
home directory without root privileges.  (Conda does this by default
anyway.)  This is done by adding `--user` to the above command:

```sh
python -m pip install --user --upgrade --force-reinstall numpy-quaternion
```

Note that pip will attempt to compile the code — which requires a
working `C` compiler.

Finally, there's also the fully manual option of just downloading the
code, changing to the code directory, and running

```sh
python -m pip install --upgrade --force-reinstall .
```

This should work regardless of the installation method, as long as you
have a compiler hanging around.


## Basic usage

The full documentation can be found on [Read the
Docs](https://quaternion.readthedocs.io/), and most functions have
docstrings that should explain the relevant points.  The following are
mostly for the purposes of example.

```python
>>> import numpy as np
>>> import quaternion
>>> np.quaternion(1,0,0,0)
quaternion(1, 0, 0, 0)
>>> q1 = np.quaternion(1,2,3,4)
>>> q2 = np.quaternion(5,6,7,8)
>>> q1 * q2
quaternion(-60, 12, 30, 24)
>>> a = np.array([q1, q2])
>>> a
array([quaternion(1, 2, 3, 4), quaternion(5, 6, 7, 8)], dtype=quaternion)
>>> np.exp(a)
array([quaternion(1.69392, -0.78956, -1.18434, -1.57912),
       quaternion(138.909, -25.6861, -29.9671, -34.2481)], dtype=quaternion)
```

Note that this package represents a quaternion as a scalar, followed
by the `x` component of the vector part, followed by `y`, followed by
`z`.  These components can be accessed directly:
```python
>>> q1.w, q1.x, q1.y, q1.z
(1.0, 2.0, 3.0, 4.0)
```
However, this only works on an individual `quaternion`; for arrays it
is better to use "vectorized" operations like `as_float_array`.

The following ufuncs are implemented (which means they run fast on
numpy arrays):
```python
add, subtract, multiply, divide, log, exp, power, negative, conjugate,
copysign, equal, not_equal, less, less_equal, isnan, isinf, isfinite, absolute
```

Quaternion components are stored as double-precision floating point
numbers — `float`s, in python language, or `float64` in more precise
numpy language.  Numpy arrays with `dtype=quaternion` can be accessed
as arrays of doubles without any (slow, memory-consuming) copying of
data; rather, a `view` of the exact same memory space can be created
within a microsecond, regardless of the shape or size of the
quaternion array.

Comparison operations follow the same lexicographic ordering as
tuples.

The unary tests isnan and isinf return true if they would return true
for any individual component; isfinite returns true if it would return
true for all components.

Real types may be cast to quaternions, giving quaternions with zero
for all three imaginary components. Complex types may also be cast to
quaternions, with their single imaginary component becoming the first
imaginary component of the quaternion. Quaternions may not be cast to
real or complex types.

Several array-conversion functions are also included.  For example, to
convert an Nx4 array of floats to an N-dimensional array of
quaternions, use `as_quat_array`:
```python
>>> import numpy as np
>>> import quaternion
>>> a = np.random.rand(7, 4)
>>> a
array([[ 0.93138726,  0.46972279,  0.18706385,  0.86605021],
       [ 0.70633523,  0.69982741,  0.93303559,  0.61440879],
       [ 0.79334456,  0.65912598,  0.0711557 ,  0.46622885],
       [ 0.88185987,  0.9391296 ,  0.73670503,  0.27115149],
       [ 0.49176628,  0.56688076,  0.13216632,  0.33309146],
       [ 0.11951624,  0.86804078,  0.77968826,  0.37229404],
       [ 0.33187593,  0.53391165,  0.8577846 ,  0.18336855]])
>>> qs = quaternion.as_quat_array(a)
>>> qs
array([ quaternion(0.931387262880247, 0.469722787598354, 0.187063852060487, 0.866050210100621),
       quaternion(0.706335233363319, 0.69982740767353, 0.933035590130247, 0.614408786768725),
       quaternion(0.793344561317281, 0.659125976566815, 0.0711557025000925, 0.466228847713644),
       quaternion(0.881859869074069, 0.939129602918467, 0.736705031709562, 0.271151494174001),
       quaternion(0.491766284854505, 0.566880763189927, 0.132166320200012, 0.333091463422536),
       quaternion(0.119516238634238, 0.86804077992676, 0.779688263524229, 0.372294043850009),
       quaternion(0.331875925159073, 0.533911652483908, 0.857784598617977, 0.183368547490701)], dtype=quaternion)
```
[Note that quaternions are printed with full precision, unlike floats,
which is why you see extra digits above.  But the actual data is
identical in the two cases.]  To convert an N-dimensional array of
quaternions to an Nx4 array of floats, use `as_float_array`:
```python
>>> b = quaternion.as_float_array(qs)
>>> b
array([[ 0.93138726,  0.46972279,  0.18706385,  0.86605021],
       [ 0.70633523,  0.69982741,  0.93303559,  0.61440879],
       [ 0.79334456,  0.65912598,  0.0711557 ,  0.46622885],
       [ 0.88185987,  0.9391296 ,  0.73670503,  0.27115149],
       [ 0.49176628,  0.56688076,  0.13216632,  0.33309146],
       [ 0.11951624,  0.86804078,  0.77968826,  0.37229404],
       [ 0.33187593,  0.53391165,  0.8577846 ,  0.18336855]])
```

It is also possible to convert a quaternion to or from a 3x3 array of
floats representing a rotation matrix, or an array of N quaternions to
or from an Nx3x3 array of floats representing N rotation matrices,
using `as_rotation_matrix` and `from_rotation_matrix`.  Similar
conversions are possible for rotation vectors using
`as_rotation_vector` and `from_rotation_vector`, and for spherical
coordinates using `as_spherical_coords` and `from_spherical_coords`.
Finally, it is possible to derive the Euler angles from a quaternion
using `as_euler_angles`, or create a quaternion from Euler angles
using `from_euler_angles` — though be aware that Euler angles are
basically the worst things
ever.<sup>[1](#1-euler-angles-are-awful)</sup> Before you complain
about those functions using something other than your favorite
conventions, please read [this
page](https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible).


## Dependencies

With the standard installation methods, hopefully you won't need to
worry about dependencies directly.  But in case you do, here's what
you need to know.

The basic requirements for this code are reasonably current versions
of `python` and `numpy`.  In particular, `python` versions 3.10
through 3.13 are routinely tested.  Because of its crucial dependence
on `numpy`, this package can only support versions of `python` that
are directly supported by `numpy` — which limits support to releases
from the past few years.  Old versions of `python` will work with
*older* versions of this package, which are still available from PyPI
and conda-forge.  Some older versions of `python` may still work with
newer versions of this package, but your mileage may vary.

However, certain advanced functions in this package (including
`squad`, `mean_rotor_in_intrinsic_metric`,
`integrate_angular_velocity`, and related functions) require
[`scipy`](http://scipy.org/) and can automatically use
[`numba`](http://numba.pydata.org/).  `Scipy` is a standard python
package for scientific computation, and implements interfaces to C and
Fortran codes for optimization (among other things) need for finding
mean and optimal rotors.  `Numba` uses [LLVM](http://llvm.org/) to
compile python code to machine code, accelerating many numerical
functions by factors of anywhere from 2 to 2000.  It is *possible* to
run all the code without `numba`, but these particular functions can
be anywhere from 4 to 400 times slower without it.

Both `scipy` and `numba` can be installed with `pip` or `conda`.
However, because `conda` is specifically geared toward scientific
python, it is generally more robust for these more complicated
packages.  In fact, the main
[`anaconda`](https://www.anaconda.com/products/individual) package
comes with both `numba` and `scipy`.  If you prefer the smaller
download size of [`miniconda`](http://conda.pydata.org/miniconda.html)
(which comes with minimal extras), you'll also have to run this
command:

```sh
conda install numpy scipy numba
```


## Bug reports and feature requests

Bug reports and feature requests are entirely welcome (with [very few
exceptions](https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible#opening-issues-and-pull-requests)).
The best way to do this is to open an [issue on this code's github
page](https://github.com/moble/quaternion/issues).  For bug reports,
please try to include a minimal working example demonstrating the
problem.

[Pull requests](https://help.github.com/articles/using-pull-requests/)
are also entirely welcome, of course, if you have an idea where the
code is going wrong, or have an idea for a new feature that you know
how to implement.

This code is routinely tested on recent versions of both python (3.8
though 3.11) and numpy (>=1.13).  But the test coverage is not
necessarily as complete as it could be, so bugs may certainly be
present, especially in the higher-level functions like
`mean_rotor_...`.


## Acknowledgments

This code is, of course, hosted on github.  Because it is an
open-source project, the hosting is free, and all the wonderful
features of github are available, including free wiki space and web
page hosting, pull requests, a nice interface to the git logs, etc.
Github user Hannes Ovrén (hovren) pointed out some errors in a
previous version of this code and suggested some nice utility
functions for rotation matrices, etc.  Github user Stijn van Drongelen
(rhymoid) contributed some code that makes compilation work with
MSVC++.  Github user Jon Long (longjon) has provided some elegant
contributions to substantially improve several tricky parts of this
code.  Rebecca Turner (9999years) and Leo Stein (duetosymmetry) did
all the work in getting the documentation onto [Read the
Docs](https://quaternion.readthedocs.io/).

Every change in this code is [automatically
tested](https://github.com/moble/quaternion/actions) on Github
Actions.  The code is downloaded and installed fresh each time, and
then tested, on each of the different supported versions of python, on
each of the supported platforms.  This ensures that no change I make
to the code breaks either installation or any of the features that I
have written tests for.  Github Actions also automatically builds the
`pip` versions of the code hosted on
[pypi](https://pypi.python.org/pypi/numpy-quaternion).  Conda-forge
also uses Github Actions to build [the conda/mamba
version](https://github.com/conda-forge/quaternion-feedstock) hosted
on [anaconda.org](https://anaconda.org/conda-forge/quaternion).  These
are all free services for open-source projects like this one.

The work of creating this code was supported in part by the Sherman
Fairchild Foundation and by NSF Grants No. PHY-1306125 and
AST-1333129.




<br/>

---

###### <sup>1</sup> Euler angles are awful

Euler angles are pretty much [the worst things
ever](https://moble.github.io/spherical_functions/#euler-angles) and it
makes me feel bad even supporting them.  Quaternions are faster, more
accurate, basically free of singularities, more intuitive, and
generally easier to understand.  You can work entirely without Euler
angles (I certainly do).  You absolutely never need them.  But if
you really can't give them up, they are mildly supported.
