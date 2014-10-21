# Quaternion modules

This Python module adds a quaternion dtype to NumPy.

The code was originally based on [code by Martin
Ling](https://github.com/martinling/numpy_quaternion) (with help from
Mark Weibe), but has been updated to work with python 3.x, and to
expand the applications of quaternions (as well as to fix a few bugs).


## Requirements and Installation

The only requirements are python and numpy.  The code is [routinely
tested](https://travis-ci.org/moble/numpy_quaternion) on python
versions 2.6, 2.7, 3.2, 3.3, and 3.4, so any of these should be
acceptable.  Numpy versions 1.7 and newer should also work, but only
some fairly recent version is tested for each python, depending on
what [Travis-CI](https://travis-ci.org/) is currently using.

I highly recommend the
[anaconda](https://store.continuum.io/cshop/anaconda/) installation,
which is easy, free, and generally the best way to stay up-to-date
with python.  It installs into your user directory (no root password
needed), can easily be uninstalled, and doesn't interfere in any way
with your system python.

To install with anaconda or virtualenv (which will just go into your
user directory), simply run

```sh
$ python setup.py install
```

With other flavors of python, you can install to the user directory
with

```sh
$ python setup.py install --user
```

## Usage

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
 >>> exp(a)
 array([quaternion(1.69392, -0.78956, -1.18434, -1.57912),
        quaternion(138.909, -25.6861, -29.9671, -34.2481)], dtype=quaternion)
```

The following ufuncs are implemented (which means they run fast on
numpy arrays):

```python
add, subtract, multiply, divide, log, exp, power, negative, conjugate,
copysign, equal, not_equal, less, less_equal, isnan, isinf, isfinite, absolute
```

Quaternion components are stored as doubles.  Numpy arrays with
`dtype=quaternion` can be accessed as arrays of doubles without any
(slow, memory-consuming) copying of data; rather, a `view` of the
exact same memory space can be created within a microsecond,
regardless of the shape or size of the quaternion array.

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
