<a href="https://travis-ci.org/moble/quaternion"><img align="right" hspace="3" alt="Status of automatic build and test suite" src="https://travis-ci.org/moble/quaternion.svg?branch=master"></a> <a href="https://github.com/moble/quaternion/blob/master/LICENSE"><img align="right" hspace="3" alt="Code distributed under the open-source MIT license" src="http://moble.github.io/spherical_functions/images/MITLicenseBadge.svg"></a>

# Quaternion modules

This Python module adds a quaternion dtype to NumPy.

The code was originally based on
[code by Martin Ling](https://github.com/martinling/numpy_quaternion) (with
help from Mark Weibe), but has been essentially totally rewritten to work with
both python 2.x and 3.x, and to expand the applications of quaternions (and to
fix a few bugs).


## Dependencies

The basic requirements for this code are reasonably current versions
of `python` and `numpy`.  In particular, `python` versions 2.6, 2.8,
3.2, 3.3, and 3.4 are
[routinely tested](https://travis-ci.org/moble/quaternion), and
any `numpy` version greater than 1.7.0 should work.

However, certain advanced functions in this package (including
`squad`, `mean_rotor_in_intrinsic_metric`,
`integrate_angular_velocity`, and related functions) require
[`scipy`](http://scipy.org/) and can automatically use
[`numba`](http://numba.pydata.org/).  `Scipy` is a standard python
package for scientific computation, and implements interfaces to C and
Fortran codes for optimization (among other things) need for finding
mean and optimal rotors.  `Numba` uses [LLVM](http://llvm.org/) to
compile python code to machine code, accelerating most numerical
functions by factors of anywhere from 2 to 2000.  It is *possible* to
run all the code without `numba`, but these functions are roughly 4 to
400 times slower without it.

The only drawback of `numba` is that it is nontrivial to install on
its own.  Fortunately, the best python installer,
[`anaconda`](http://continuum.io/downloads), makes it trivial.  Just
install the main `anaconda` package, which installs both `numba` and
`scipy`.

If you prefer the smaller download size of
[`miniconda`](http://conda.pydata.org/miniconda.html) (which comes
with no extras beyond python), you'll also have to run this command:

```sh
conda install pip numpy scipy numba
```

## Installation

Installation is simple with `pip` (which should be present in any
reasonably new installation of `python`):

```sh
pip install git+git://github.com/moble/quaternion
```

If you refuse to use anaconda, you might want to install inside your
home directory without root privileges.  (Anaconda does this by
default anyway.)  This is done by adding `--user` to the above
command:

```sh
pip install --user git+git://github.com/moble/quaternion
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


## Bug reports and feature requests

Bug reports and feature requests are entirely welcome.  The best way
to do this is to open an
[issue on this code's github page](https://github.com/moble/quaternion/issues).
For bug reports, please try to include a minimal working example
demonstrating the problem.

[Pull requests](https://help.github.com/articles/using-pull-requests/)
are also entirely welcome, of course, if you have an idea where the
code is going wrong, or have an idea for a new feature that you know
how to implement.

This code is
[routinely tested](https://travis-ci.org/moble/quaternion) on
recent versions of both python (2.x and 3.x) and numpy (>=1.7).  But
the test coverage is not necessarily as complete as it should be, so
bugs may certainly be present, especially in the higher-level
functions like `mean_rotor_...`.


## Acknowledgments

This code is, of course, hosted on github.  Because it is an
open-source project, the hosting is free, and all the wonderful
features of github are available, including free wiki space and web
page hosting, pull requests, a nice interface to the git logs, etc.

Every change in this code is
[auomatically tested](https://travis-ci.org/moble/quaternion) on
[Travis-CI](https://travis-ci.org/).  This is a free service (for
open-source projects like this one), which integrates beautifully with
github, detecting each commit and automatically re-running the tests.
The code is downloaded and installed fresh each time, and then tested,
on each of the five different versions of python.  This ensures that
no change I make to the code breaks either installation or any of the
features that I have written tests for.

The work of creating this code was supported in part by the Sherman
Fairchild Foundation and by NSF Grants No. PHY-1306125 and
AST-1333129.


<br/><br/>
###### <sup>1</sup> Running without `numba`

It is technically *possible* to run these functions without `numba`.
However, some functions can be sped up by factors of up to 1500 when
using `numba`.  The anaconda distribution makes installation of
`numba` so simple that there's just no reason not to use it.
