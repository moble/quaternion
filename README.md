This Python module adds a quaternion dtype to NumPy.

The code was originally based on [code by Martin
Ling](https://github.com/martinling/numpy_quaternion) (with help from
Mark Weibe), but has been updated to work with python 3.4, and to
expand the applications of quaternions (as well as to fix a few bugs).

To build:

 $ python setup.py build

To install (may require administrator rights):

 # python setup.py install

Example:

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

The following ufuncs are implemented:
```python
add, subtract, multiply, divide, log, exp, power, negative, conjugate,
copysign, equal, not_equal, less, less_equal, isnan, isinf, isfinite, absolute
```

Quaternion components are stored as doubles.

Comparison operations follow the same lexicographic ordering as tuples.

The unary tests isnan and isinf return true if they would return true for any
individual component; isfinite returns true if it would return true for all
components.

Real types may be cast to quaternions, giving quaternions with zero for all
three imaginary components. Complex types may also be cast to quaternions,
with their single imaginary component becoming the first imaginary component of
the quaternion. Quaternions may not be cast to real or complex types.
