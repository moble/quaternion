# Numpy

The most important part of this package's API is actually numpy's API,
because the `quaternion` type is added as a possible dtype for numpy
arrays. This means that everything you can do with numpy arrays will
also apply to quaternion arrays.

It can be helpful to think of quaternions as generalized complex
numbers; all the operations that you can do with complex numbers are
also available for quaternions.  This includes all the usual
arithmetic operations (addition, subtraction, multiplication,
division, and exponentiation), as well as exponential and logarithmic
functions.  Note that it is also possible to convert arrays of
quaternions to and from arrays of floats (with an extra dimension of
size 4), using the `from_float_array` and `as_float_array` functions.

Here, we will single out a few of the very most important numpy
functions that are particularly useful for quaternions.

## <code class="doc-symbol doc-symbol-heading doc-symbol-function"></code> `numpy.exp(q)`

This returns the exponential function applied to each quaternion in
the array.  This is the quaternion analog of the exponential function
for complex numbers, and is defined as usual:
```math
\exp(g) = \sum_{n=0}^\infty \frac{g^n}{n!}.
```
The interpretation is that if `g` is the "generator" of a rotation,
then `numpy.exp(g)` is the quaternion representation of the
corresponding rotation operator.

```python
import numpy as np
import quaternion
g = quaternion.from_float_array([[0, np.pi/4, 0, 0], [0, 0, np.pi/2, 0]])
R = np.exp(g)
print(R)
```
Here, `g` contains two quaternions, which are "generators" of a
rotation about the x-axis and a rotation about the y-axis.  The output
is
```python
[quaternion(0.707107, 0.707107, 0, 0)
 quaternion(0, 0, 1, 0)]
```
which represent the same things as unit quaternions.

However, note that these rotations are *not* by $\pi/4$ and $\pi/2$,
as one might guess by looking at $g$.  Rather, they are rotations by
*twice* those angles, because of the way quaternions represent
rotations.  Therefore, $g$ is not the axis-angle representation; it is
*half* the axis-angle representation.  If you wish to convert to or
from axis-angle representation, use the `from_rotation_vector` and
`as_rotation_vector` functions instead.


## <code class="doc-symbol doc-symbol-heading doc-symbol-function"></code> `numpy.log(R)`

The `log` function is (almost) the inverse of the `exp` function.  In
particular, we can go backwards from the `R` defined above to get back
to `g`.  If we evaluate

```python
print(np.log(R))
```

we see that the result is precisely the same as `g`.

However, just as with the complex logarithm, the quaternion logarithm
is multi-valued, and we must make a choice of branch cut.  The `log`
function always returns the value of the logarithm whose vector part
has magnitude less than or equal to $\pi$.  Thus, $exp(log(q)) = q$
will always be true, but $log(exp(g)) = g$ will only be true if the
vector part of $g$ has magnitude less than or equal to $\pi$.


## <code class="doc-symbol doc-symbol-heading doc-symbol-method"></code> `ndarray.conjugate()`

Just as with complex numbers and the complex conjugate, the quaternion
conjugate is extremely important.  This conjugate reverses the sign of
the vector part of the quaternion, and is used in many calculations,
including finding the inverse and norm of a quaternion, and rotating
vectors.

Numpy has two ways of accessing the
conjugate of an array: `conj` and `conjugate`.  The output of
`R.conj()` or `R.conjugate()` is

```python
[quaternion(0.707107, -0.707107, 0, 0)
 quaternion(0, 0, -1, 0)]
```

Note that `R * v * R.conj()` is a valid way to rotate the pure-vector
quaternion `v`, but it can be more efficient and accurate to use the
`rotate_vectors` function instead.
