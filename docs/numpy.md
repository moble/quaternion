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
functions.

Here, we will single out a few of the very most important numpy
functions that are particularly useful for quaternions.

## <code class="doc-symbol doc-symbol-heading doc-symbol-function"></code> `numpy.exp(q)`

This returns the exponential function applied to each quaternion in
the array.  This is the quaternion analog of the exponential function
for complex numbers.  The interpretation is that if `q` is the
axis-angle representation of a rotation, then `numpy.exp(q)` is the
quaternion representation of the same rotation.

```python
import numpy as np
import quaternion
q = quaternion.from_float_array([[0, np.pi/4, 0, 0], [0, 0, np.pi/2, 0]])
R = np.exp(q)
print(R)
```
Here, `q` contains two quaternions, which are the axis-angle
representations of a $\pi/4$ (45°) rotation about the x-axis and a
$\pi/2$ (90°) rotation about the y-axis.  The output is
```python
[quaternion(0.707107, 0.707107, 0, 0)
 quaternion(0, 0, 1, 0)]
```
which represent the same things as unit quaternions.


## <code class="doc-symbol doc-symbol-heading doc-symbol-function"></code> `numpy.log(R)`

The `log` function is (almost) the inverse of the `exp` function.  In
particular, we can go backwards from the `R` defined above to get back
to `q`.  If we evaluate

```python
print(np.log(R))
```

we see that the result is precisely the same as `q`.


## <code class="doc-symbol doc-symbol-heading doc-symbol-method"></code> `ndarray.conjugate()`

Just as with complex numbers and the complex conjugate, the quaternion
conjugate is extremely important.

```python
R.conjugate()
R.conj()
```


