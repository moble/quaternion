#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import os
import operator

import math
import numpy as np
import quaternion
from numpy import *
import pytest

try:
    import scipy
    has_scipy = True
except:
    has_scipy = False


from sys import platform
on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())


eps = np.finfo(float).eps


def allclose(*args, **kwargs):
    kwargs.update({'verbose': True})
    return quaternion.allclose(*args, **kwargs)


def passer(b):
    pass
# Change this to strict_assert = assert_ to check for missing tests
strict_assert = passer


def ufunc_binary_utility(array1, array2, op, rtol=2*eps, atol=0.0):
    """Make sure broadcasting is consistent with individual operations

    Given two arrays, we expect a broadcast binary operation to be consistent with the individual operations.  This
    utility function simply goes through and checks that that is true.  For example, if the input operation is `*`,
    this function checks for each `i` that

        array1[i] * array2  ==  np.array([array1[i]*array2[j] for j in range(len(array2))])

    """
    for arg1 in array1:
        assert allclose(op(arg1, array2),
                        np.array([op(arg1, arg2) for arg2 in array2]),
                        rtol=rtol, atol=atol)
    for arg2 in array2:
        assert allclose(op(array1, arg2),
                        np.array([op(arg1, arg2) for arg1 in array1]),
                        rtol=rtol, atol=atol)

    if array1.shape == array2.shape:
        assert allclose(op(array1, array2),
                        np.array([op(arg1, arg2) for arg1, arg2 in zip(array1, array2)]),
                        rtol=rtol, atol=atol)


# The following fixtures are used to establish some re-usable data
# for the tests; they need to be re-constructed because some of the
# tests will change the values, but we want the values to be constant
# on every entry into a test.

@pytest.fixture
def Qs():
    q_nan1 = quaternion.dual_quaternion(np.nan, 0., 0., 0., 0., 0., 0., 0.)
    q_inf1 = quaternion.dual_quaternion(np.inf, 0., 0., 0., 0., 0., 0., 0.)
    q_minf1 = quaternion.dual_quaternion(-np.inf, 0., 0., 0., 0., 0., 0., 0.)
    q_0 = quaternion.dual_quaternion(0., 0., 0., 0., 0., 0., 0., 0.)
    q_1 = quaternion.dual_quaternion(1., 0., 0., 0., 0., 0., 0., 0.)
    x = quaternion.dual_quaternion(0., 1., 0., 0., 0., 0., 0., 0.)
    y = quaternion.dual_quaternion(0., 0., 1., 0., 0., 0., 0., 0.)
    z = quaternion.dual_quaternion(0., 0., 0., 1., 0., 0., 0., 0.)
    er = quaternion.dual_quaternion(0., 0., 0., 0., 1., 0., 0., 0.)
    ei = quaternion.dual_quaternion(0., 0., 0., 0., 0., 1., 0., 0.)
    ej = quaternion.dual_quaternion(0., 0., 0., 0., 0., 0., 1., 0.)
    ek = quaternion.dual_quaternion(0., 0., 0., 0., 0., 0., 0., 1.)
    Q = quaternion.dual_quaternion(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8)
    Qneg = quaternion.dual_quaternion(-1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7, -8.8)
    Qbar = quaternion.dual_quaternion(1.1, -2.2, -3.3, -4.4, 5.5, -6.6, -7.7, -8.8)
    return np.asarray([q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, er, ei, ej, ek, Q, Qneg, Qbar])


q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, er, ei, ej, ek, Q, Qneg, Qbar,  = range(len(Qs()))
Qs_zero = [i for i in range(len(Qs())) if not Qs()[i].nonzero()]
Qs_nonzero = [i for i in range(len(Qs())) if Qs()[i].nonzero()]
Qs_nan = [i for i in range(len(Qs())) if Qs()[i].isnan()]
Qs_nonnan = [i for i in range(len(Qs())) if not Qs()[i].isnan()]
Qs_nonnannonzero = [i for i in range(len(Qs())) if not Qs()[i].isnan() and Qs()[i].nonzero()]
Qs_inf = [i for i in range(len(Qs())) if Qs()[i].isinf()]
Qs_noninf = [i for i in range(len(Qs())) if not Qs()[i].isinf()]
Qs_noninfnonzero = [i for i in range(len(Qs())) if not Qs()[i].isinf() and Qs()[i].nonzero()]
Qs_finite = [i for i in range(len(Qs())) if Qs()[i].isfinite()]
Qs_nonfinite = [i for i in range(len(Qs())) if not Qs()[i].isfinite()]
Qs_finitenonzero = [i for i in range(len(Qs())) if Qs()[i].isfinite() and Qs()[i].nonzero()]


@pytest.fixture
def Rs():
    ones = [0, -1., 1.]
    rs = [np.quaternion(w, x, y, z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    np.random.seed(1842)
    rs = rs + [r.normalized() for r in [np.quaternion(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                                                      np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for i in range(20)]]
    return np.array(rs)


def dual_quaternion_all_close(dq1, dq2):
    eps = 1e-10
    if abs(dq1.w - dq2.w) > eps: return False
    if abs(dq1.x - dq2.x) > eps: return False
    if abs(dq1.y - dq2.y) > eps: return False
    if abs(dq1.z - dq2.z) > eps: return False
    if abs(dq1.er - dq2.er) > eps: return False
    if abs(dq1.ei - dq2.ei) > eps: return False
    if abs(dq1.ej - dq2.ej) > eps: return False
    if abs(dq1.ek - dq2.ek) > eps: return False
    return True


def test_dual_quaternion_members():
    Q = quaternion.dual_quaternion(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8)
    assert Q.real == 1.1
    assert Q.w == 1.1
    assert Q.x == 2.2
    assert Q.y == 3.3
    assert Q.z == 4.4
    assert Q.er == 5.5
    assert Q.ei == 6.6
    assert Q.ej == 7.7
    assert Q.ek == 8.8

    # quaternion tested members when quaternion is created without the real
    # component here. If we implement a similar constructor for dual_quaternion
    # then add test here


# Unary bool returners
def test_dual_quaternion_nonzero(Qs):
    assert not Qs[q_0].nonzero()  # Do this one explicitly, to not use circular logic
    assert Qs[q_1].nonzero()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_zero]:
        assert not q.nonzero()
    for q in Qs[Qs_nonzero]:
        assert q.nonzero()


def test_dual_quaternion_isnan(Qs):
    assert not Qs[q_0].isnan()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isnan()  # Do this one explicitly, to not use circular logic
    assert Qs[q_nan1].isnan()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nan]:
        assert q.isnan()
    for q in Qs[Qs_nonnan]:
        assert not q.isnan()


def test_dual_quaternion_isinf(Qs):
    assert not Qs[q_0].isinf()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isinf()  # Do this one explicitly, to not use circular logic
    assert Qs[q_inf1].isinf()  # Do this one explicitly, to not use circular logic
    assert Qs[q_minf1].isinf()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_inf]:
        assert q.isinf()
    for q in Qs[Qs_noninf]:
        assert not q.isinf()


def test_dual_quaternion_isfinite(Qs):
    assert not Qs[q_nan1].isfinite()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_inf1].isfinite()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_minf1].isfinite()  # Do this one explicitly, to not use circular logic
    assert Qs[q_0].isfinite()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nonfinite]:
        assert not q.isfinite()
    for q in Qs[Qs_finite]:
        assert q.isfinite()


# Binary bool returners
def test_dual_quaternion_equal(Qs):
    for j in Qs_nonnan:
        assert Qs[j] == Qs[j]  # self equality
        for k in range(len(Qs)):  # non-self inequality
            assert (j == k) or (not (Qs[j] == Qs[k]))
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert not q == p  # nan should never equal anything


def test_dual_quaternion_not_equal(Qs):
    for j in Qs_nonnan:
        assert not (Qs[j] != Qs[j])  # self non-not_equality
        for k in Qs_nonnan:  # non-self not_equality
            assert (j == k) or (Qs[j] != Qs[k])
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert q != p  # nan should never equal anything


def test_dual_quaternion_richcompare(Qs):
    for p in Qs:
        for q in Qs[Qs_nan]:
            assert not p < q
            assert not q < p
            assert not p <= q
            assert not q <= p
            assert not p.greater(q)
            assert not q.greater(p)
            assert not p.greater_equal(q)
            assert not q.greater_equal(p)
    for j in Qs_nonnan:
        p = Qs[j]
        assert (p < Qs[q_inf1]) or (j == q_inf1)
        assert (p <= Qs[q_inf1])
        assert (Qs[q_minf1] < p) or (j == q_minf1)
        assert (Qs[q_minf1] <= p)
        assert (Qs[q_inf1].greater(p)) or (j == q_inf1)
        assert (Qs[q_inf1].greater_equal(p))
        assert (p.greater(Qs[q_minf1])) or (j == q_minf1)
        assert (p.greater_equal(Qs[q_minf1]))
    for p in [Qs[q_1], Qs[x], Qs[y], Qs[z], Qs[Q], Qs[Qbar]]:
        assert Qs[q_0] < p
        assert Qs[q_0] <= p
        assert p.greater(Qs[q_0])
        assert p.greater_equal(Qs[q_0])
    for p in [Qs[Qneg]]:
        assert p < Qs[q_0]
        assert p <= Qs[q_0]
        assert Qs[q_0].greater(p)
        assert Qs[q_0].greater_equal(p)
    for p in [Qs[x], Qs[y], Qs[z]]:
        assert p < Qs[q_1]
        assert p <= Qs[q_1]
        assert Qs[q_1].greater(p)
        assert Qs[q_1].greater_equal(p)


# Unary quaternion returners
def test_dual_quaternion_negative(Qs):
    assert -Qs[Q] == Qs[Qneg]
    for q in Qs[Qs_finite]:
        assert -q == -1.0 * q
    for q in Qs[Qs_nonnan]:
        assert -(-q) == q


def test_dual_quaternion_conjugate(Qs):
    assert Qs[Q].conjugate() == Qs[Qbar]
    for q in Qs[Qs_nonnan]:
        assert q.conjugate() == q.conj()
        assert q.conjugate().conjugate() == q
        c = q.conjugate()
        assert c.w == q.w
        assert c.x == -q.x
        assert c.y == -q.y
        assert c.z == -q.z
        assert c.er == q.er
        assert c.ei == -q.ei
        assert c.ej == -q.ej
        assert c.ek == -q.ek


# Quaternion-quaternion binary quaternion returners
@pytest.mark.xfail
def test_dual_quaternion_copysign(Qs):
    assert False


# Quaternion-quaternion, scalar-quaternion, or quaternion-scalar binary quaternion returners
def test_dual_quaternion_add(Qs):
    for j in Qs_nonnan:
        for k in Qs_nonnan:
            q = Qs[j]
            p = Qs[k]
            assert (q + p == quaternion.dual_quaternion(q.w + p.w, q.x + p.x, q.y + p.y, q.z + p.z,
                                                        q.er + p.er, q.ei + p.ei, q.ej + p.ej, q.ek + p.ek)
                    or (j == q_inf1 and k == q_minf1)
                    or (k == q_inf1 and j == q_minf1))
    for q in Qs[Qs_nonnan]:
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert (q + s == quaternion.dual_quaternion(q.w + s, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek))
            assert (s + q == quaternion.dual_quaternion(q.w + s, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek))


@pytest.mark.xfail
def test_dual_quaternion_add_ufunc(Qs):
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.add)


def test_dual_quaternion_subtract(Qs):
    for q in Qs[Qs_finite]:
        for p in Qs[Qs_finite]:
            assert q - p == quaternion.dual_quaternion(q.w - p.w, q.x - p.x, q.y - p.y, q.z - p.z,
                                                  q.er - p.er, q.ei - p.ei, q.ej - p.ej, q.ek - p.ek)
    for q in Qs[Qs_nonnan]:
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert (q - s == quaternion.dual_quaternion(q.w - s, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek))
            assert (s - q == quaternion.dual_quaternion(s - q.w, -q.x, -q.y, -q.z, -q.er, -q.ei, -q.ej, -q.ek))

@pytest.mark.xfail
def test_quaternion_subtract_ufunc(Qs):
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.sub)


def test_dual_quaternion_multiply(Qs):
    # Check scalar multiplication
    for q in Qs[Qs_finite]:
        assert q * Qs[q_1] == q
    for q in Qs[Qs_finite]:
        assert q * 1.0 == q
        assert q * 1 == q
        assert 1.0 * q == q
        assert 1 * q == q
    for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
        for q in Qs[Qs_finite]:
            assert q * s == quaternion.dual_quaternion(s * q.w, s * q.x, s * q.y, s * q.z,
                                                       s * q.er, s * q.ei, s * q.ej, s * q.ek)
            assert s * q == q * s
    for q in Qs[Qs_finite]:
        assert 0.0 * q == Qs[q_0]
        assert 0.0 * q == q * 0.0

    # Check linearity
    for q1 in Qs[Qs_finite]:
        for q2 in Qs[Qs_finite]:
            for q3 in Qs[Qs_finite]:
                assert dual_quaternion_all_close(q1*(q2+q3), (q1*q2)+(q1*q3))
                assert dual_quaternion_all_close((q1+q2)*q3, (q1*q3)+(q2*q3))

    # Check the multiplication table
    for q in [Qs[q_1], Qs[x], Qs[y], Qs[z], Qs[ei], Qs[ej], Qs[ek]]:
        assert Qs[q_1] * q == q
        assert q * Qs[q_1] == q
    assert Qs[x] * Qs[x] == -Qs[q_1]
    assert Qs[x] * Qs[y] == Qs[z]
    assert Qs[x] * Qs[z] == -Qs[y]
    assert Qs[x] * Qs[er] == Qs[ei]
    assert Qs[x] * Qs[ei] == -Qs[er]
    assert Qs[x] * Qs[ej] == Qs[ek]
    assert Qs[x] * Qs[ek] == -Qs[ej]

    assert Qs[y] * Qs[x] == -Qs[z]
    assert Qs[y] * Qs[y] == -Qs[q_1]
    assert Qs[y] * Qs[z] == Qs[x]
    assert Qs[y] * Qs[er] == Qs[ej]
    assert Qs[y] * Qs[ei] == -Qs[ek]
    assert Qs[y] * Qs[ej] == -Qs[er]
    assert Qs[y] * Qs[ek] == Qs[ei]

    assert Qs[z] * Qs[x] == Qs[y]
    assert Qs[z] * Qs[y] == -Qs[x]
    assert Qs[z] * Qs[z] == -Qs[q_1]
    assert Qs[z] * Qs[er] == Qs[ek]
    assert Qs[z] * Qs[ei] == Qs[ej]
    assert Qs[z] * Qs[ej] == -Qs[ei]
    assert Qs[z] * Qs[ek] == -Qs[er]

    assert Qs[er] * Qs[x] == Qs[ei]
    assert Qs[er] * Qs[y] == Qs[ej]
    assert Qs[er] * Qs[z] == Qs[ek]
    assert Qs[er] * Qs[er] == Qs[q_0]
    assert Qs[er] * Qs[ei] == Qs[q_0]
    assert Qs[er] * Qs[ej] == Qs[q_0]
    assert Qs[er] * Qs[ek] == Qs[q_0]

    assert Qs[ei] * Qs[x] == -Qs[er]
    assert Qs[ei] * Qs[y] == Qs[ek]
    assert Qs[ei] * Qs[z] == -Qs[ej]
    assert Qs[ei] * Qs[er] == Qs[q_0]
    assert Qs[ei] * Qs[ei] == Qs[q_0]
    assert Qs[ei] * Qs[ej] == Qs[q_0]
    assert Qs[ei] * Qs[ek] == Qs[q_0]

    assert Qs[ej] * Qs[x] == -Qs[ek]
    assert Qs[ej] * Qs[y] == -Qs[er]
    assert Qs[ej] * Qs[z] == Qs[ei]
    assert Qs[ej] * Qs[er] == Qs[q_0]
    assert Qs[ej] * Qs[ei] == Qs[q_0]
    assert Qs[ej] * Qs[ej] == Qs[q_0]
    assert Qs[ej] * Qs[ek] == Qs[q_0]

    assert Qs[ek] * Qs[x] == Qs[ej]
    assert Qs[ek] * Qs[y] == -Qs[ei]
    assert Qs[ek] * Qs[z] == -Qs[er]
    assert Qs[ek] * Qs[er] == Qs[q_0]
    assert Qs[ek] * Qs[ei] == Qs[q_0]
    assert Qs[ek] * Qs[ej] == Qs[q_0]
    assert Qs[ek] * Qs[ek] == Qs[q_0]

@pytest.mark.xfail
def test_quaternion_multiply_ufunc(Qs):
    ufunc_binary_utility(np.array([quaternion.one]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([quaternion.one]), operator.mul)
    ufunc_binary_utility(np.array([1.0]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1.0]), operator.mul)
    ufunc_binary_utility(np.array([1]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1]), operator.mul)
    ufunc_binary_utility(np.array([0.0]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([0.0]), operator.mul)
    ufunc_binary_utility(np.array([0]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([0]), operator.mul)

    ufunc_binary_utility(np.array([-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]),
                         Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite],
                         np.array([-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]), operator.mul)

    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.mul)

@pytest.mark.xfail
def test_quaternion_divide(Qs):
    # Check scalar division
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q / q, quaternion.one)
        assert allclose(1 / q, q.inverse())
        assert allclose(1.0 / q, q.inverse())
        assert 0.0 / q == quaternion.zero
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert allclose(s / q, s * (q.inverse()))
    for q in Qs[Qs_nonnan]:
        assert q / 1.0 == q
        assert q / 1 == q
        for s in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
            assert allclose(q / s, q * (1.0/s))

    # Check linearity
    for q1 in Qs[Qs_finite]:
        for q2 in Qs[Qs_finite]:
            for q3 in Qs[Qs_finitenonzero]:
                assert allclose((q1+q2)/q3, (q1/q3)+(q2/q3))

    # Check the multiplication table
    for q in [Qs[q_1], Qs[x], Qs[y], Qs[z]]:
        assert Qs[q_1] / q == q.conj()
        assert q / Qs[q_1] == q
    assert Qs[x] / Qs[x] == Qs[q_1]
    assert Qs[x] / Qs[y] == -Qs[z]
    assert Qs[x] / Qs[z] == Qs[y]
    assert Qs[y] / Qs[x] == Qs[z]
    assert Qs[y] / Qs[y] == Qs[q_1]
    assert Qs[y] / Qs[z] == -Qs[x]
    assert Qs[z] / Qs[x] == -Qs[y]
    assert Qs[z] / Qs[y] == Qs[x]
    assert Qs[z] / Qs[z] == Qs[q_1]

@pytest.mark.xfail
def test_quaternion_divide_ufunc(Qs):
    ufunc_binary_utility(np.array([quaternion.one]), Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], np.array([quaternion.one]), operator.truediv)
    ufunc_binary_utility(np.array([1.0]), Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1.0]), operator.truediv)
    ufunc_binary_utility(np.array([1]), Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1]), operator.truediv)
    ufunc_binary_utility(np.array([0.0]), Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(np.array([0]), Qs[Qs_finitenonzero], operator.truediv)

    ufunc_binary_utility(np.array([-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]),
                         Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(Qs[Qs_finitenonzero],
                         np.array([-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]), operator.truediv)

    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finitenonzero], operator.floordiv)

    ufunc_binary_utility(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero], operator.truediv)
    ufunc_binary_utility(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero], operator.floordiv)




def test_quaternion_getset(Qs):
    # get parts a and b
    for q in Qs[Qs_nonnan]:
        assert q.a == q.w + 1j * q.z
        assert q.b == q.y + 1j * q.x
    # Check multiplication law for parts a and b
    part_mul_precision = 1.e-14
    for p in Qs[Qs_finite]:
        for q in Qs[Qs_finite]:
            assert abs((p * q).a - (p.a * q.a - p.b.conjugate() * q.b)) < part_mul_precision
            assert abs((p * q).b - (p.b * q.a + p.a.conjugate() * q.b)) < part_mul_precision
    # get components/vec
    for q in Qs[Qs_nonnan]:
        assert np.array_equal(q.components, np.array([q.w, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek]))
        assert np.array_equal(q.vec, np.array([q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek]))
        assert np.array_equal(q.imag, np.array([q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek]))
    # set components/vec from np.array, list, tuple
    for q in Qs[Qs_nonnan]:
        for seq_type in [np.array, list, tuple]:
            p = np.dual_quaternion(*q.components)
            r = np.dual_quaternion(*q.components)
            s = np.dual_quaternion(*q.components)
            p.components = seq_type((-1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8))
            r.vec = seq_type((6.6, -7.7, 8.8, 2.2, -3.3, 4.4, -5.5))
            s.imag = seq_type((6.6, -7.7, 8.8, 2.2, -3.3, 4.4, -5.5))
            assert np.array_equal(p.components, np.array([-1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8]))
            assert np.array_equal(r.components, np.array([q.w, 6.6, -7.7, 8.8, 2.2, -3.3, 4.4, -5.5]))
            assert np.array_equal(s.components, np.array([q.w, 6.6, -7.7, 8.8, 2.2, -3.3, 4.4, -5.5]))
    # TypeError when setting components with the wrong type or size of thing
    for q in Qs:
        for seq_type in [np.array, list, tuple]:
            p = np.dual_quaternion(*q.components)
            r = np.dual_quaternion(*q.components)
            s = np.dual_quaternion(*q.components)
            with pytest.raises(TypeError):
                p.components = '1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8'
            with pytest.raises(TypeError):
                p.components = seq_type([])
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5,))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6,))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6, -7.7,))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6, -7.7, 8.8, -9.9))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6, -7.7, 8.8, -9.9, 1.1, 2.2, 3.3, 4.4))

            with pytest.raises(TypeError):
                r.vec = '2.2, 3.3, 4.4, 5.5, 6.6, -7.7, 8.8'
            with pytest.raises(TypeError):
                r.vec = seq_type([])
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5,))
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5, 6.6))
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5, 6.6, -7.7, 8.8))
            with pytest.raises(TypeError):
                r.vec = seq_type((2.2, 3.3, 4.4, -5.5, 6.6, -7.7, 8.8, 9,9))

            with pytest.raises(TypeError):
                s.vec = '2.2, 3.3, 4.4, 5.5, 6.6, -7.7, 8.8'
            with pytest.raises(TypeError):
                s.vec = seq_type([])
            with pytest.raises(TypeError):
                s.vec = seq_type((-5.5,))
            with pytest.raises(TypeError):
                s.vec = seq_type((-5.5, 6.6))
            with pytest.raises(TypeError):
                s.vec = seq_type((-5.5, 6.6, -7.7, 8.8))
            with pytest.raises(TypeError):
                s.vec = seq_type((2.2, 3.3, 4.4, -5.5, 6.6, -7.7, 8.8, 9, 9))



@pytest.mark.xfail
def test_arrfuncs():
    # nonzero
    # copyswap
    # copyswapn
    # getitem
    # setitem
    # compare
    # argmax
    # fillwithscalar
    assert False


def test_setitem_quat(Qs):
    Ps = Qs.copy()
    # setitem from quaternion
    for j in range(len(Ps)):
        Ps[j] = np.dual_quaternion(1.3, 2.4, 3.5, 4.7, 5.6, 6.8, 7.3, 8.2)
        for k in range(j + 1):
            assert Ps[k] == np.dual_quaternion(1.3, 2.4, 3.5, 4.7, 5.6, 6.8, 7.3, 8.2)
        for k in range(j + 1, len(Ps)):
            assert Ps[k] == Qs[k]
    # setitem from np.array, list, or tuple
    for seq_type in [np.array, list, tuple]:
        Ps = Qs.copy()
        with pytest.raises(TypeError):
            Ps[0] = seq_type(())
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4,))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5, 4.7, 5.9))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5, 4.7, 5.9, 6.6, 7.8, 8.3, 9.2))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5, 4.7, 5.9, np.nan))
        for j in range(len(Ps)):
            Ps[j] = seq_type((1.3, 2.4, 3.5, 4.7, 5.6, 6.8, 7.3, 8.2))
            for k in range(j + 1):
                assert Ps[k] == np.dual_quaternion(1.3, 2.4, 3.5, 4.7, 5.6, 6.8, 7.3, 8.2)
            for k in range(j + 1, len(Ps)):
                assert Ps[k] == Qs[k]
    with pytest.raises(TypeError):
        Ps[0] = 's'
    with pytest.raises(TypeError):
        Ps[0] = 's'


@pytest.mark.xfail
def test_arraydescr():
    # new
    # richcompare
    # hash
    # repr
    # str
    assert False


@pytest.mark.xfail
def test_casts():
    # FLOAT, npy_float
    # DOUBLE, npy_double
    # LONGDOUBLE, npy_longdouble
    # BOOL, npy_bool
    # BYTE, npy_byte
    # UBYTE, npy_ubyte
    # SHORT, npy_short
    # USHORT, npy_ushort
    # INT, npy_int
    # UINT, npy_uint
    # LONG, npy_long
    # ULONG, npy_ulong
    # LONGLONG, npy_longlong
    # ULONGLONG, npy_ulonglong
    # CFLOAT, npy_float
    # CDOUBLE, npy_double
    # CLONGDOUBLE, npy_longdouble
    assert False

@pytest.mark.xfail
def test_ufuncs(Rs, Qs):
    np.random.seed(1234)
    assert np.allclose(np.abs(Rs), np.ones(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(np.log(Rs) - np.array([r.log() for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                       rtol=1.e-15)
    assert np.allclose(np.abs(np.exp(Rs) - np.array([r.exp() for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                       rtol=1.e-15)
    assert np.allclose(np.abs(Rs - Rs), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(Rs + (-Rs)), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(np.conjugate(Rs) - np.array([r.conjugate() for r in Rs])), np.zeros(Rs.shape),
                       atol=1.e-14, rtol=1.e-15)
    assert np.all(Rs == Rs)
    assert np.all(Rs <= Rs)
    for i in range(10):
        x = np.random.uniform(-10, 10)
        assert np.allclose(np.abs(Rs * x - np.array([r * x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        # assert np.allclose( np.abs( x*Rs - np.array([r*x for r in Rs]) ), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        strict_assert(False)
        assert np.allclose(np.abs(Rs / x - np.array([r / x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        assert np.allclose(np.abs(Rs ** x - np.array([r ** x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                           rtol=1.e-15)
    assert np.allclose(
        np.abs(Qs[Qs_finite] + Qs[Qs_finite] - np.array([q1 + q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(
        np.abs(Qs[Qs_finite] - Qs[Qs_finite] - np.array([q1 - q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(
        np.abs(Qs[Qs_finite] * Qs[Qs_finite] - np.array([q1 * q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    for Q in Qs[Qs_finite]:
        assert np.allclose(np.abs(Qs[Qs_finite] * Q - np.array([q1 * Q for q1 in Qs[Qs_finite]])),
                           np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
        # assert np.allclose( np.abs( Q*Qs[Qs_finite] - np.array([Q*q1 for q1 in Qs[Qs_finite]]) ),
        # np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(Qs[Qs_finitenonzero] / Qs[Qs_finitenonzero]
                              - np.array([q1 / q2 for q1, q2 in zip(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero])])),
                       np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(Qs[Qs_finitenonzero] ** Qs[Qs_finitenonzero]
                              - np.array([q1 ** q2 for q1, q2 in zip(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero])])),
                       np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(~Qs[Qs_finitenonzero]
                              - np.array([q.inverse() for q in Qs[Qs_finitenonzero]])),
                       np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)


def test_numpy_array_conversion(Qs):
    "Check conversions between array as quaternions and array as floats"
    # First, just check 1-d array
    Q = Qs[Qs_nonnan][:12]  # Select first 3x4=12 non-nan elements in Qs
    assert Q.dtype == np.dtype(np.dual_quaternion)
    q = quaternion.as_float_array(Q)  # View as array of floats
    assert q.dtype == np.dtype(np.float)
    assert q.shape == (12, 8)  # This is the expected shape
    for j in range(12):
        for k in range(4):  # Check each component individually
            assert q[j][k] == Q[j].components[k]
    assert np.array_equal(quaternion.as_dual_quat_array(q), Q)  # Check that we can go backwards
    # Next, see how that works if I flatten the q array
    q = q.flatten()
    assert q.dtype == np.dtype(np.float)
    assert q.shape == (96,)
    for j in range(96):
        assert q[j] == Q[j // 8].components[j % 8]
    assert np.array_equal(quaternion.as_dual_quat_array(q), Q)  # Check that we can go backwards
    # Now, reshape into 2-d array, and re-check
    P = Q.reshape(3, 4)  # Reshape into 3x4 array of quaternions
    p = quaternion.as_float_array(P)  # View as array of floats
    assert p.shape == (3, 4, 8)  # This is the expected shape
    for j in range(3):
        for k in range(4):
            for l in range(8):  # Check each component individually
                assert p[j][k][l] == Q[4 * j + k].components[l]
    assert np.array_equal(quaternion.as_dual_quat_array(p), P)  # Check that we can go backwards
    # Check that we get an exception if the final dimension is not divisible by 8
    with pytest.raises(ValueError):
        quaternion.as_dual_quat_array(np.random.rand(4, 1))
    with pytest.raises(ValueError):
        quaternion.as_dual_quat_array(np.random.rand(4, 2))
    with pytest.raises(ValueError):
        quaternion.as_dual_quat_array(np.random.rand(4, 14))
    with pytest.raises(ValueError):
        quaternion.as_dual_quat_array(np.random.rand(4, 17))
    with pytest.raises(ValueError):
        quaternion.as_dual_quat_array(np.random.rand(4, 5, 3, 2, 1))
    # Finally, check that it works on non-contiguous arrays, by adding random padding and then slicing
    q = quaternion.as_float_array(Q)
    q = np.concatenate((np.random.rand(q.shape[0], 3), q, np.random.rand(q.shape[0], 3)), axis=1)
    assert np.array_equal(quaternion.as_dual_quat_array(q[:, 3:11]), Q)


if __name__ == '__main__':
    print("The tests should be run automatically via pytest (pip install pytest)")




