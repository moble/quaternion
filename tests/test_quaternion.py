#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    return make_Qs()
    
def make_Qs():
    q_nan1 = quaternion.quaternion(np.nan, 0., 0., 0.)
    q_inf1 = quaternion.quaternion(np.inf, 0., 0., 0.)
    q_minf1 = quaternion.quaternion(-np.inf, 0., 0., 0.)
    q_0 = quaternion.quaternion(0., 0., 0., 0.)
    q_1 = quaternion.quaternion(1., 0., 0., 0.)
    x = quaternion.quaternion(0., 1., 0., 0.)
    y = quaternion.quaternion(0., 0., 1., 0.)
    z = quaternion.quaternion(0., 0., 0., 1.)
    Q = quaternion.quaternion(1.1, 2.2, 3.3, 4.4)
    Qneg = quaternion.quaternion(-1.1, -2.2, -3.3, -4.4)
    Qbar = quaternion.quaternion(1.1, -2.2, -3.3, -4.4)
    Qnormalized = quaternion.quaternion(0.18257418583505537115232326093360,
                                        0.36514837167011074230464652186720,
                                        0.54772255750516611345696978280080,
                                        0.73029674334022148460929304373440)
    Qlog = quaternion.quaternion(1.7959088706354, 0.515190292664085,
                                 0.772785438996128, 1.03038058532817)
    Qexp = quaternion.quaternion(2.81211398529184, -0.392521193481878,
                                 -0.588781790222817, -0.785042386963756)
    return np.array([q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp],
                    dtype=np.quaternion)

Qs_array = make_Qs()


q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp, = range(len(Qs_array))
Qs_zero = [i for i in range(len(Qs_array)) if not Qs_array[i].nonzero()]
Qs_nonzero = [i for i in range(len(Qs_array)) if Qs_array[i].nonzero()]
Qs_nan = [i for i in range(len(Qs_array)) if Qs_array[i].isnan()]
Qs_nonnan = [i for i in range(len(Qs_array)) if not Qs_array[i].isnan()]
Qs_nonnannonzero = [i for i in range(len(Qs_array)) if not Qs_array[i].isnan() and Qs_array[i].nonzero()]
Qs_inf = [i for i in range(len(Qs_array)) if Qs_array[i].isinf()]
Qs_noninf = [i for i in range(len(Qs_array)) if not Qs_array[i].isinf()]
Qs_noninfnonzero = [i for i in range(len(Qs_array)) if not Qs_array[i].isinf() and Qs_array[i].nonzero()]
Qs_finite = [i for i in range(len(Qs_array)) if Qs_array[i].isfinite()]
Qs_nonfinite = [i for i in range(len(Qs_array)) if not Qs_array[i].isfinite()]
Qs_finitenonzero = [i for i in range(len(Qs_array)) if Qs_array[i].isfinite() and Qs_array[i].nonzero()]


@pytest.fixture
def Rs():
    ones = [0, -1., 1.]
    rs = [np.quaternion(w, x, y, z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    np.random.seed(1842)
    rs = rs + [r.normalized() for r in [np.quaternion(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                                                      np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for i in range(20)]]
    return np.array(rs)


def test_quaternion_members():
    Q = quaternion.quaternion(1.1, 2.2, 3.3, 4.4)
    assert Q.real == 1.1
    assert Q.w == 1.1
    assert Q.x == 2.2
    assert Q.y == 3.3
    assert Q.z == 4.4


def test_quaternion_constructors():
    Q = quaternion.quaternion(2.2, 3.3, 4.4)
    assert Q.real == 0.0
    assert Q.w == 0.0
    assert Q.x == 2.2
    assert Q.y == 3.3
    assert Q.z == 4.4
    
    P = quaternion.quaternion(1.1, 2.2, 3.3, 4.4)
    Q = quaternion.quaternion(P)
    assert Q.real == 1.1
    assert Q.w == 1.1
    assert Q.x == 2.2
    assert Q.y == 3.3
    assert Q.z == 4.4

    Q = quaternion.quaternion(1.1)
    assert Q.real == 1.1
    assert Q.w == 1.1
    assert Q.x == 0.0
    assert Q.y == 0.0
    assert Q.z == 0.0

    Q = quaternion.quaternion(0.0)
    assert Q.real == 0.0
    assert Q.w == 0.0
    assert Q.x == 0.0
    assert Q.y == 0.0
    assert Q.z == 0.0

    with pytest.raises(TypeError):
        quaternion.quaternion(1.2, 3.4)

    with pytest.raises(TypeError):
        quaternion.quaternion(1.2, 3.4, 5.6, 7.8, 9.0)


def test_constants():
    assert quaternion.one == np.quaternion(1.0, 0.0, 0.0, 0.0)
    assert quaternion.x == np.quaternion(0.0, 1.0, 0.0, 0.0)
    assert quaternion.y == np.quaternion(0.0, 0.0, 1.0, 0.0)
    assert quaternion.z == np.quaternion(0.0, 0.0, 0.0, 1.0)


def test_isclose():
    from quaternion import x, y

    assert np.array_equal(quaternion.isclose([1e10*x, 1e-7*y], [1.00001e10*x, 1e-8*y], rtol=1.e-5, atol=2.e-8),
                          np.array([True, False]))
    assert np.array_equal(quaternion.isclose([1e10*x, 1e-8*y], [1.00001e10*x, 1e-9*y], rtol=1.e-5, atol=2.e-8),
                          np.array([True, True]))
    assert np.array_equal(quaternion.isclose([1e10*x, 1e-8*y], [1.0001e10*x, 1e-9*y], rtol=1.e-5, atol=2.e-8),
                          np.array([False, True]))
    assert np.array_equal(quaternion.isclose([x, np.nan*y], [x, np.nan*y]),
                          np.array([True, False]))
    assert np.array_equal(quaternion.isclose([x, np.nan*y], [x, np.nan*y], equal_nan=True),
                          np.array([True, True]))

    np.random.seed(1234)
    a = quaternion.as_quat_array(np.random.random((3, 5, 4)))
    assert quaternion.allclose(1e10 * a, 1.00001e10 * a, rtol=1.e-5, atol=2.e-8) == True
    assert quaternion.allclose(1e-7 * a, 1e-8 * a, rtol=1.e-5, atol=2.e-8) == False
    assert quaternion.allclose(1e10 * a, 1.00001e10 * a, rtol=1.e-5, atol=2.e-8) == True
    assert quaternion.allclose(1e-8 * a, 1e-9 * a, rtol=1.e-5, atol=2.e-8) == True
    assert quaternion.allclose(1e10 * a, 1.0001e10 * a, rtol=1.e-5, atol=2.e-8) == False
    assert quaternion.allclose(1e-8 * a, 1e-9 * a, rtol=1.e-5, atol=2.e-8) == True
    assert quaternion.allclose(np.nan * a, np.nan * a) == False
    assert quaternion.allclose(np.nan * a, np.nan * a, equal_nan=True) == True


@pytest.mark.parametrize("q", make_Qs())
def test_bad_conversions(q):
    with pytest.raises((TypeError, ValueError)):
        s = int(q)
    with pytest.raises((TypeError, ValueError)):
        s = float(q)
    with pytest.raises((TypeError, ValueError)):
        a = np.zeros(3, dtype=int)
        a[0] = q
    with pytest.raises((TypeError, ValueError)):
        a = np.zeros(3)
        a[0] = q


def test_as_float_quat(Qs):
    qs = Qs[Qs_nonnan]
    for quats in [qs, np.vstack((qs,)*3), np.vstack((qs,)*(3*5)).reshape((3, 5)+qs.shape),
                  np.vstack((qs,)*(3*5*6)).reshape((3, 5, 6)+qs.shape)]:
        floats = quaternion.as_float_array(quats)
        assert floats.shape == quats.shape+(4,)
        assert allclose(quaternion.as_quat_array(floats), quats)
        assert allclose(quaternion.from_float_array(floats), quats)
        # Test that we can handle a list just like an array
        assert np.array_equal(quaternion.as_quat_array(floats), quaternion.as_quat_array(floats.tolist()))
    a = np.arange(12).reshape(3, 4)
    assert np.array_equal(quaternion.as_float_array(quaternion.as_quat_array(a)),
                          a.astype(float))
    assert quaternion.as_float_array(quaternion.x).ndim == 1


def test_vector_conversions():
    from quaternion import from_vector_part, as_vector_part
    v = np.random.rand(3)
    q = from_vector_part(v, vector_axis=-1)
    assert q.dtype == np.quaternion
    assert q.shape == v.shape[:-1]
    v2 = as_vector_part(q)
    assert np.array_equal(v, v2)
    v = np.random.rand(7, 3)
    q = from_vector_part(v, vector_axis=-1)
    assert q.dtype == np.quaternion
    assert q.shape == v.shape[:-1]
    v2 = as_vector_part(q)
    assert np.array_equal(v, v2)
    v = np.random.rand(18, 7, 3)
    q = from_vector_part(v, vector_axis=-1)
    assert q.dtype == np.quaternion
    assert q.shape == v.shape[:-1]
    v2 = as_vector_part(q)
    assert np.array_equal(v, v2)
    v = np.random.rand(18, 3, 7)
    q = from_vector_part(v, vector_axis=1)
    assert q.dtype == np.quaternion
    assert q.shape == v.shape[:1] + v.shape[2:]
    v2 = as_vector_part(q)
    assert np.array_equal(v, np.moveaxis(v2, -1, 1))


def test_as_rotation_matrix(Rs):
    def quat_mat(quat):
        return np.array([(quat * v * quat.inverse()).vec for v in [quaternion.x, quaternion.y, quaternion.z]]).T

    def quat_mat_vec(quats):
        mat_vec = np.array([quaternion.as_float_array(quats * v * np.reciprocal(quats))[..., 1:]
                            for v in [quaternion.x, quaternion.y, quaternion.z]])
        return np.transpose(mat_vec, tuple(range(mat_vec.ndim))[1:-1]+(-1, 0))

    with pytest.raises(ZeroDivisionError):
        quaternion.as_rotation_matrix(quaternion.zero)

    for R in Rs:
        # Test correctly normalized rotors:
        assert allclose(quat_mat(R), quaternion.as_rotation_matrix(R), atol=2*eps)
        # Test incorrectly normalized rotors:
        assert allclose(quat_mat(R), quaternion.as_rotation_matrix(1.1*R), atol=2*eps)

        for v in [quaternion.x, quaternion.y, quaternion.z]:
            assert allclose(
                quaternion.as_rotation_matrix(R) @ v.vec,
                quaternion.as_float_array(R * v * R.conjugate())[..., 1:],
                rtol=0, atol=2*eps
            )

    Rs0 = Rs.copy()
    Rs0[Rs.shape[0]//2] = quaternion.zero
    with pytest.raises(ZeroDivisionError):
        quaternion.as_rotation_matrix(Rs0)

    # Test correctly normalized rotors:
    assert allclose(quat_mat_vec(Rs), quaternion.as_rotation_matrix(Rs), atol=2*eps)
    # Test incorrectly normalized rotors:
    assert allclose(quat_mat_vec(Rs), quaternion.as_rotation_matrix(1.1*Rs), atol=2*eps)

    # Simply test that this function succeeds and returns the right shape
    assert quaternion.as_rotation_matrix(Rs.reshape((2, 5, 10))).shape == (2, 5, 10, 3, 3)


def test_from_rotation_matrix(Rs):
    try:
        from scipy import linalg
        have_linalg = True
    except ImportError:
        have_linalg = False

    for nonorthogonal in [True, False]:
        if nonorthogonal and have_linalg:
            rot_mat_eps = 10*eps
        else:
            rot_mat_eps = 5*eps

        for R1 in Rs:
            rot = quaternion.as_rotation_matrix(R1)
            R = quaternion.from_rotation_matrix(rot, nonorthogonal=nonorthogonal)
            for v in [quaternion.x, quaternion.y, quaternion.z]:
                assert allclose(
                    rot @ v.vec,
                    quaternion.as_float_array(R * v * R.conjugate())[..., 1:],
                    rtol=0, atol=rot_mat_eps
                )

        for i, R1 in enumerate(Rs):
            R2 = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(R1), nonorthogonal=nonorthogonal)
            d = quaternion.rotation_intrinsic_distance(R1, R2)
            assert d < rot_mat_eps, (i, R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

        Rs2 = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(Rs), nonorthogonal=nonorthogonal)
        for R1, R2 in zip(Rs, Rs2):
            d = quaternion.rotation_intrinsic_distance(R1, R2)
            assert d < rot_mat_eps, (R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

        Rs3 = Rs.reshape((2, 5, 10))
        Rs4 = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(Rs3))
        for R3, R4 in zip(Rs3.flatten(), Rs4.flatten()):
            d = quaternion.rotation_intrinsic_distance(R3, R4)
            assert d < rot_mat_eps, (R3, R4, d)  # Can't use allclose here; we don't care about rotor sign


def test_as_rotation_vector():
    np.random.seed(1234)
    n_tests = 1000
    vecs = np.random.uniform(high=math.pi/math.sqrt(3), size=n_tests*3).reshape((n_tests, 3))
    quats = np.zeros(vecs.shape[:-1]+(4,))
    quats[..., 1:] = vecs[...]
    quats = quaternion.as_quat_array(quats)
    quats = np.exp(quats/2)
    quat_vecs = quaternion.as_rotation_vector(quats)
    assert allclose(quat_vecs, vecs)


def test_from_rotation_vector():
    np.random.seed(1234)
    n_tests = 1000
    vecs = np.random.uniform(high=math.pi/math.sqrt(3), size=n_tests*3).reshape((n_tests, 3))
    quats = np.zeros(vecs.shape[:-1]+(4,))
    quats[..., 1:] = vecs[...]
    quats = quaternion.as_quat_array(quats)
    quats = np.exp(quats/2)
    quat_vecs = quaternion.as_rotation_vector(quats)
    quats2 = quaternion.from_rotation_vector(quat_vecs)
    assert allclose(quats, quats2)


def test_rotate_vectors(Rs):
    np.random.seed(1234)
    # Test (1)*(1)
    vecs = np.random.rand(3)
    quats = quaternion.z
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    assert allclose(vecsprime,
                       (quats * quaternion.quaternion(*vecs) * quats.inverse()).vec,
                       rtol=0.0, atol=0.0)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (1)*(5)
    vecs = np.random.rand(5, 3)
    quats = quaternion.z
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    for i, vec in enumerate(vecs):
        assert allclose(vecsprime[i],
                        (quats * quaternion.quaternion(*vec) * quats.inverse()).vec,
                        rtol=0.0, atol=0.0)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (1)*(5) inner axis
    vecs = np.random.rand(3, 5)
    quats = quaternion.z
    vecsprime = quaternion.rotate_vectors(quats, vecs, axis=-2)
    for i, vec in enumerate(vecs.T):
        assert allclose(vecsprime[:, i],
                        (quats * quaternion.quaternion(*vec) * quats.inverse()).vec,
                        rtol=0.0, atol=0.0)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(1)
    vecs = np.random.rand(3)
    quats = Rs
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    assert allclose(vecsprime,
                    [vprime.vec for vprime in quats * quaternion.quaternion(*vecs) * ~quats],
                    rtol=1e-15, atol=1e-15)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(5)
    vecs = np.random.rand(5, 3)
    quats = Rs
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    for i, vec in enumerate(vecs):
        assert allclose(vecsprime[:, i],
                        [vprime.vec for vprime in quats * quaternion.quaternion(*vec) * ~quats],
                        rtol=1e-15, atol=1e-15)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(5) inner axis
    vecs = np.random.rand(3, 5)
    quats = Rs
    vecsprime = quaternion.rotate_vectors(quats, vecs, axis=-2)
    for i, vec in enumerate(vecs.T):
        assert allclose(vecsprime[:, :, i],
                        [vprime.vec for vprime in quats * quaternion.quaternion(*vec) * ~quats],
                        rtol=1e-15, atol=1e-15)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)

    for Rshape in [(1,), (10,), (100,), (1000,), (5, 7), (5, 7, 23)]:
        R = np.random.normal(size=Rshape+(4,))
        R = quaternion.from_float_array(R / np.linalg.norm(R, axis=-1)[..., np.newaxis])
        for vshape in [(1,), (2,), (3,), (4,), (20,), (200,), (2000,), (11, 13), (11, 13, 29)]:
            v = np.random.normal(size=vshape+(3,))
            Rprime = quaternion.rotate_vectors(R, v)
            expected_shape = Rshape + vshape + (3,)
            assert Rprime.shape == expected_shape
        for vshape, axis in [((7, 3, 5), 1), ((7, 3, 5), -2), ((7, 3, 5, 11), 1), ((7, 3, 5, 11), -3)]:
            v = np.random.normal(size=vshape)
            Rprime = quaternion.rotate_vectors(R, v, axis=axis)
            expected_shape = Rshape + vshape
            assert Rprime.shape == expected_shape


def test_allclose(Qs):
    for q in Qs[Qs_nonnan]:
        assert quaternion.allclose(q, q, rtol=0.0, atol=0.0)
    assert quaternion.allclose(Qs[Qs_nonnan], Qs[Qs_nonnan], rtol=0.0, atol=0.0)

    for q in Qs[Qs_finitenonzero]:
        assert quaternion.allclose(q, q*(1+1e-13), rtol=1.1e-13, atol=0.0)
        assert ~quaternion.allclose(q, q*(1+1e-13), rtol=0.9e-13, atol=0.0)
        for e in [quaternion.one, quaternion.x, quaternion.y, quaternion.z]:
            assert quaternion.allclose(q, q+(1e-13*e), rtol=0.0, atol=1.1e-13)
            assert ~quaternion.allclose(q, q+(1e-13*e), rtol=0.0, atol=0.9e-13)
    assert quaternion.allclose(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero]*(1+1e-13), rtol=1.1e-13, atol=0.0)
    assert ~quaternion.allclose(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero]*(1+1e-13), rtol=0.9e-13, atol=0.0)
    for e in [quaternion.one, quaternion.x, quaternion.y, quaternion.z]:
        assert quaternion.allclose(Qs[Qs_finite], Qs[Qs_finite]+(1e-13*e), rtol=0.0, atol=1.1e-13)
        assert ~quaternion.allclose(Qs[Qs_finite], Qs[Qs_finite]+(1e-13*e), rtol=0.0, atol=0.9e-13)
    assert quaternion.allclose(Qs[Qs_zero], Qs[Qs_zero]*2, rtol=0.0, atol=1.1e-13)

    for qnan in Qs[Qs_nan]:
        assert ~quaternion.allclose(qnan, qnan, rtol=1.0, atol=1.0)
        for q in Qs:
            assert ~quaternion.allclose(q, qnan, rtol=1.0, atol=1.0)


def test_from_spherical_coords():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        q = quaternion.from_spherical_coords(vartheta, varphi)
        assert abs((np.quaternion(0, 0, 0, varphi / 2.).exp() * np.quaternion(0, 0, vartheta / 2., 0).exp())
                   - q) < 1.e-15
        xprime = q * quaternion.x * q.inverse()
        yprime = q * quaternion.y * q.inverse()
        zprime = q * quaternion.z * q.inverse()
        nhat = np.quaternion(0.0, math.sin(vartheta)*math.cos(varphi), math.sin(vartheta)*math.sin(varphi),
                             math.cos(vartheta))
        thetahat = np.quaternion(0.0, math.cos(vartheta)*math.cos(varphi), math.cos(vartheta)*math.sin(varphi),
                                 -math.sin(vartheta))
        phihat = np.quaternion(0.0, -math.sin(varphi), math.cos(varphi), 0.0)
        assert abs(xprime - thetahat) < 1.e-15
        assert abs(yprime - phihat) < 1.e-15
        assert abs(zprime - nhat) < 1.e-15
    assert np.max(np.abs(quaternion.from_spherical_coords(random_angles)
                         - np.array([quaternion.from_spherical_coords(vartheta, varphi)
                                     for vartheta, varphi in random_angles]))) < 1.e-15


def test_as_spherical_coords(Rs):
    np.random.seed(1843)
    # First test on rotors that are precisely spherical-coordinate rotors
    random_angles = [[np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        vartheta2, varphi2 = quaternion.as_spherical_coords(quaternion.from_spherical_coords(vartheta, varphi))
        varphi2 = (varphi2 + 2*np.pi) if varphi2 < 0 else varphi2
        assert abs(vartheta - vartheta2) < 1e-12, ((vartheta, varphi), (vartheta2, varphi2))
        assert abs(varphi - varphi2) < 1e-12, ((vartheta, varphi), (vartheta2, varphi2))
    # Now test that arbitrary rotors rotate z to the appropriate location
    for R in Rs:
        vartheta, varphi = quaternion.as_spherical_coords(R)
        R2 = quaternion.from_spherical_coords(vartheta, varphi)
        assert (R*quaternion.z*R.inverse() - R2*quaternion.z*R2.inverse()).abs() < 4e-15, (R, R2, (vartheta, varphi))


def test_from_euler_angles():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        assert abs((np.quaternion(0, 0, 0, alpha / 2.).exp()
                    * np.quaternion(0, 0, beta / 2., 0).exp()
                    * np.quaternion(0, 0, 0, gamma / 2.).exp()
                   )
                   - quaternion.from_euler_angles(alpha, beta, gamma)) < 1.e-15
    assert np.max(np.abs(quaternion.from_euler_angles(random_angles)
                         - np.array([quaternion.from_euler_angles(alpha, beta, gamma)
                                     for alpha, beta, gamma in random_angles]))) < 1.e-15


def test_as_euler_angles():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        R1 = quaternion.from_euler_angles(alpha, beta, gamma)
        R2 = quaternion.from_euler_angles(*list(quaternion.as_euler_angles(R1)))
        d = quaternion.rotation_intrinsic_distance(R1, R2)
        assert d < 6e3*eps, ((alpha, beta, gamma), R1, R2, d)  # Can't use allclose here; we don't care about rotor sign
    q0 = quaternion.quaternion(0, 0.6, 0.8, 0)
    assert q0.norm() == 1.0
    assert abs(q0 - quaternion.from_euler_angles(*list(quaternion.as_euler_angles(q0)))) < 1.e-15


# Unary bool returners
def test_quaternion_nonzero(Qs):
    assert not Qs[q_0].nonzero()  # Do this one explicitly, to not use circular logic
    assert Qs[q_1].nonzero()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_zero]:
        assert not q.nonzero()
    for q in Qs[Qs_nonzero]:
        assert q.nonzero()


def test_quaternion_isnan(Qs):
    assert not Qs[q_0].isnan()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isnan()  # Do this one explicitly, to not use circular logic
    assert Qs[q_nan1].isnan()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nan]:
        assert q.isnan()
    for q in Qs[Qs_nonnan]:
        assert not q.isnan()


def test_quaternion_isinf(Qs):
    assert not Qs[q_0].isinf()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isinf()  # Do this one explicitly, to not use circular logic
    assert Qs[q_inf1].isinf()  # Do this one explicitly, to not use circular logic
    assert Qs[q_minf1].isinf()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_inf]:
        assert q.isinf()
    for q in Qs[Qs_noninf]:
        assert not q.isinf()


def test_quaternion_isfinite(Qs):
    assert not Qs[q_nan1].isfinite()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_inf1].isfinite()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_minf1].isfinite()  # Do this one explicitly, to not use circular logic
    assert Qs[q_0].isfinite()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nonfinite]:
        assert not q.isfinite()
    for q in Qs[Qs_finite]:
        assert q.isfinite()


# Binary bool returners
def test_quaternion_equal(Qs):
    for j in Qs_nonnan:
        assert Qs[j] == Qs[j]  # self equality
        for k in range(len(Qs)):  # non-self inequality
            assert (j == k) or (not (Qs[j] == Qs[k]))
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert not q == p  # nan should never equal anything


def test_quaternion_not_equal(Qs):
    for j in Qs_nonnan:
        assert not (Qs[j] != Qs[j])  # self non-not_equality
        for k in Qs_nonnan:  # non-self not_equality
            assert (j == k) or (Qs[j] != Qs[k])
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert q != p  # nan should never equal anything


def test_quaternion_richcompare(Qs):
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
    for p in [Qs[Qlog], Qs[Qexp]]:
        assert Qs[q_1] < p
        assert Qs[q_1] <= p
        assert p.greater(Qs[q_1])
        assert p.greater_equal(Qs[q_1])


# Unary float returners
def test_quaternion_absolute(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.abs())
    for q in Qs[Qs_inf]:
        if on_windows:
            assert np.isinf(q.abs()) or np.isnan(q.abs())
        else:
            assert np.isinf(q.abs())
    for q, a in [(Qs[q_0], 0.0), (Qs[q_1], 1.0), (Qs[x], 1.0), (Qs[y], 1.0), (Qs[z], 1.0),
                 (Qs[Q], np.sqrt(Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2)),
                 (Qs[Qbar], np.sqrt(Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2))]:
        assert np.allclose(q.abs(), a)


def test_quaternion_norm(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.norm())
    for q in Qs[Qs_inf]:
        if on_windows:
            assert np.isinf(q.norm()) or np.isnan(q.norm())
        else:
            assert np.isinf(q.norm())
    for q, a in [(Qs[q_0], 0.0), (Qs[q_1], 1.0), (Qs[x], 1.0), (Qs[y], 1.0), (Qs[z], 1.0),
                 (Qs[Q], Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2),
                 (Qs[Qbar], Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2)]:
        assert np.allclose(q.norm(), a)


# Unary quaternion returners
def test_quaternion_negative(Qs):
    assert -Qs[Q] == Qs[Qneg]
    for q in Qs[Qs_finite]:
        assert -q == -1.0 * q
    for q in Qs[Qs_nonnan]:
        assert -(-q) == q


def test_quaternion_conjugate(Qs):
    assert Qs[Q].conjugate() == Qs[Qbar]
    for q in Qs[Qs_nonnan]:
        assert q.conjugate() == q.conj()
        assert q.conjugate().conjugate() == q
        c = q.conjugate()
        assert c.w == q.w
        assert c.x == -q.x
        assert c.y == -q.y
        assert c.z == -q.z


def test_quaternion_sqrt(Qs):
    sqrt_precision = 2.e-15
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q.sqrt() * q.sqrt(), q, rtol=sqrt_precision)
        # Ensure that non-unit quaternions are handled correctly
        for s in [1, -1, 2, -2, 3.4, -3.4]:
            for r in [1, quaternion.x, quaternion.y, quaternion.z]:
                srq = s*r*q
                assert allclose(srq.sqrt() * srq.sqrt(), srq, rtol=sqrt_precision)
    # Ensure that inputs close to zero are handled gracefully
    sqrt_dbl_min = math.sqrt(np.finfo(float).tiny)
    assert quaternion.quaternion(0, 0, 0, 2e-8*sqrt_dbl_min).sqrt() == quaternion.quaternion(0, 0, 0, 0)
    assert quaternion.quaternion(0, 0, 0, 0.9999*sqrt_dbl_min).sqrt() == quaternion.quaternion(0, 0, 0, 0)
    assert quaternion.quaternion(0, 0, 0, 1e-16*sqrt_dbl_min).sqrt() == quaternion.quaternion(0, 0, 0, 0)
    assert quaternion.quaternion(0, 0, 0, 1.1*sqrt_dbl_min).sqrt() != quaternion.quaternion(0, 0, 0, 0)


def test_quaternion_square(Qs):
    square_precision = 1.e-15
    for q in Qs[Qs_finite]:
        assert np.norm(q*q - q**2) < square_precision
        a = np.array([q])
        assert np.norm(a**2 - np.array([q**2])) < square_precision


def test_quaternion_log_exp(Qs):
    qlogexp_precision = 4.e-15
    assert (Qs[Q].log() - Qs[Qlog]).abs() < qlogexp_precision
    assert (Qs[Q].exp() - Qs[Qexp]).abs() < qlogexp_precision
    assert (Qs[Q].log().exp() - Qs[Q]).abs() < qlogexp_precision
    assert (Qs[Q].exp().log() - Qs[Q]).abs() > qlogexp_precision  # Note order of operations!
    assert quaternion.one.log() == quaternion.zero
    assert quaternion.x.log() == (np.pi / 2) * quaternion.x
    assert quaternion.y.log() == (np.pi / 2) * quaternion.y
    assert quaternion.z.log() == (np.pi / 2) * quaternion.z
    assert (-quaternion.one).log() == (np.pi) * quaternion.x
    strict_assert(False)  # logs of interesting scalars * basis vectors
    strict_assert(False)  # logs of negative scalars


def test_angle(Rs):
    angle_precision = 4.e-15
    unit_vecs = [quaternion.x, quaternion.y, quaternion.z,
                 -quaternion.x, -quaternion.y, -quaternion.z]
    for u in unit_vecs:
        for theta in linspace(-2 * np.pi, 2 * np.pi, num=50):
            assert abs((theta * u / 2).exp().angle() - abs(theta)) < angle_precision


def test_quaternion_normalized(Qs):
    assert abs(Qs[Q].normalized()-Qs[Qnormalized]) < 4e-16
    for q in Qs[Qs_finitenonzero]:
        assert abs(q.normalized().abs() - 1.0) < 1.e-15


def test_quaternion_parity_conjugates(Qs):
    for q in Qs[Qs_finite]:
        assert q.x_parity_conjugate() == np.quaternion(q.w, q.x, -q.y, -q.z)
        assert q.y_parity_conjugate() == np.quaternion(q.w, -q.x, q.y, -q.z)
        assert q.z_parity_conjugate() == np.quaternion(q.w, -q.x, -q.y, q.z)
        assert q.parity_conjugate() == np.quaternion(q.w, q.x, q.y, q.z)
    assert np.array_equal(np.x_parity_conjugate(Qs[Qs_finite]),
                          np.array([q.x_parity_conjugate() for q in Qs[Qs_finite]]))
    assert np.array_equal(np.y_parity_conjugate(Qs[Qs_finite]),
                          np.array([q.y_parity_conjugate() for q in Qs[Qs_finite]]))
    assert np.array_equal(np.z_parity_conjugate(Qs[Qs_finite]),
                          np.array([q.z_parity_conjugate() for q in Qs[Qs_finite]]))
    assert np.array_equal(np.parity_conjugate(Qs[Qs_finite]), np.array([q.parity_conjugate() for q in Qs[Qs_finite]]))


# Quaternion-quaternion binary quaternion returners
@pytest.mark.xfail
def test_quaternion_copysign(Qs):
    assert False


# Quaternion-quaternion, scalar-quaternion, or quaternion-scalar binary quaternion returners
def test_quaternion_add(Qs):
    for j in Qs_nonnan:
        for k in Qs_nonnan:
            q = Qs[j]
            p = Qs[k]
            assert (q + p == quaternion.quaternion(q.w + p.w, q.x + p.x, q.y + p.y, q.z + p.z)
                    or (j == q_inf1 and k == q_minf1)
                    or (k == q_inf1 and j == q_minf1))
    for q in Qs[Qs_nonnan]:
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert (q + s == quaternion.quaternion(q.w + s, q.x, q.y, q.z))
            assert (s + q == quaternion.quaternion(q.w + s, q.x, q.y, q.z))


def test_quaternion_add_ufunc(Qs):
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.add)


def test_quaternion_subtract(Qs):
    for q in Qs[Qs_finite]:
        for p in Qs[Qs_finite]:
            assert q - p == quaternion.quaternion(q.w - p.w, q.x - p.x, q.y - p.y, q.z - p.z)
    for q in Qs[Qs_nonnan]:
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert (q - s == quaternion.quaternion(q.w - s, q.x, q.y, q.z))
            assert (s - q == quaternion.quaternion(s - q.w, -q.x, -q.y, -q.z))


def test_quaternion_subtract_ufunc(Qs):
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.sub)


def test_quaternion_multiply(Qs):
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
            assert q * s == quaternion.quaternion(s * q.w, s * q.x, s * q.y, s * q.z)
            assert s * q == q * s
    for q in Qs[Qs_finite]:
        assert 0.0 * q == Qs[q_0]
        assert 0.0 * q == q * 0.0

    # Check linearity
    for q1 in Qs[Qs_finite]:
        for q2 in Qs[Qs_finite]:
            for q3 in Qs[Qs_finite]:
                assert allclose(q1*(q2+q3), (q1*q2)+(q1*q3))
                assert allclose((q1+q2)*q3, (q1*q3)+(q2*q3))

    # Check the multiplication table
    for q in [Qs[q_1], Qs[x], Qs[y], Qs[z]]:
        assert Qs[q_1] * q == q
        assert q * Qs[q_1] == q
    assert Qs[x] * Qs[x] == -Qs[q_1]
    assert Qs[x] * Qs[y] == Qs[z]
    assert Qs[x] * Qs[z] == -Qs[y]
    assert Qs[y] * Qs[x] == -Qs[z]
    assert Qs[y] * Qs[y] == -Qs[q_1]
    assert Qs[y] * Qs[z] == Qs[x]
    assert Qs[z] * Qs[x] == Qs[y]
    assert Qs[z] * Qs[y] == -Qs[x]
    assert Qs[z] * Qs[z] == -Qs[q_1]


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


def test_quaternion_divide(Qs):
    # Check identity between "inverse" and "reciprocal"
    for q in Qs[Qs_finitenonzero]:
        assert q.inverse() == q.reciprocal()

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


def test_quaternion_power(Qs):
    import math
    qpower_precision = 4*eps

    # Test equivalence between scalar and real-quaternion exponentiation
    for b in [0, 0.0, 1, 1.0, 2, 2.0, 5.6]:
        for e in [0, 0.0, 1, 1.0, 2, 2.0, 4.5]:
            be = np.quaternion(b**e, 0, 0, 0)
            assert allclose(be, np.quaternion(b, 0, 0, 0)**np.quaternion(e, 0, 0, 0), rtol=qpower_precision)
            assert allclose(be, b**np.quaternion(e, 0, 0, 0), rtol=qpower_precision)
            assert allclose(be, np.quaternion(b, 0, 0, 0)**e, rtol=qpower_precision)
    for q in [-3*quaternion.one, -2*quaternion.one, -quaternion.one, quaternion.zero, quaternion.one, 3*quaternion.one]:
        for s in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
            for t in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
                assert allclose((s*t)**q, (s**q)*(t**q), rtol=2*qpower_precision)

    # Test basic integer-exponent and additive-exponent properties
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q ** 0, np.quaternion(1, 0, 0, 0), rtol=qpower_precision)
        assert allclose(q ** 0.0, np.quaternion(1, 0, 0, 0), rtol=qpower_precision)
        assert allclose(q ** np.quaternion(0, 0, 0, 0), np.quaternion(1, 0, 0, 0), rtol=qpower_precision)
        assert allclose(((q ** 0.5) * (q ** 0.5)), q, rtol=qpower_precision)
        assert allclose(q ** 1.0, q, rtol=qpower_precision)
        assert allclose(q ** 1, q, rtol=qpower_precision)
        assert allclose(q ** np.quaternion(1, 0, 0, 0), q, rtol=qpower_precision)
        assert allclose(q ** 2.0, q * q, rtol=qpower_precision)
        assert allclose(q ** 2, q * q, rtol=qpower_precision)
        assert allclose(q ** np.quaternion(2, 0, 0, 0), q * q, rtol=qpower_precision)
        assert allclose(q ** 3, q * q * q, rtol=qpower_precision)
        assert allclose(q ** -1, q.inverse(), rtol=qpower_precision)
        assert allclose(q ** -1.0, q.inverse(), rtol=qpower_precision)
        for s in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
            for t in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
                assert allclose(q**(s+t), (q**s)*(q**t), rtol=2*qpower_precision)
                assert allclose(q**(s-t), (q**s)/(q**t), rtol=2*qpower_precision)

    # Check that exp(q) is the same as e**q
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q.exp(), math.e**q, rtol=qpower_precision)
        for s in [0, 0., 1.0, 1, 1.2, 2.3, 3]:
            for t in [0, 0., 1.0, 1, 1.2, 2.3, 3]:
                assert allclose((s*t)**q, (s**q)*(t**q), rtol=3*qpower_precision)
        for s in [1.0, 1, 1.2, 2.3, 3]:
            assert allclose(s**q, (q*math.log(s)).exp(), rtol=qpower_precision)

    qinverse_precision = 2*eps
    for q in Qs[Qs_finitenonzero]:
        assert allclose((q ** -1.0) * q, Qs[q_1], rtol=qinverse_precision)
    for q in Qs[Qs_finitenonzero]:
        assert allclose((q ** -1) * q, Qs[q_1], rtol=qinverse_precision)
    for q in Qs[Qs_finitenonzero]:
        assert allclose((q ** Qs[q_1]), q, rtol=qpower_precision)
    strict_assert(False)  # Try more edge cases

    for q in [quaternion.x, quaternion.y, quaternion.z]:
        assert allclose(quaternion.quaternion(math.exp(-math.pi/2), 0, 0, 0),
                        q**q, rtol=qpower_precision)
    assert allclose(quaternion.quaternion(math.cos(math.pi/2), 0, 0, math.sin(math.pi/2)),
                    quaternion.x**quaternion.y, rtol=qpower_precision)
    assert allclose(quaternion.quaternion(math.cos(math.pi/2), 0, -math.sin(math.pi/2), 0),
                    quaternion.x**quaternion.z, rtol=qpower_precision)
    assert allclose(quaternion.quaternion(math.cos(math.pi/2), 0, 0, -math.sin(math.pi/2)),
                    quaternion.y**quaternion.x, rtol=qpower_precision)
    assert allclose(quaternion.quaternion(math.cos(math.pi/2), math.sin(math.pi/2), 0, 0),
                    quaternion.y**quaternion.z, rtol=qpower_precision)
    assert allclose(quaternion.quaternion(math.cos(math.pi/2), 0, math.sin(math.pi/2), 0),
                    quaternion.z**quaternion.x, rtol=qpower_precision)
    assert allclose(quaternion.quaternion(math.cos(math.pi/2), -math.sin(math.pi/2), 0, 0),
                    quaternion.z**quaternion.y, rtol=qpower_precision)



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
        assert np.array_equal(q.components, np.array([q.w, q.x, q.y, q.z]))
        assert np.array_equal(q.vec, np.array([q.x, q.y, q.z]))
        assert np.array_equal(q.imag, np.array([q.x, q.y, q.z]))
    # set components/vec from np.array, list, tuple
    for q in Qs[Qs_nonnan]:
        for seq_type in [np.array, list, tuple]:
            p = np.quaternion(*q.components)
            r = np.quaternion(*q.components)
            s = np.quaternion(*q.components)
            p.components = seq_type((-5.5, 6.6, -7.7, 8.8))
            r.vec = seq_type((6.6, -7.7, 8.8))
            s.imag = seq_type((6.6, -7.7, 8.8))
            assert np.array_equal(p.components, np.array([-5.5, 6.6, -7.7, 8.8]))
            assert np.array_equal(r.components, np.array([q.w, 6.6, -7.7, 8.8]))
            assert np.array_equal(s.components, np.array([q.w, 6.6, -7.7, 8.8]))
    # TypeError when setting components with the wrong type or size of thing
    for q in Qs:
        for seq_type in [np.array, list, tuple]:
            p = np.quaternion(*q.components)
            r = np.quaternion(*q.components)
            s = np.quaternion(*q.components)
            with pytest.raises(TypeError):
                p.components = '1.1, 2.2, 3.3, 4.4'
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
                r.vec = '2.2, 3.3, 4.4'
            with pytest.raises(TypeError):
                r.vec = seq_type([])
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5,))
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5, 6.6))
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5, 6.6, -7.7, 8.8))
            with pytest.raises(TypeError):
                s.vec = '2.2, 3.3, 4.4'
            with pytest.raises(TypeError):
                s.vec = seq_type([])
            with pytest.raises(TypeError):
                s.vec = seq_type((-5.5,))
            with pytest.raises(TypeError):
                s.vec = seq_type((-5.5, 6.6))
            with pytest.raises(TypeError):
                s.vec = seq_type((-5.5, 6.6, -7.7, 8.8))


def test_metrics(Rs):
    metric_precision = 4.e-15
    intrinsic_funcs = (quaternion.rotor_intrinsic_distance, quaternion.rotation_intrinsic_distance)
    chordal_funcs = (quaternion.rotor_chordal_distance, quaternion.rotation_chordal_distance)
    metric_funcs = intrinsic_funcs + chordal_funcs
    rotor_funcs = (quaternion.rotor_intrinsic_distance, quaternion.rotor_chordal_distance)
    rotation_funcs = (quaternion.rotation_intrinsic_distance, quaternion.rotation_chordal_distance)
    distance_dict = {func: func(Rs, Rs[:, np.newaxis]) for func in metric_funcs}

    # Check non-negativity
    for mat in distance_dict.values():
        assert (mat >= 0.).all()

    # Check discernibility
    for func in metric_funcs:
        if func in chordal_funcs:
            eps = 0
        else:
            eps = 5.e-16
        if func in rotor_funcs:
            target = Rs != Rs[:, np.newaxis]
        else:
            target = np.logical_and(Rs != Rs[:, np.newaxis], Rs != - Rs[:, np.newaxis])
        assert ((distance_dict[func] > eps) == target).all()

    # Check symmetry
    for mat in distance_dict.values():
        assert np.allclose(mat, mat.T, atol=metric_precision, rtol=0)

    # Check triangle inequality
    for mat in distance_dict.values():
        assert ((mat - metric_precision)[:, np.newaxis, :] <= mat[:, :, np.newaxis] + mat).all()

    # Check distances from self or -self
    for func in metric_funcs:
        # All distances from self should be 0.0
        if func in chordal_funcs:
            eps = 0
        else:
            eps = 5.e-16
        assert (np.diag(distance_dict[func]) <= eps).all()

    # Chordal rotor distance from -self should be 2
    assert (abs(quaternion.rotor_chordal_distance(Rs, -Rs) - 2.0) < metric_precision).all()
    # Intrinsic rotor distance from -self should be 2pi
    assert (abs(quaternion.rotor_intrinsic_distance(Rs, -Rs) - 2.0 * np.pi) < metric_precision).all()
    # Rotation distances from -self should be 0
    assert (quaternion.rotation_chordal_distance(Rs, -Rs) == 0.0).all()
    assert (quaternion.rotation_intrinsic_distance(Rs, -Rs) < 5.e-16).all()

    # We expect the chordal distance to be smaller than the intrinsic distance (or equal, if the distance is zero)
    assert np.logical_or(quaternion.rotor_chordal_distance(quaternion.one, Rs)
                           < quaternion.rotor_intrinsic_distance(quaternion.one, Rs),
                         Rs == quaternion.one).all()
    # Check invariance under overall rotations: d(R1, R2) = d(R3*R1, R3*R2) = d(R1*R3, R2*R3)
    for func in quaternion.rotor_chordal_distance, quaternion.rotation_intrinsic_distance:
        rotations = Rs[:, np.newaxis] * Rs
        right_distances = func(rotations, rotations[:, np.newaxis])
        assert (abs(distance_dict[func][:, :, np.newaxis] - right_distances) < metric_precision).all()
        left_distances = func(rotations[:, :, np.newaxis], rotations[:, np.newaxis])
        assert (abs(distance_dict[func] - left_distances) < metric_precision).all()


def test_unflip_rotors(Rs):
    unflip_precision = 4e-16
    f = 2 * np.random.rand(17, 1_000, 4) - 1
    q = quaternion.as_quat_array(f)
    q = q / abs(q)
    ndim = q.ndim
    axis = -1
    inplace = False
    q_out = quaternion.unflip_rotors(q, axis=axis, inplace=inplace)
    diff = np.linalg.norm(np.diff(quaternion.as_float_array(q_out), axis=(axis % ndim)), axis=-1)
    assert np.sum(diff > 1.4142135623730950488016887242097) == 0
    q_in = np.vstack((Rs, -Rs))
    q_out = quaternion.unflip_rotors(q_in, axis=0, inplace=False)
    assert np.array_equal(q_out[1], Rs)
    q_in = np.vstack((Rs, -Rs))
    quaternion.unflip_rotors(q_in, axis=0, inplace=True)
    assert np.array_equal(q_in[1], Rs)


def test_slerp(Rs):
    from quaternion import slerp_evaluate, slerp, allclose
    slerp_precision = 4.e-15
    ones = [quaternion.one, quaternion.x, quaternion.y, quaternion.z, -quaternion.x, -quaternion.y, -quaternion.z]
    # Check extremes
    for Q1 in ones:
        assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, Q1, 0.0), Q1) < slerp_precision
        assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, Q1, 1.0), Q1) < slerp_precision
        assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, -Q1, 0.0), Q1) < slerp_precision
        assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, -Q1, 1.0), Q1) < slerp_precision
        for Q2 in ones:
            assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, Q2, 0.0), Q1) < slerp_precision
            assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, Q2, 1.0), Q2) < slerp_precision
            assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, -Q2, 0.0), Q1) < slerp_precision
            assert quaternion.rotation_chordal_distance(slerp_evaluate(Q1, -Q2, 1.0), -Q2) < slerp_precision
            assert quaternion.rotation_chordal_distance(slerp_evaluate(Q2, Q1, 0.0), Q2) < slerp_precision
            assert quaternion.rotation_chordal_distance(slerp_evaluate(Q2, Q1, 1.0), Q1) < slerp_precision
    # Test simple increases in each dimension
    for Q2 in ones[1:]:
        for t in np.linspace(0.0, 1.0, num=100, endpoint=True):
            assert quaternion.rotation_chordal_distance(
                slerp_evaluate(quaternion.one, Q2, t),
                (np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2)
            ) < slerp_precision
        t = np.linspace(0.0, 1.0, num=100, endpoint=True)
        assert allclose(
            slerp(quaternion.one, Q2, 0.0, 1.0, t),
            np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2,
            rtol=slerp_precision
        )
        assert allclose(
            slerp(quaternion.one, Q2, -10.0, 20.0, 30 * t - 10.0),
            np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2,
            rtol=slerp_precision
        )
        t = 1.5 * t - 0.125
        assert allclose(
            slerp(quaternion.one, Q2, 0.0, 1.0, t),
            np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2,
            rtol=slerp_precision
        )
    # Test that slerp(rotate(rotors)) equals rotate(slerp(rotors))
    for R in Rs:
        for Q2 in ones[1:]:
            for t in np.linspace(0.0, 1.0, num=100, endpoint=True):
                assert quaternion.rotation_chordal_distance(
                    R * slerp_evaluate(quaternion.one, Q2, t),
                    slerp_evaluate(R * quaternion.one, R * Q2, t)
                ) < slerp_precision
            t = np.linspace(0.0, 1.0, num=100, endpoint=True)
            assert allclose(
                R * slerp(quaternion.one, Q2, 0.0, 1.0, t),
                slerp(R * quaternion.one, R * Q2, 0.0, 1.0, t),
                rtol=slerp_precision
            )


@pytest.mark.skipif(os.environ.get('FAST'), reason="Takes ~2 seconds")
def test_squad(Rs):
    from quaternion import slerp_evaluate
    np.random.seed(1234)
    squad_precision = 4.e-15
    ones = [quaternion.one, quaternion.x, quaternion.y, quaternion.z, -quaternion.x, -quaternion.y, -quaternion.z]
    t_in = np.linspace(0.0, 1.0, num=13, endpoint=True)
    t_out = np.linspace(0.0, 1.0, num=37, endpoint=True)
    t_out2 = np.array(sorted([np.random.uniform(0.0, 1.0) for i in range(59)]))
    # squad interpolated onto the inputs should be the identity
    for R1 in Rs:
        for R2 in Rs:
            R_in = np.array([slerp_evaluate(R1, R2, t) for t in t_in])
            assert np.all(np.abs(quaternion.squad(R_in, t_in, t_in) - R_in) < squad_precision)
    # squad should be the same as slerp for linear interpolation
    for R in ones:
        R_in = np.array([slerp_evaluate(quaternion.one, R, t) for t in t_in])
        R_out_squad = quaternion.squad(R_in, t_in, t_out)
        R_out_slerp = np.array([slerp_evaluate(quaternion.one, R, t) for t in t_out])
        # print(
        #     R, "\n",
        #     np.argmax(np.abs(R_out_squad - R_out_slerp)),
        #     len(R_out_squad), "\n",
        #     np.max(np.abs(R_out_squad - R_out_slerp)), "\n",
        #     R_out_squad[-6:], "\n",
        #     R_out_slerp[-6:],
        # )
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision), (
            R,
            np.argmax(np.abs(R_out_squad - R_out_slerp)),
            len(R_out_squad),
            R_out_squad[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
            R_out_slerp[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
        )
        R_out_squad = quaternion.squad(R_in, t_in, t_out2)
        R_out_slerp = np.array([slerp_evaluate(quaternion.one, R, t) for t in t_out2])
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision)
        # assert False # Test unequal input time steps, and correct squad output [0,-2,-1]

    for i in range(len(ones)):
        R3 = np.roll(ones, i)[:3]
        R_in = np.array([[slerp_evaluate(quaternion.one, R, t) for R in R3] for t in t_in])
        R_out_squad = quaternion.squad(R_in, t_in, t_out)
        R_out_slerp = np.array([[slerp_evaluate(quaternion.one, R, t) for R in R3] for t in t_out])
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision), (
            R,
            np.argmax(np.abs(R_out_squad - R_out_slerp)),
            len(R_out_squad),
            R_out_squad[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
            R_out_slerp[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
        )
        R_out_squad = quaternion.squad(R_in, t_in, t_out2)
        R_out_slerp = np.array([[slerp_evaluate(quaternion.one, R, t) for R in R3] for t in t_out2])
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision)


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
        Ps[j] = np.quaternion(1.3, 2.4, 3.5, 4.7)
        for k in range(j + 1):
            assert Ps[k] == np.quaternion(1.3, 2.4, 3.5, 4.7)
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
            Ps[0] = seq_type((1.3, 2.4, 3.5, 4.7, 5.9, np.nan))
        for j in range(len(Ps)):
            Ps[j] = seq_type((1.3, 2.4, 3.5, 4.7))
            for k in range(j + 1):
                assert Ps[k] == np.quaternion(1.3, 2.4, 3.5, 4.7)
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


def test_ufuncs(Rs, Qs):
    np.random.seed(1234)
    assert allclose(np.abs(Rs), np.ones(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(np.abs(np.log(Rs) - np.array([r.log() for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                    rtol=1.e-15)
    assert allclose(np.abs(np.exp(Rs) - np.array([r.exp() for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                    rtol=1.e-15)
    assert allclose(np.abs(Rs - Rs), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(np.abs(Rs + (-Rs)), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(np.abs(np.conjugate(Rs) - np.array([r.conjugate() for r in Rs])), np.zeros(Rs.shape),
                    atol=1.e-14, rtol=1.e-15)
    assert np.all(Rs == Rs)
    assert np.all(Rs <= Rs)
    for i in range(10):
        x = np.random.uniform(-10, 10)
        assert allclose(np.abs(Rs * x - np.array([r * x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        # assert allclose( np.abs( x*Rs - np.array([r*x for r in Rs]) ), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        strict_assert(False)
        assert allclose(np.abs(Rs / x - np.array([r / x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        assert allclose(np.abs(Rs ** x - np.array([r ** x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                        rtol=1.e-15)
    assert allclose(
        np.abs(Qs[Qs_finite] + Qs[Qs_finite] - np.array([q1 + q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(
        np.abs(Qs[Qs_finite] - Qs[Qs_finite] - np.array([q1 - q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(
        np.abs(Qs[Qs_finite] * Qs[Qs_finite] - np.array([q1 * q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    for Q in Qs[Qs_finite]:
        assert allclose(np.abs(Qs[Qs_finite] * Q - np.array([q1 * Q for q1 in Qs[Qs_finite]])),
                           np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
        # assert allclose( np.abs( Q*Qs[Qs_finite] - np.array([Q*q1 for q1 in Qs[Qs_finite]]) ),
        # np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(np.abs(Qs[Qs_finitenonzero] / Qs[Qs_finitenonzero]
                           - np.array([q1 / q2 for q1, q2 in zip(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero])])),
                    np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(np.abs(Qs[Qs_finitenonzero] ** Qs[Qs_finitenonzero]
                           - np.array([q1 ** q2 for q1, q2 in zip(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero])])),
                    np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)
    assert allclose(np.abs(~Qs[Qs_finitenonzero]
                           - np.array([q.inverse() for q in Qs[Qs_finitenonzero]])),
                    np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)


@pytest.mark.parametrize(
    ("ufunc",),
    [
        # Complete list obtained from from https://docs.scipy.org/doc/numpy/reference/ufuncs.html on Sep 30, 2019
        (np.add,),
        (np.subtract,),
        (np.multiply,),
        (np.divide,),
        (np.true_divide,),
        (np.floor_divide,),
        (np.negative,),
        (np.positive,),
        (np.power,),
        (np.absolute,),
        (np.conj,),
        (np.conjugate,),
        (np.exp,),
        (np.log,),
        (np.sqrt,),
        (np.square,),
        (np.reciprocal,),
        (np.invert,),
        (np.less,),
        (np.less_equal,),
        (np.not_equal,),
        (np.equal,),
        (np.isfinite,),
        (np.isinf,),
        (np.isnan,),
        (np.copysign,),
        pytest.param(np.logaddexp, marks=pytest.mark.xfail),
        pytest.param(np.logaddexp2, marks=pytest.mark.xfail),
        pytest.param(np.remainder, marks=pytest.mark.xfail),
        pytest.param(np.mod, marks=pytest.mark.xfail),
        pytest.param(np.fmod, marks=pytest.mark.xfail),
        pytest.param(np.divmod, marks=pytest.mark.xfail),
        pytest.param(np.fabs, marks=pytest.mark.xfail),
        pytest.param(np.rint, marks=pytest.mark.xfail),
        pytest.param(np.sign, marks=pytest.mark.xfail),
        pytest.param(np.heaviside, marks=pytest.mark.xfail),
        pytest.param(np.exp2, marks=pytest.mark.xfail),
        pytest.param(np.log2, marks=pytest.mark.xfail),
        pytest.param(np.log10, marks=pytest.mark.xfail),
        pytest.param(np.expm1, marks=pytest.mark.xfail),
        pytest.param(np.log1p, marks=pytest.mark.xfail),
        pytest.param(np.cbrt, marks=pytest.mark.xfail),
        pytest.param(np.gcd, marks=pytest.mark.xfail),
        pytest.param(np.lcm, marks=pytest.mark.xfail),
        pytest.param(np.sin, marks=pytest.mark.xfail),
        pytest.param(np.cos, marks=pytest.mark.xfail),
        pytest.param(np.tan, marks=pytest.mark.xfail),
        pytest.param(np.arcsin, marks=pytest.mark.xfail),
        pytest.param(np.arccos, marks=pytest.mark.xfail),
        pytest.param(np.arctan, marks=pytest.mark.xfail),
        pytest.param(np.arctan2, marks=pytest.mark.xfail),
        pytest.param(np.hypot, marks=pytest.mark.xfail),
        pytest.param(np.sinh, marks=pytest.mark.xfail),
        pytest.param(np.cosh, marks=pytest.mark.xfail),
        pytest.param(np.tanh, marks=pytest.mark.xfail),
        pytest.param(np.arcsinh, marks=pytest.mark.xfail),
        pytest.param(np.arccosh, marks=pytest.mark.xfail),
        pytest.param(np.arctanh, marks=pytest.mark.xfail),
        pytest.param(np.deg2rad, marks=pytest.mark.xfail),
        pytest.param(np.rad2deg, marks=pytest.mark.xfail),
        pytest.param(np.bitwise_and, marks=pytest.mark.xfail),
        pytest.param(np.bitwise_or, marks=pytest.mark.xfail),
        pytest.param(np.bitwise_xor, marks=pytest.mark.xfail),
        pytest.param(np.left_shift, marks=pytest.mark.xfail),
        pytest.param(np.right_shift, marks=pytest.mark.xfail),
        pytest.param(np.greater, marks=pytest.mark.xfail),
        pytest.param(np.greater_equal, marks=pytest.mark.xfail),
        pytest.param(np.logical_and, marks=pytest.mark.xfail),
        pytest.param(np.logical_or, marks=pytest.mark.xfail),
        pytest.param(np.logical_xor, marks=pytest.mark.xfail),
        pytest.param(np.logical_not, marks=pytest.mark.xfail),
        pytest.param(np.maximum, marks=pytest.mark.xfail),
        pytest.param(np.minimum, marks=pytest.mark.xfail),
        pytest.param(np.fmax, marks=pytest.mark.xfail),
        pytest.param(np.fmin, marks=pytest.mark.xfail),
        pytest.param(np.isnat, marks=pytest.mark.xfail),
        pytest.param(np.fabs, marks=pytest.mark.xfail),
        pytest.param(np.signbit, marks=pytest.mark.xfail),
        pytest.param(np.nextafter, marks=pytest.mark.xfail),
        pytest.param(np.spacing, marks=pytest.mark.xfail),
        pytest.param(np.modf, marks=pytest.mark.xfail),
        pytest.param(np.ldexp, marks=pytest.mark.xfail),
        pytest.param(np.frexp, marks=pytest.mark.xfail),
        pytest.param(np.fmod, marks=pytest.mark.xfail),
        pytest.param(np.floor, marks=pytest.mark.xfail),
        pytest.param(np.ceil, marks=pytest.mark.xfail),
        pytest.param(np.trunc, marks=pytest.mark.xfail),
    ],
    ids=lambda uf:uf.__name__
)
def test_ufunc_existence(ufunc):
    qarray = Qs_array[Qs_finitenonzero]
    if ufunc.nin == 1:
        result = ufunc(qarray)
    elif ufunc.nin == 2:
        result = ufunc(qarray, qarray)


def test_numpy_array_conversion(Qs):
    "Check conversions between array as quaternions and array as floats"
    # First, just check 1-d array
    Q = Qs[Qs_nonnan][:12]  # Select first 3x4=12 non-nan elements in Qs
    assert Q.dtype == np.dtype(np.quaternion)
    q = quaternion.as_float_array(Q)  # View as array of floats
    assert q.dtype == np.dtype(np.float64)
    assert q.shape == (12, 4)  # This is the expected shape
    for j in range(12):
        for k in range(4):  # Check each component individually
            assert q[j][k] == Q[j].components[k]
    assert np.array_equal(quaternion.as_quat_array(q), Q)  # Check that we can go backwards
    # Next, see how that works if I flatten the q array
    q = q.flatten()
    assert q.dtype == np.dtype(np.float64)
    assert q.shape == (48,)
    for j in range(48):
        assert q[j] == Q[j // 4].components[j % 4]
    assert np.array_equal(quaternion.as_quat_array(q), Q)  # Check that we can go backwards
    # Now, reshape into 2-d array, and re-check
    P = Q.reshape(3, 4)  # Reshape into 3x4 array of quaternions
    p = quaternion.as_float_array(P)  # View as array of floats
    assert p.shape == (3, 4, 4)  # This is the expected shape
    for j in range(3):
        for k in range(4):
            for l in range(4):  # Check each component individually
                assert p[j][k][l] == Q[4 * j + k].components[l]
    assert np.array_equal(quaternion.as_quat_array(p), P)  # Check that we can go backwards
    # Check that we get an exception if the final dimension is not divisible by 4
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 1))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 2))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 3))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 5))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 5, 3, 2, 1))
    # Finally, check that it works on non-contiguous arrays, by adding random padding and then slicing
    q = quaternion.as_float_array(Q)
    q = np.concatenate((np.random.rand(q.shape[0], 3), q, np.random.rand(q.shape[0], 3)), axis=1)
    assert np.array_equal(quaternion.as_quat_array(q[:, 3:7]), Q)


def test_not_implemented():
    Q = quaternion.quaternion(1.1, 2.2, 3.3, 4.4)
    class B:
        pass
    b = B()
    with pytest.raises(TypeError):
        b * Q
    with pytest.raises(TypeError):
        Q * b
    class A:
        def __init__(self, data: float):
            self._data = data
        def __rmul__(self, other):
            return other * self._data
        def __mul__(self, other):
            return self._data * other
    a = A(1.2)
    assert a * Q == 1.2 * Q
    assert Q * a == Q * 1.2


@pytest.mark.skipif(not has_scipy, reason="Scipy is not installed")
def test_integrate_angular_velocity():
    import math
    import numpy as np
    import quaternion

    t0 = 0.0
    t2 = 10000.0
    Omega_orb = 2 * math.pi * 100 / t2
    Omega_prec = 2 * math.pi * 10 / t2
    alpha = 0.125 * math.pi
    alphadot = 2 * alpha / t2
    nu = 0.2 * alpha
    Omega_nu = Omega_prec
    R0 = np.exp(-1.1 * alpha * quaternion.x / 2)

    def R(t):
        return (R0
                * np.exp(Omega_prec * t * quaternion.z / 2) * np.exp((alpha + alphadot * t) * quaternion.x / 2)
                * np.exp(-Omega_prec * t * quaternion.z / 2)
                * np.exp(Omega_orb * t * quaternion.z / 2)
                * np.exp(nu * np.cos(Omega_nu * t) * quaternion.y / 2))

    def Rdot(t):
        R_dynamic = R0.inverse() * R(t)
        R_prec = np.exp(Omega_prec * t * quaternion.z / 2)
        R_nu = np.exp(nu * np.cos(Omega_nu * t) * quaternion.y / 2)
        return R0 * (0.5 * Omega_prec * quaternion.z * R_dynamic
                     + 0.5 * alphadot * R_prec * quaternion.x * R_prec.conj() * R_dynamic
                     + 0.5 * (Omega_orb - Omega_prec) * R_dynamic * R_nu.inverse() * quaternion.z * R_nu
                     + 0.5 * (-Omega_nu * nu * np.sin(Omega_nu * t)) * R_dynamic * quaternion.y)

    def Omega_tot(t):
        Rotor = R(t)
        RotorDot = Rdot(t)
        return (2 * RotorDot * Rotor.inverse()).vec

    # Test with exact Omega function
    t, R_approx = quaternion.integrate_angular_velocity(Omega_tot, 0.0, t2, R0=R(t0))
    R_exact = R(t)
    phi_Delta = np.array([quaternion.rotation_intrinsic_distance(e, a) for e, a in zip(R_exact, R_approx)])
    assert np.max(phi_Delta) < 1e-10, np.max(phi_Delta)

    # Test with exact Omega function taking two arguments
    t, R_approx = quaternion.integrate_angular_velocity(lambda t, R: Omega_tot(t), 0.0, t2, R0=R(t0))
    R_exact = R(t)
    phi_Delta = np.array([quaternion.rotation_intrinsic_distance(e, a) for e, a in zip(R_exact, R_approx)])
    assert np.max(phi_Delta) < 1e-10, np.max(phi_Delta)

    # Test with explicit values, given at the moments output above
    v = np.array([Omega_tot(ti) for ti in t])
    t, R_approx = quaternion.integrate_angular_velocity((t, v), 0.0, t2, R0=R(t0))
    R_exact = R(t)
    phi_Delta = np.array([quaternion.rotation_intrinsic_distance(e, a) for e, a in zip(R_exact, R_approx)])
    assert np.max(phi_Delta) < 1e-4, np.max(phi_Delta)


def test_mean_rotor_in_chordal_metric():
    # Test interpolation of some random constant quaternion
    q = quaternion.quaternion(*np.random.rand(4)).normalized()
    qs = np.array([q]*10)
    ts = np.linspace(0.1, 23.4, num=10)
    for length in range(1, 4):
        mean1 = quaternion.mean_rotor_in_chordal_metric(qs[:length])
        assert np.abs(q-mean1) < 1e-15, (q, mean1, length)
        with pytest.raises(ValueError):
            quaternion.mean_rotor_in_chordal_metric(qs[:length], ts[:length])
    for length in range(4, 11):
        mean1 = quaternion.mean_rotor_in_chordal_metric(qs[:length])
        assert np.abs(q-mean1) < 1e-15, (q, mean1, length)
        mean2 = quaternion.mean_rotor_in_chordal_metric(qs[:length], ts[:length])
        assert np.abs(q-mean2) < 1e-15, (q, mean2, length)


@pytest.mark.skipif(not has_scipy, reason="Scipy is not installed")
def test_optimal_alignment_in_Euclidean_metric():
    N = 10
    a⃗ = np.random.normal(size=(N, 3))
    R = quaternion.quaternion(*np.random.normal(size=(4))).normalized()

    # Test the exact result
    b⃗ = quaternion.rotate_vectors(R, a⃗)
    Rprm = quaternion.optimal_alignment_in_Euclidean_metric(a⃗, b⃗)
    assert quaternion.rotation_intrinsic_distance(R, np.conjugate(Rprm)) < 25*eps
    assert np.max(np.abs(a⃗ - quaternion.rotate_vectors(Rprm, b⃗))) < 40*eps

    # Uniform time steps
    t = np.linspace(-1.2, 3.4, num=N)
    Rprmprm = quaternion.optimal_alignment_in_Euclidean_metric(a⃗, b⃗, t)
    assert quaternion.rotation_intrinsic_distance(R, np.conjugate(Rprmprm)) < 25*eps
    assert np.max(np.abs(a⃗ - quaternion.rotate_vectors(Rprmprm, b⃗))) < 40*eps

    # Perturb b⃗ slightly
    δ = np.sqrt(eps)
    b⃗prmprmprm = [b⃗[i] + (2*(np.random.rand(3) - 0.5) * δ/np.sqrt(3)) for i in range(N)]
    Rprmprmprm = quaternion.optimal_alignment_in_Euclidean_metric(a⃗, b⃗prmprmprm)
    assert quaternion.rotation_intrinsic_distance(R, np.conjugate(Rprmprmprm)) < 25*δ
    assert np.max(np.abs(a⃗ - quaternion.rotate_vectors(Rprmprmprm, b⃗prmprmprm))) < 40*δ


def test_numpy_save_and_load():
    import tempfile
    a = quaternion.as_quat_array(np.random.rand(5,3,4))
    with tempfile.TemporaryFile() as temp:
        np.save(temp, a)
        temp.seek(0)  # Only needed here to simulate closing & reopening file, per np.save docs
        b = np.load(temp).view(dtype=np.quaternion)
    assert np.array_equal(a, b)


def test_pickle():
    import pickle
    a = quaternion.one
    assert pickle.loads(pickle.dumps(a)) == a


if __name__ == '__main__':
    print("The tests should be run automatically via pytest (`pip install pytest` and then just `pytest`)")




