#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import os
import operator

import numpy as np
import quaternion
from numpy import *
import pytest


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


q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp, = range(len(Qs()))
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


def test_quaternion_members():
    Q = quaternion.quaternion(1.1, 2.2, 3.3, 4.4)
    assert Q.real == 1.1
    assert Q.w == 1.1
    assert Q.x == 2.2
    assert Q.y == 3.3
    assert Q.z == 4.4


def test_constants():
    assert quaternion.one == np.quaternion(1.0, 0.0, 0.0, 0.0)
    assert quaternion.x == np.quaternion(0.0, 1.0, 0.0, 0.0)
    assert quaternion.y == np.quaternion(0.0, 0.0, 1.0, 0.0)
    assert quaternion.z == np.quaternion(0.0, 0.0, 0.0, 1.0)


def test_as_float_quat(Qs):
    qs = Qs[Qs_nonnan]
    for quats in [qs, np.vstack((qs,)*3), np.vstack((qs,)*(3*5)).reshape((3, 5)+qs.shape),
                  np.vstack((qs,)*(3*5*6)).reshape((3, 5, 6)+qs.shape)]:
        floats = quaternion.as_float_array(quats)
        assert floats.shape == quats.shape+(4,)
        assert allclose(quaternion.as_quat_array(floats), quats)


def test_as_rotation_matrix(Rs):
    def quat_mat(quat):
        return np.array([(quat * v * quat.inverse()).vec for v in [quaternion.x, quaternion.y, quaternion.z]]).T

    def quat_mat_vec(quats):
        mat_vec = np.array([quaternion.as_float_array(quats * v * np.invert(quats))[..., 1:]
                            for v in [quaternion.x, quaternion.y, quaternion.z]])
        return np.transpose(mat_vec, tuple(range(mat_vec.ndim))[1:-1]+(-1, 0))

    with pytest.raises(ZeroDivisionError):
        quaternion.as_rotation_matrix(quaternion.zero)

    for R in Rs:
        # Test correctly normalized rotors:
        assert allclose(quat_mat(R), quaternion.as_rotation_matrix(R), atol=2*eps)
        # Test incorrectly normalized rotors:
        assert allclose(quat_mat(R), quaternion.as_rotation_matrix(1.1*R), atol=2*eps)

    Rs0 = Rs.copy()
    Rs0[Rs.shape[0]//2] = quaternion.zero
    with pytest.raises(ZeroDivisionError):
        quaternion.as_rotation_matrix(Rs0)

    # Test correctly normalized rotors:
    assert allclose(quat_mat_vec(Rs), quaternion.as_rotation_matrix(Rs), atol=2*eps)
    # Test incorrectly normalized rotors:
    assert allclose(quat_mat_vec(Rs), quaternion.as_rotation_matrix(1.1*Rs), atol=2*eps)


def test_from_rotation_matrix(Rs):
    rot_mat_eps = 10*eps
    try:
        from scipy import linalg
    except ImportError:
        rot_mat_eps = 400*eps

    for i, R1 in enumerate(Rs):
        R2 = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(R1))
        d = quaternion.rotation_intrinsic_distance(R1, R2)
        assert d < rot_mat_eps, (i, R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

    Rs2 = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(Rs))
    for R1, R2 in zip(Rs, Rs2):
        d = quaternion.rotation_intrinsic_distance(R1, R2)
        assert d < rot_mat_eps, (R1, R2, d)  # Can't use allclose here; we don't care about rotor sign


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
        assert abs((np.quaternion(0, 0, 0, varphi / 2.).exp() * np.quaternion(0, 0, vartheta / 2., 0).exp())
                   - quaternion.from_spherical_coords(vartheta, varphi)) < 1.e-15


def test_as_spherical_coords(Rs):
    np.random.seed(1843)
    # First test on rotors that are precisely spherical-coordinate rotors
    random_angles = [[np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        vartheta2, varphi2 = quaternion.as_spherical_coords(quaternion.from_spherical_coords(vartheta, varphi))
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
        assert q.abs() == a


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
        assert q.norm() == a


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


def test_quaternion_sqrt(Qs):
    sqrt_precision = 4.e-15
    for q in Qs[Qs_finitenonzero]:
        assert ( (q.sqrt()) * (q.sqrt()) - q ).abs() < sqrt_precision


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
                assert allclose((s*t)**q, (s**q)*(t**q), rtol=2*qpower_precision)
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
    # set components/vec from np.array, list, tuple
    for q in Qs[Qs_nonnan]:
        for seq_type in [np.array, list, tuple]:
            p = np.quaternion(*q.components)
            r = np.quaternion(*q.components)
            p.components = seq_type((-5.5, 6.6, -7.7, 8.8))
            r.vec = seq_type((6.6, -7.7, 8.8))
            assert np.array_equal(p.components, np.array([-5.5, 6.6, -7.7, 8.8]))
            assert np.array_equal(r.components, np.array([q.w, 6.6, -7.7, 8.8]))
    # TypeError when setting components with the wrong type or size of thing
    for q in Qs:
        for seq_type in [np.array, list, tuple]:
            p = np.quaternion(*q.components)
            r = np.quaternion(*q.components)
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


@pytest.mark.skipif(os.environ.get('FAST'), reason="Takes ~10 seconds")
def test_metrics(Rs):
    metric_precision = 4.e-15
    # Check non-negativity
    for R1 in Rs:
        for R2 in Rs:
            assert quaternion.rotor_chordal_distance(R1, R2) >= 0.
            assert quaternion.rotor_intrinsic_distance(R1, R2) >= 0.
            assert quaternion.rotation_chordal_distance(R1, R2) >= 0.
            assert quaternion.rotation_intrinsic_distance(R1, R2) >= 0.
    # Check discernibility
    for R1 in Rs:
        for R2 in Rs:
            assert bool(quaternion.rotor_chordal_distance(R1, R2) > 0.) != bool(R1 == R2)
            assert bool(quaternion.rotor_intrinsic_distance(R1, R2) > 5.e-16) != bool(R1 == R2)
            assert bool(quaternion.rotation_chordal_distance(R1, R2) > 0.) != bool(R1 == R2 or R1 == -R2)
            assert bool(quaternion.rotation_intrinsic_distance(R1, R2) > 5.e-16) != bool(R1 == R2 or R1 == -R2)
    # Check symmetry
    for R1 in Rs:
        for R2 in Rs:
            assert abs(quaternion.rotor_chordal_distance(R1, R2)
                       - quaternion.rotor_chordal_distance(R2, R1)) < metric_precision
            assert abs(quaternion.rotor_intrinsic_distance(R1, R2)
                       - quaternion.rotor_intrinsic_distance(R2, R1)) < metric_precision
            assert abs(quaternion.rotation_chordal_distance(R1, R2)
                       - quaternion.rotation_chordal_distance(R2, R1)) < metric_precision
            assert abs(quaternion.rotation_intrinsic_distance(R1, R2)
                       - quaternion.rotation_intrinsic_distance(R2, R1)) < metric_precision
    # Check triangle inequality
    for R1 in Rs:
        for R2 in Rs:
            for R3 in Rs:
                assert quaternion.rotor_chordal_distance(R1, R3) - metric_precision \
                       <= (quaternion.rotor_chordal_distance(R1, R2) + quaternion.rotor_chordal_distance(R2, R3))
                assert quaternion.rotor_chordal_distance(R1, R3) - metric_precision \
                       <= (quaternion.rotor_chordal_distance(R1, R2) + quaternion.rotor_chordal_distance(R2, R3))
                assert quaternion.rotation_intrinsic_distance(R1, R3) - metric_precision \
                       <= (
                    quaternion.rotation_intrinsic_distance(R1, R2) + quaternion.rotation_intrinsic_distance(R2, R3))
                assert quaternion.rotation_intrinsic_distance(R1, R3) - metric_precision \
                       <= (
                    quaternion.rotation_intrinsic_distance(R1, R2) + quaternion.rotation_intrinsic_distance(R2, R3))
    # Check distances from self or -self
    for R in Rs:
        # All distances from self should be 0.0
        assert quaternion.rotor_chordal_distance(R, R) == 0.0
        assert quaternion.rotor_intrinsic_distance(R, R) < 5.e-16
        assert quaternion.rotation_chordal_distance(R, R) == 0.0
        assert quaternion.rotation_intrinsic_distance(R, R) < 5.e-16
        # Chordal rotor distance from -self should be 2
        assert abs(quaternion.rotor_chordal_distance(R, -R) - 2.0) < metric_precision
        # Intrinsic rotor distance from -self should be 2pi
        assert abs(quaternion.rotor_intrinsic_distance(R, -R) - 2.0 * np.pi) < metric_precision
        # Rotation distances from -self should be 0
        assert quaternion.rotation_chordal_distance(R, -R) == 0.0
        assert quaternion.rotation_intrinsic_distance(R, -R) < 5.e-16
    # We expect the chordal distance to be smaller than the intrinsic distance (or equal, if the distance is zero)
    for R in Rs:
        assert (
            quaternion.rotor_chordal_distance(quaternion.one, R) < quaternion.rotor_intrinsic_distance(quaternion.one,
                                                                                                       R)
            or R == quaternion.one)
    # Check invariance under overall rotations: d(R1, R2) = d(R3*R1, R3*R2) = d(R1*R3, R2*R3)
    for R1 in Rs:
        for R2 in Rs:
            for R3 in Rs:
                assert abs(quaternion.rotor_chordal_distance(R1, R2)
                           - quaternion.rotor_chordal_distance(R3 * R1, R3 * R2)) < metric_precision
                assert abs(quaternion.rotor_chordal_distance(R1, R2)
                           - quaternion.rotor_chordal_distance(R1 * R3, R2 * R3)) < metric_precision
                assert abs(quaternion.rotation_intrinsic_distance(R1, R2)
                           - quaternion.rotation_intrinsic_distance(R3 * R1, R3 * R2)) < metric_precision
                assert abs(quaternion.rotation_intrinsic_distance(R1, R2)
                           - quaternion.rotation_intrinsic_distance(R1 * R3, R2 * R3)) < metric_precision


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
            assert quaternion.rotation_chordal_distance(slerp_evaluate(quaternion.one, Q2, t),
                                                        (np.cos(np.pi * t / 2) * quaternion.one + np.sin(
                                                            np.pi * t / 2) * Q2)) < slerp_precision
        t = np.linspace(0.0, 1.0, num=100, endpoint=True)
        assert allclose(slerp(quaternion.one, Q2, 0.0, 1.0, t),
                        np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2, verbose=True)
        assert allclose(slerp(quaternion.one, Q2, -10.0, 20.0, 30 * t - 10.0),
                        np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2, verbose=True)
        t = 1.5 * t - 0.125
        assert allclose(slerp(quaternion.one, Q2, 0.0, 1.0, t),
                        np.cos(np.pi * t / 2) * quaternion.one + np.sin(np.pi * t / 2) * Q2, verbose=True)
    # Test that (slerp of rotated rotors) is (rotated slerp of rotors)
    for R in Rs:
        for Q2 in ones[1:]:
            for t in np.linspace(0.0, 1.0, num=100, endpoint=True):
                assert quaternion.rotation_chordal_distance(R * slerp_evaluate(quaternion.one, Q2, t),
                                                            slerp_evaluate(R * quaternion.one, R * Q2,
                                                                             t)) < slerp_precision
            t = np.linspace(0.0, 1.0, num=100, endpoint=True)
            assert allclose(R * slerp(quaternion.one, Q2, 0.0, 1.0, t),
                            slerp(R * quaternion.one, R * Q2, 0.0, 1.0, t),
                            verbose=True)


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


@pytest.mark.xfail
def test_integration_of_angular_velocity():
    assert False


def test_setitem_quat(Qs):
    Ps = Qs[:]
    # setitem from quaternion
    for j in range(len(Ps)):
        Ps[j] = np.quaternion(1.3, 2.4, 3.5, 4.7)
        for k in range(j):
            assert Ps[k] == np.quaternion(1.3, 2.4, 3.5, 4.7)
        for k in range(j + 1, len(Ps)):
            assert Ps[k] == Qs[k]
    # setitem from np.array, list, or tuple
    for seq_type in [np.array, list, tuple]:
        Ps = Qs[:]
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
            for k in range(j):
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
    assert Q.dtype == np.dtype(np.quaternion)
    q = quaternion.as_float_array(Q)  # View as array of floats
    assert q.dtype == np.dtype(np.float)
    assert q.shape == (12, 4)  # This is the expected shape
    for j in range(12):
        for k in range(4):  # Check each component individually
            assert q[j][k] == Q[j].components[k]
    assert np.array_equal(quaternion.as_quat_array(q), Q)  # Check that we can go backwards
    # Next, see how that works if I flatten the q array
    q = q.flatten()
    assert q.dtype == np.dtype(np.float)
    assert q.shape == (48,)
    for j in range(48):
        assert q[j] == Q[j // 4].components[j % 4]
    assert np.array_equal(quaternion.as_quat_array(q), Q)  # Check that we can go backwards
    # Finally, reshape into 2-d array, and re-check
    P = Q.reshape(3, 4)  # Reshape into 3x4 array of quaternions
    p = quaternion.as_float_array(P)  # View as array of floats
    assert p.shape == (3, 4, 4)  # This is the expected shape
    for j in range(3):
        for k in range(4):
            for l in range(4):  # Check each component individually
                assert p[j][k][l] == Q[4 * j + k].components[l]
    assert np.array_equal(quaternion.as_quat_array(p), P)  # Check that we can go backwards


if __name__ == '__main__':
    print("The tests should be run automatically via py.test (pip install pytest)")
