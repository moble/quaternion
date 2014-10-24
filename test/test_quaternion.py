#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import quaternion
from numpy import *
import warnings
import sys
import random
import pytest

def passer(b):
    pass
# Change this to strict_assert = assert_ to check for missing tests
strict_assert = passer


## The following fixtures are used to establish some re-usable data
## for the tests; they need to be re-constructed because some of the
## tests will change the values, but we want the values to be constant
## on every entry into a test.

@pytest.fixture
def Qs():
    q_nan1      = quaternion.quaternion(np.nan,0.,0.,0.)
    q_inf1      = quaternion.quaternion(np.inf,0.,0.,0.)
    q_minf1     = quaternion.quaternion(-np.inf,0.,0.,0.)
    q_0         = quaternion.quaternion(0.,0.,0.,0.)
    q_1         = quaternion.quaternion(1.,0.,0.,0.)
    x           = quaternion.quaternion(0.,1.,0.,0.)
    y           = quaternion.quaternion(0.,0.,1.,0.)
    z           = quaternion.quaternion(0.,0.,0.,1.)
    Q           = quaternion.quaternion(1.1,2.2,3.3,4.4)
    Qneg        = quaternion.quaternion(-1.1,-2.2,-3.3,-4.4)
    Qbar        = quaternion.quaternion(1.1,-2.2,-3.3,-4.4)
    Qnormalized = quaternion.quaternion(0.18257418583505539,0.36514837167011077,0.54772255750516607,0.73029674334022154)
    Qlog        = quaternion.quaternion(1.7959088706354, 0.515190292664085,
                                        0.772785438996128, 1.03038058532817)
    Qexp        = quaternion.quaternion(2.81211398529184, -0.392521193481878,
                                        -0.588781790222817, -0.785042386963756)
    return np.array([q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp,], dtype=np.quaternion)
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


def test_quaternion_members():
    Q = quaternion.quaternion(1.1,2.2,3.3,4.4)
    assert Q.real==1.1
    assert Q.w==1.1
    assert Q.x==2.2
    assert Q.y==3.3
    assert Q.z==4.4

def test_from_spherical_coords():
    random.seed(1843)
    random_angles = [[random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        assert abs((np.quaternion(0,0,0,varphi/2.).exp() * np.quaternion(0,0,vartheta/2.,0).exp())
                   - quaternion.from_spherical_coords(vartheta,varphi)) < 1.e-15

def test_from_euler_angles():
    random.seed(1843)
    random_angles = [[random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha,beta,gamma in random_angles:
        assert abs((np.quaternion(0,0,0,alpha/2.).exp() * np.quaternion(0,0,beta/2.,0).exp() * np.quaternion(0,0,0,gamma/2.).exp())
                   - quaternion.from_euler_angles(alpha, beta, gamma)) < 1.e-15


## Unary bool returners
def test_quaternion_nonzero(Qs):
    assert not Qs[q_0].nonzero() # Do this one explicitly, to not use circular logic
    assert Qs[q_1].nonzero() # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_zero]:
        assert not q.nonzero()
    for q in Qs[Qs_nonzero]:
        assert q.nonzero()
def test_quaternion_isnan(Qs):
    assert not Qs[q_0].isnan() # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isnan() # Do this one explicitly, to not use circular logic
    assert Qs[q_nan1].isnan() # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nan]:
        assert q.isnan()
    for q in Qs[Qs_nonnan]:
        assert not q.isnan()
def test_quaternion_isinf(Qs):
    assert not Qs[q_0].isinf() # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isinf() # Do this one explicitly, to not use circular logic
    assert Qs[q_inf1].isinf() # Do this one explicitly, to not use circular logic
    assert Qs[q_minf1].isinf() # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_inf]:
        assert q.isinf()
    for q in Qs[Qs_noninf]:
        assert not q.isinf()
def test_quaternion_isfinite(Qs):
    assert not Qs[q_nan1].isfinite() # Do this one explicitly, to not use circular logic
    assert not Qs[q_inf1].isfinite() # Do this one explicitly, to not use circular logic
    assert not Qs[q_minf1].isfinite() # Do this one explicitly, to not use circular logic
    assert Qs[q_0].isfinite() # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nonfinite]:
        assert not q.isfinite()
    for q in Qs[Qs_finite]:
        assert q.isfinite()


## Binary bool returners
def test_quaternion_equal(Qs):
    for j in Qs_nonnan:
        assert Qs[j]==Qs[j] # self equality
        for k in range(len(Qs)): # non-self inequality
            assert (j==k) or (not (Qs[j]==Qs[k]))
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert not q==p # nan should never equal anything
def test_quaternion_not_equal(Qs):
    for j in Qs_nonnan:
        assert not (Qs[j]!=Qs[j]) # self non-not_equality
        for k in Qs_nonnan: # non-self not_equality
            assert (j==k) or (Qs[j]!=Qs[k])
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert q!=p # nan should never equal anything
def test_quaternion_richcompare(Qs):
    for p in Qs:
        for q in Qs[Qs_nan]:
            assert not p<q
            assert not q<p
            assert not p<=q
            assert not q<=p
            assert not p.greater(q)
            assert not q.greater(p)
            assert not p.greater_equal(q)
            assert not q.greater_equal(p)
    for j in Qs_nonnan:
        p = Qs[j]
        assert (p<Qs[q_inf1]) or (j==q_inf1)
        assert (p<=Qs[q_inf1])
        assert (Qs[q_minf1]<p) or (j==q_minf1)
        assert (Qs[q_minf1]<=p)
        assert (Qs[q_inf1].greater(p)) or (j==q_inf1)
        assert (Qs[q_inf1].greater_equal(p))
        assert (p.greater(Qs[q_minf1])) or (j==q_minf1)
        assert (p.greater_equal(Qs[q_minf1]))
    for p in [Qs[q_1], Qs[x], Qs[y], Qs[z], Qs[Q], Qs[Qbar]]:
        assert Qs[q_0]<p
        assert Qs[q_0]<=p
        assert p.greater(Qs[q_0])
        assert p.greater_equal(Qs[q_0])
    for p in [Qs[Qneg]]:
        assert p<Qs[q_0]
        assert p<=Qs[q_0]
        assert Qs[q_0].greater(p)
        assert Qs[q_0].greater_equal(p)
    for p in [Qs[x], Qs[y], Qs[z]]:
        assert p<Qs[q_1]
        assert p<=Qs[q_1]
        assert Qs[q_1].greater(p)
        assert Qs[q_1].greater_equal(p)
    for p in [Qs[Qlog], Qs[Qexp]]:
        assert Qs[q_1]<p
        assert Qs[q_1]<=p
        assert p.greater(Qs[q_1])
        assert p.greater_equal(Qs[q_1])


## Unary float returners
def test_quaternion_absolute(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.abs())
    for q in Qs[Qs_inf]:
        assert np.isinf(q.abs())
    for q,a in [(Qs[q_0],0.0), (Qs[q_1], 1.0), (Qs[x],1.0), (Qs[y],1.0), (Qs[z],1.0),
                (Qs[Q],   np.sqrt(Qs[Q].w**2+Qs[Q].x**2+Qs[Q].y**2+Qs[Q].z**2)),
                (Qs[Qbar],np.sqrt(Qs[Q].w**2+Qs[Q].x**2+Qs[Q].y**2+Qs[Q].z**2))]:
        assert q.abs() == a
def test_quaternion_norm(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.norm())
    for q in Qs[Qs_inf]:
        assert np.isinf(q.norm())
    for q,a in [(Qs[q_0],0.0), (Qs[q_1], 1.0), (Qs[x],1.0), (Qs[y],1.0), (Qs[z],1.0),
                (Qs[Q],   Qs[Q].w**2+Qs[Q].x**2+Qs[Q].y**2+Qs[Q].z**2),
                (Qs[Qbar],Qs[Q].w**2+Qs[Q].x**2+Qs[Q].y**2+Qs[Q].z**2)]:
        assert q.norm() == a


## Unary quaternion returners
def test_quaternion_negative(Qs):
    assert -Qs[Q] == Qs[Qneg]
    for q in Qs[Qs_finite]:
        assert -q==-1.0*q
    for q in Qs[Qs_nonnan]:
        assert -(-q)==q
def test_quaternion_conjugate(Qs):
    assert Qs[Q].conjugate() == Qs[Qbar]
    for q in Qs[Qs_nonnan]:
        assert q.conjugate() == q.conj()
        assert q.conjugate().conjugate() == q
def test_quaternion_log_exp(Qs):
    qlogexp_precision = 4.e-15
    assert (Qs[Q].log()-Qs[Qlog]).abs() < qlogexp_precision
    assert (Qs[Q].exp()-Qs[Qexp]).abs() < qlogexp_precision
    assert (Qs[Q].log().exp()-Qs[Q]).abs() < qlogexp_precision
    assert (Qs[Q].exp().log()-Qs[Q]).abs() > qlogexp_precision # Note order of operations!
    strict_assert(False) # logs of basis vectors
    strict_assert(False) # logs of interesting scalars * basis vectors
def test_quaternion_normalized(Qs):
    assert Qs[Q].normalized() == Qs[Qnormalized]
    for q in Qs[Qs_finitenonzero]:
        assert abs(q.normalized().abs()-1.0) < 1.e-15

## Quaternion-quaternion binary quaternion returners
def test_quaternion_add(Qs):
    for j in Qs_nonnan:
        for k in Qs_nonnan:
            q = Qs[j]
            p = Qs[k]
            assert (q+p==quaternion.quaternion(q.w+p.w,q.x+p.x,q.y+p.y,q.z+p.z)
                    or (j==q_inf1 and k==q_minf1)
                    or (k==q_inf1 and j==q_minf1))
    strict_assert(False) # Check nans and (Qs[q_inf1]+Qs[q_minf1]) and (Qs[q_minf1]+Qs[q_inf1])
def test_quaternion_subtract(Qs):
    for q in Qs[Qs_finite]:
        for p in Qs[Qs_finite]:
            assert q-p==quaternion.quaternion(q.w-p.w,q.x-p.x,q.y-p.y,q.z-p.z)
    strict_assert(False) # Check non-finite
def test_quaternion_copysign(Qs):
    strict_assert(False)


## Quaternion-quaternion or quaternion-scalar binary quaternion returners
def test_quaternion_multiply(Qs):
    for q in Qs[Qs_finite]: # General quaternion mult. would use inf*0.0
        assert q*Qs[q_1]==q
    for q in Qs[Qs_finite]: # General quaternion mult. would use inf*0.0
        assert q*1.0==q
    for s in [-2.3,-1.2,-1.0,1.0,1.2,2.3]:
        for q in Qs[Qs_finite]:
            assert q*s==quaternion.quaternion(s*q.w,s*q.x,s*q.y,s*q.z)
            assert s*q==q*s
    for q in Qs[Qs_finite]:
        assert 0.0*q==Qs[q_0]
        assert 0.0*q==q*0.0
    for q in [Qs[q_1], Qs[x], Qs[y], Qs[z]]:
        assert Qs[q_1]*q==q
        assert q*Qs[q_1]==q
    assert Qs[x]*Qs[x]==-Qs[q_1]
    assert Qs[x]*Qs[y]==Qs[z]
    assert Qs[x]*Qs[z]==-Qs[y]
    assert Qs[y]*Qs[x]==-Qs[z]
    assert Qs[y]*Qs[y]==-Qs[q_1]
    assert Qs[y]*Qs[z]==Qs[x]
    assert Qs[z]*Qs[x]==Qs[y]
    assert Qs[z]*Qs[y]==-Qs[x]
    assert Qs[z]*Qs[z]==-Qs[q_1]
def test_quaternion_divide(Qs):
    for q in Qs[Qs_finitenonzero]:
        assert  ((q/q)-Qs[q_1]).abs()<np.finfo(float).eps
    for q in Qs[Qs_nonnan]:
        assert  q/1.0==q
    strict_assert(False) # Division by non-unit scalar
    strict_assert(False) # Each of the 16 basic products
    for q in [Qs[q_1], Qs[x], Qs[y], Qs[z]]:
        assert Qs[q_1]/q==q.conj()
        assert q/Qs[q_1]==q
    assert Qs[x]/Qs[x]==Qs[q_1]
    assert Qs[x]/Qs[y]==-Qs[z]
    assert Qs[x]/Qs[z]==Qs[y]
    assert Qs[y]/Qs[x]==Qs[z]
    assert Qs[y]/Qs[y]==Qs[q_1]
    assert Qs[y]/Qs[z]==-Qs[x]
    assert Qs[z]/Qs[x]==-Qs[y]
    assert Qs[z]/Qs[y]==Qs[x]
    assert Qs[z]/Qs[z]==Qs[q_1]
def test_quaternion_power(Qs):
    qpower_precision = 4.e-13
    for q in Qs[Qs_finitenonzero]:
        assert  (((q**0.5)*(q**0.5))-q).abs()<qpower_precision
        assert  (q**1.0-q).abs()<qpower_precision
        assert  (q**2.0-q*q).abs()<qpower_precision
        assert  (q**2-q*q).abs()<qpower_precision
        assert  (q**3-q*q*q).abs()<qpower_precision
    qinverse_precision = 5.e-16
    for q in Qs[Qs_finitenonzero]:
        assert  ((q**-1.0)*q - Qs[q_1]).abs()<qinverse_precision
    for q in Qs[Qs_finitenonzero]:
        assert  ((q**Qs[q_1])-q).abs()<qpower_precision
    strict_assert(False)


def test_quaternion_getset(Qs):
    # get parts a and b
    for q in Qs[Qs_nonnan]:
        assert q.a == q.w+1j*q.z
        assert q.b == q.y+1j*q.x
    # Check multiplication law for parts a and b
    part_mul_precision = 1.e-14
    for p in Qs[Qs_finite]:
        for q in Qs[Qs_finite]:
            assert abs((p*q).a - (p.a*q.a - p.b.conjugate()*q.b)) < part_mul_precision
            assert abs((p*q).b - (p.b*q.a + p.a.conjugate()*q.b)) < part_mul_precision
    # get components/vec
    for q in Qs[Qs_nonnan]:
        assert np.array_equal(q.components, np.array([q.w,q.x,q.y,q.z]))
        assert np.array_equal(q.vec, np.array([q.x,q.y,q.z]))
    # set components/vec from np.array, list, tuple
    for q in Qs[Qs_nonnan]:
        for seq_type in [np.array, list, tuple]:
            p = np.quaternion(*q.components)
            r = np.quaternion(*q.components)
            p.components = seq_type((-5.5, 6.6,-7.7,8.8))
            r.vec = seq_type((6.6,-7.7,8.8))
            assert np.array_equal(p.components, np.array([-5.5, 6.6,-7.7,8.8]))
            assert np.array_equal(r.components, np.array([q.w, 6.6,-7.7,8.8]))
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
                p.components = seq_type((-5.5, 6.6,-7.7,))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6,-7.7,8.8,-9.9))
            with pytest.raises(TypeError):
                r.vec = '2.2, 3.3, 4.4'
            with pytest.raises(TypeError):
                r.vec = seq_type([])
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5,))
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5, 6.6))
            with pytest.raises(TypeError):
                r.vec = seq_type((-5.5, 6.6,-7.7,8.8))


def test_arrfuncs():
    # nonzero
    # copyswap
    # copyswapn
    # getitem
    # setitem
    # compare
    # argmax
    # fillwithscalar
    pass

def test_setitem_quat(Qs):
    Ps = Qs[:]
    # setitem from quaternion
    for j in range(len(Ps)):
        Ps[j] = np.quaternion(1.3,2.4,3.5,4.7)
        for k in range(j):
            assert Ps[k] == np.quaternion(1.3,2.4,3.5,4.7)
        for k in range(j+1, len(Ps)):
            assert Ps[k] == Qs[k]
    # setitem from np.array, list, or tuple
    print("")
    for seq_type in [np.array, list, tuple]:
        Ps = Qs[:]
        with pytest.raises(TypeError):
            Ps[0] = seq_type(())
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,2.4,))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,2.4,3.5))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,2.4,3.5,4.7,5.9))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,2.4,3.5,4.7,5.9, np.nan))
        for j in range(len(Ps)):
            print("Trying to set from sequence")
            Ps[j] = seq_type((1.3,2.4,3.5,4.7))
            for k in range(j):
                assert Ps[k] == np.quaternion(1.3,2.4,3.5,4.7)
                for k in range(j+1, len(Ps)):
                    assert Ps[k] == Qs[k]
    with pytest.raises(TypeError):
        Ps[0] = 's'
    with pytest.raises(TypeError):
        Ps[0] = 's'

# def test_arraydescr():
#     # new
#     # richcompare
#     # hash
#     # repr
#     # str


# def test_casts():
#     # FLOAT, npy_float
#     # DOUBLE, npy_double
#     # LONGDOUBLE, npy_longdouble
#     # BOOL, npy_bool
#     # BYTE, npy_byte
#     # UBYTE, npy_ubyte
#     # SHORT, npy_short
#     # USHORT, npy_ushort
#     # INT, npy_int
#     # UINT, npy_uint
#     # LONG, npy_long
#     # ULONG, npy_ulong
#     # LONGLONG, npy_longlong
#     # ULONGLONG, npy_ulonglong
#     # CFLOAT, npy_float
#     # CDOUBLE, npy_double
#     # CLONGDOUBLE, npy_longdouble


def test_numpy_array_conversion(Qs):
    "Check conversions between array as quaternions and array as floats"
    # First, just check 1-d array
    Q = Qs[Qs_nonnan][:12] # Select first 3x4=12 non-nan elements in Qs
    assert Q.dtype == np.dtype(np.quaternion)
    q = quaternion.as_float_array(Q) # View as array of floats
    assert q.dtype == np.dtype(np.float)
    assert q.shape==(12,4) # This is the expected shape
    for j in range(12):
        for k in range(4): # Check each component individually
            assert q[j][k] == Q[j].components[k]
    assert np.array_equal( quaternion.as_quat_array(q), Q ) # Check that we can go backwards
    # Next, see how that works if I flatten the q array
    q = q.flatten()
    assert q.dtype == np.dtype(np.float)
    assert q.shape==(48,)
    for j in range(48):
        assert q[j] == Q[j//4].components[j%4]
    assert np.array_equal( quaternion.as_quat_array(q), Q ) # Check that we can go backwards
    # Finally, reshape into 2-d array, and re-check
    P = Q.reshape(3,4) # Reshape into 3x4 array of quaternions
    p = quaternion.as_float_array(P) # View as array of floats
    assert p.shape==(3,4,4) # This is the expected shape
    for j in range(3):
        for k in range(4):
            for l in range(4): # Check each component individually
                assert p[j][k][l] == Q[4*j+k].components[l]
    assert np.array_equal( quaternion.as_quat_array(p), P ) # Check that we can go backwards


if __name__=='__main__':
    print("quaternion_members")
    test_quaternion_members()
    print("quaternion_methods")
    test_quaternion_methods()
    print("qaternion_getset")
    test_quaternion_getset()
    print("arrfuncs")
    test_arrfuncs()
    print("arraydescr")
    test_arraydescr()
    print("casts")
    test_casts()
    print("numpy_array_conversion")
    test_numpy_array_conversion()
    print("Finished")
