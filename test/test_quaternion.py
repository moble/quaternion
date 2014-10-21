#!/usr/bin/env python

from __future__ import division
from numpy import *
import numpy as np
import quaternion
import warnings
import sys
import pytest

def passer(b):
    pass
# Change this to strict_assert = assert_ to check for missing tests
strict_assert = passer



# Create a nice variety of quaternion objects.  N.B.: Do NOT make
# two that are identical, or some tests (especially equality) will fail
q_nan1  = quaternion.quaternion(np.nan,0.,0.,0.)
q_inf1  = quaternion.quaternion(np.inf,0.,0.,0.)
q_minf1 = quaternion.quaternion(-np.inf,0.,0.,0.)
q_0     = quaternion.quaternion(0.,0.,0.,0.)
q_1     = quaternion.quaternion(1.,0.,0.,0.)
x       = quaternion.quaternion(0.,1.,0.,0.)
y       = quaternion.quaternion(0.,0.,1.,0.)
z       = quaternion.quaternion(0.,0.,0.,1.)
Q       = quaternion.quaternion(1.1,2.2,3.3,4.4)
Qneg    = quaternion.quaternion(-1.1,-2.2,-3.3,-4.4)
Qbar    = quaternion.quaternion(1.1,-2.2,-3.3,-4.4)
Qlog    = quaternion.quaternion(1.7959088706354, 0.515190292664085,
                                0.772785438996128, 1.03038058532817)
Qexp    = quaternion.quaternion(2.81211398529184, -0.392521193481878,
                                -0.588781790222817, -0.785042386963756)
Qs = [q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, Q, Qneg, Qbar, Qlog, Qexp,]
Qs_zero = [q for q in Qs if not q.nonzero()]
Qs_nonzero = [q for q in Qs if q.nonzero()]
Qs_nan = [q for q in Qs if q.isnan()]
Qs_nonnan = [q for q in Qs if not q.isnan()]
Qs_nonnannonzero = [q for q in Qs if not q.isnan() and q.nonzero()]
Qs_inf = [q for q in Qs if q.isinf()]
Qs_noninf = [q for q in Qs if not q.isinf()]
Qs_noninfnonzero = [q for q in Qs if not q.isinf() and q.nonzero()]
Qs_finite = [q for q in Qs if q.isfinite()]
Qs_nonfinite = [q for q in Qs if not q.isfinite()]
Qs_finitenonzero = [q for q in Qs if q.isfinite() and q.nonzero()]

def test_quaternion_members():
    Q = quaternion.quaternion(1.1,2.2,3.3,4.4)
    assert Q.real==1.1
    assert Q.w==1.1
    assert Q.x==2.2
    assert Q.y==3.3
    assert Q.z==4.4

def test_quaternion_methods():
    ## Unary bool returners
    # nonzero
    assert not q_0.nonzero() # Do this one explicitly, to not use circular logic
    assert q_1.nonzero() # Do this one explicitly, to not use circular logic
    for q in Qs_zero:
        assert not q.nonzero()
    for q in Qs_nonzero:
        assert q.nonzero()
    # isnan
    assert not q_0.isnan() # Do this one explicitly, to not use circular logic
    assert q_nan1.isnan() # Do this one explicitly, to not use circular logic
    for q in Qs_nan:
        assert q.isnan()
    for q in Qs_nonnan:
        assert not q.isnan()
    # isinf
    assert not q_0.isinf() # Do this one explicitly, to not use circular logic
    assert q_inf1.isinf() # Do this one explicitly, to not use circular logic
    for q in Qs_inf:
        assert q.isinf()
    for q in Qs_noninf:
        assert not q.isinf()
    # isfinite
    assert not q_nan1.isfinite() # Do this one explicitly, to not use circular logic
    assert not q_inf1.isfinite() # Do this one explicitly, to not use circular logic
    assert not q_minf1.isfinite() # Do this one explicitly, to not use circular logic
    assert q_0.isfinite() # Do this one explicitly, to not use circular logic
    for q in Qs_nonfinite:
        assert not q.isfinite()
    for q in Qs_finite:
        assert q.isfinite()


    ## Binary bool returners
    # equal
    for q in Qs_nonnan:
        assert q==q # self equality
        for p in Qs: # non-self inequality
            assert (q is p) or (not (q==p))
    for q in Qs:
        for p in Qs_nan:
            assert not q==p # nan should never equal anything
    # not_equal
    for q in Qs_nonnan:
        assert not (q!=q) # self non-not_equality
        for p in Qs_nonnan: # non-self not_equality
            assert (q is p) or (q!=p)
    for q in Qs:
        for p in Qs_nan:
            assert q!=p # nan should never equal anything
    # less, less_equal, greater, greater_equal
    for p in Qs:
        for q in Qs_nan:
            assert not p<q
            assert not q<p
            assert not p<=q
            assert not q<=p
            assert not p.greater(q)
            assert not q.greater(p)
            assert not p.greater_equal(q)
            assert not q.greater_equal(p)
    for p in Qs_nonnan:
        assert (p<q_inf1) or (p is q_inf1)
        assert (p<=q_inf1)
        assert (q_minf1<p) or (p is q_minf1)
        assert (q_minf1<=p)
        assert (q_inf1.greater(p)) or (p is q_inf1)
        assert (q_inf1.greater_equal(p))
        assert (p.greater(q_minf1)) or (p is q_minf1)
        assert (p.greater_equal(q_minf1))
    for p in [q_1, x, y, z, Q, Qbar]:
        assert q_0<p
        assert q_0<=p
        assert p.greater(q_0)
        assert p.greater_equal(q_0)
    for p in [Qneg]:
        assert p<q_0
        assert p<=q_0
        assert q_0.greater(p)
        assert q_0.greater_equal(p)
    for p in [x, y, z]:
        assert p<q_1
        assert p<=q_1
        assert q_1.greater(p)
        assert q_1.greater_equal(p)
    for p in [Qlog, Qexp]:
        assert q_1<p
        assert q_1<=p
        assert p.greater(q_1)
        assert p.greater_equal(q_1)

    ## Unary float returners
    # absolute
    for q in Qs_nan:
        assert np.isnan(q.abs())
    for q in Qs_inf:
        assert np.isinf(q.abs())
    for q,a in [(q_0,0.0), (q_1, 1.0), (x,1.0), (y,1.0), (z,1.0),
                (Q,   np.sqrt(Q.w**2+Q.x**2+Q.y**2+Q.z**2)),
                (Qbar,np.sqrt(Q.w**2+Q.x**2+Q.y**2+Q.z**2))]:
        assert q.abs() == a
    # norm
    for q in Qs_nan:
        assert np.isnan(q.norm())
    for q in Qs_inf:
        assert np.isinf(q.norm())
    for q,a in [(q_0,0.0), (q_1, 1.0), (x,1.0), (y,1.0), (z,1.0),
                (Q,   Q.w**2+Q.x**2+Q.y**2+Q.z**2),
                (Qbar,Q.w**2+Q.x**2+Q.y**2+Q.z**2)]:
        assert q.norm() == a


    ## Unary quaternion returners
    # negative
    assert -Q == Qneg
    for q in Qs_finite:
        assert -q==-1.0*q
    for q in Qs_nonnan:
        assert -(-q)==q
    # conjugate
    assert Q.conjugate() == Qbar
    for q in Qs_nonnan:
        assert q.conjugate() == q.conj()
        assert q.conjugate().conjugate() == q
    # log, exp
    qlogexp_precision = 4.e-15
    assert (Q.log()-Qlog).abs() < qlogexp_precision
    assert (Q.exp()-Qexp).abs() < qlogexp_precision
    assert (Q.log().exp()-Q).abs() < qlogexp_precision
    assert (Q.exp().log()-Q).abs() > qlogexp_precision # Note order of operations!
    strict_assert(False) # logs of basis vectors
    strict_assert(False) # logs of interesting scalars * basis vectors


    ## Quaternion-quaternion binary quaternion returners
    # add
    for q in Qs_nonnan:
        for p in Qs_nonnan:
            assert (q+p==quaternion.quaternion(q.w+p.w,q.x+p.x,q.y+p.y,q.z+p.z)
                    or (q is q_inf1 and p is q_minf1)
                    or (p is q_inf1 and q is q_minf1))
    strict_assert(False) # Check nans and (q_inf1+q_minf1) and (q_minf1+q_inf1)
    # subtract
    for q in Qs_finite:
        for p in Qs_finite:
            assert q-p==quaternion.quaternion(q.w-p.w,q.x-p.x,q.y-p.y,q.z-p.z)
    strict_assert(False) # Check non-finite
    # copysign
    strict_assert(False)

    ## Quaternion-quaternion or quaternion-scalar binary quaternion returners
    # multiply
    for q in Qs_finite: # General quaternion mult. would use inf*0.0
        assert q*q_1==q
    for q in Qs_finite: # General quaternion mult. would use inf*0.0
        assert q*1.0==q
    for s in [-2.3,-1.2,-1.0,1.0,1.2,2.3]:
        for q in Qs_finite:
            assert q*s==quaternion.quaternion(s*q.w,s*q.x,s*q.y,s*q.z)
            assert s*q==q*s
    for q in Qs_finite:
        assert 0.0*q==q_0
        assert 0.0*q==q*0.0
    for q in [q_1, x, y, z]:
        assert q_1*q==q
        assert q*q_1==q
    assert x*x==-q_1
    assert x*y==z
    assert x*z==-y
    assert y*x==-z
    assert y*y==-q_1
    assert y*z==x
    assert z*x==y
    assert z*y==-x
    assert z*z==-q_1
    # divide
    for q in Qs_finitenonzero:
        assert  ((q/q)-q_1).abs()<np.finfo(float).eps
    for q in Qs_nonnan:
        assert  q/1.0==q
    strict_assert(False) # Division by non-unit scalar
    strict_assert(False) # Each of the 16 basic products
    for q in [q_1, x, y, z]:
        assert q_1/q==q.conj()
        assert q/q_1==q
    assert x/x==q_1
    assert x/y==-z
    assert x/z==y
    assert y/x==z
    assert y/y==q_1
    assert y/z==-x
    assert z/x==-y
    assert z/y==x
    assert z/z==q_1
    # power
    qpower_precision = 4.e-13
    for q in Qs:
        if(q.isfinite() and q.nonzero()):
            assert  (((q**0.5)*(q**0.5))-q).abs()<qpower_precision
            assert  (q**1.0-q).abs()<qpower_precision
            assert  (q**2.0-q*q).abs()<qpower_precision
            assert  (q**2-q*q).abs()<qpower_precision
            assert  (q**3-q*q*q).abs()<qpower_precision
    qinverse_precision = 5.e-16
    for q in Qs:
        if(q.isfinite() and q.nonzero()):
            assert  ((q**-1.0)*q - q_1).abs()<qinverse_precision
    for q in Qs:
        if(q.isfinite() and q.nonzero()):
            assert  ((q**q_1)-q).abs()<qpower_precision
    strict_assert(False)


def test_getset():
    # get components/vec
    for q in Qs_nonnan:
        assert np.array_equal(q.components, np.array([q.w,q.x,q.y,q.z]))
        assert np.array_equal(q.vec, np.array([q.x,q.y,q.z]))
    # set components/vec from np.array, list, tuple
    for q in Qs_nonnan:
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
    # getitem
    # setitem
    # copyswap
    # copyswapn
    # compare
    # argmax
    # nonzero
    # fillwithscalar
    pass

def test_arraydescr():
    # new
    # richcompare
    # hash
    # repr
    # str
    pass

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
    pass



def test_numpy_array_conversion():
    pass

if __name__=='__main__':
    test_quaternion_members()
    test_quaternion_methods()
    test_quaternion_getset()
    test_arrfuncs()
    test_arraydescr()
    test_casts()
    test_numpy_array_conversion()
