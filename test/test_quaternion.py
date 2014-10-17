#!/usr/bin/env python

from __future__ import division
from numpy import *
import numpy as np
import quaternion
import warnings
import sys
from numpy.testing import assert_

def passer(b):
    pass
# Change this to strict_assert = assert_ to check for missing tests
strict_assert = passer

def test_quaternion_members():
    Q = quaternion.quaternion(1.1,2.2,3.3,4.4)
    assert_(Q.real==1.1)
    assert_(Q.w==1.1)
    assert_(Q.x==2.2)
    assert_(Q.y==3.3)
    assert_(Q.z==4.4)

def test_quaternion_methods():
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

    ## Unary bool returners
    # nonzero
    assert_(not q_0.nonzero()) # Do this one explicitly, to not use circular logic
    assert_(q_1.nonzero()) # Do this one explicitly, to not use circular logic
    for q in Qs_zero:
        assert_(not q.nonzero())
    for q in Qs_nonzero:
        assert_(q.nonzero())
    # isnan
    assert_(not q_0.isnan()) # Do this one explicitly, to not use circular logic
    assert_(q_nan1.isnan()) # Do this one explicitly, to not use circular logic
    for q in Qs_nan:
        assert_(q.isnan())
    for q in Qs_nonnan:
        assert_(not q.isnan())
    # isinf
    assert_(not q_0.isinf()) # Do this one explicitly, to not use circular logic
    assert_(q_inf1.isinf()) # Do this one explicitly, to not use circular logic
    for q in Qs_inf:
        assert_(q.isinf())
    for q in Qs_noninf:
        assert_(not q.isinf())
    # isfinite
    assert_(not q_nan1.isfinite()) # Do this one explicitly, to not use circular logic
    assert_(not q_inf1.isfinite()) # Do this one explicitly, to not use circular logic
    assert_(not q_minf1.isfinite()) # Do this one explicitly, to not use circular logic
    assert_(q_0.isfinite()) # Do this one explicitly, to not use circular logic
    for q in Qs_nonfinite:
        assert_(not q.isfinite())
    for q in Qs_finite:
        assert_(q.isfinite())


    ## Binary bool returners
    # equal
    for q in Qs_nonnan:
        assert_(q==q) # self equality
        for p in Qs: # non-self inequality
            assert_((q is p) or (not (q==p)))
    for q in Qs:
        for p in Qs_nan:
            assert_(not q==p) # nan should never equal anything
    # not_equal
    for q in Qs_nonnan:
        assert_(not (q!=q)) # self non-not_equality
        for p in Qs_nonnan: # non-self not_equality
            assert_((q is p) or (q!=p))
    for q in Qs:
        for p in Qs_nan:
            assert_(q!=p) # nan should never equal anything
    # less, less_equal, greater, greater_equal
    for p in Qs:
        for q in Qs_nan:
            assert_(not p<q)
            assert_(not q<p)
            assert_(not p<=q)
            assert_(not q<=p)
            assert_(not p.greater(q))
            assert_(not q.greater(p))
            assert_(not p.greater_equal(q))
            assert_(not q.greater_equal(p))
    for p in Qs_nonnan:
        assert_((p<q_inf1) or (p is q_inf1))
        assert_((p<=q_inf1))
        assert_((q_minf1<p) or (p is q_minf1))
        assert_((q_minf1<=p))
        assert_((q_inf1.greater(p)) or (p is q_inf1))
        assert_((q_inf1.greater_equal(p)))
        assert_((p.greater(q_minf1)) or (p is q_minf1))
        assert_((p.greater_equal(q_minf1)))
    for p in [q_1, x, y, z, Q, Qbar]:
        assert_(q_0<p)
        assert_(q_0<=p)
        assert_(p.greater(q_0))
        assert_(p.greater_equal(q_0))
    for p in [Qneg]:
        assert_(p<q_0)
        assert_(p<=q_0)
        assert_(q_0.greater(p))
        assert_(q_0.greater_equal(p))
    for p in [x, y, z]:
        assert_(p<q_1)
        assert_(p<=q_1)
        assert_(q_1.greater(p))
        assert_(q_1.greater_equal(p))
    for p in [Qlog, Qexp]:
        assert_(q_1<p)
        assert_(q_1<=p)
        assert_(p.greater(q_1))
        assert_(p.greater_equal(q_1))

    ## Unary float returners
    # absolute
    for q in Qs_nan:
        assert_(np.isnan(q.abs()))
    for q in Qs_inf:
        assert_(np.isinf(q.abs()))
    for q,a in [(q_0,0.0), (q_1, 1.0), (x,1.0), (y,1.0), (z,1.0),
                (Q,   np.sqrt(Q.w**2+Q.x**2+Q.y**2+Q.z**2)),
                (Qbar,np.sqrt(Q.w**2+Q.x**2+Q.y**2+Q.z**2))]:
        assert_(q.absolute() == q.abs())
        assert_(q.abs() == a)


    ## Unary quaternion returners
    # negative
    assert_(-Q == Qneg)
    for q in Qs_finite:
        assert_(-q==-1.0*q)
    for q in Qs_nonnan:
        assert_(-(-q)==q)
    # conjugate
    assert_(Q.conjugate() == Qbar)
    for q in Qs_nonnan:
        assert_(q.conjugate() == q.conj())
        assert_(q.conjugate().conjugate() == q)
    # log, exp
    qlogexp_precision = 4.e-15
    assert_((Q.log()-Qlog).abs() < qlogexp_precision)
    assert_((Q.exp()-Qexp).abs() < qlogexp_precision)
    assert_((Q.log().exp()-Q).abs() < qlogexp_precision)
    assert_((Q.exp().log()-Q).abs() > qlogexp_precision) # Note order of operations!
    strict_assert(False) # logs of basis vectors
    strict_assert(False) # logs of interesting scalars * basis vectors


    ## Quaternion-quaternion binary quaternion returners
    # add
    for q in Qs_nonnan:
        for p in Qs_nonnan:
            assert_(q+p==quaternion.quaternion(q.w+p.w,q.x+p.x,q.y+p.y,q.z+p.z)
                    or (q is q_inf1 and p is q_minf1)
                    or (p is q_inf1 and q is q_minf1))
    strict_assert(False) # Check nans and (q_inf1+q_minf1) and (q_minf1+q_inf1)
    # subtract
    for q in Qs_finite:
        for p in Qs_finite:
            assert_(q-p==quaternion.quaternion(q.w-p.w,q.x-p.x,q.y-p.y,q.z-p.z))
    strict_assert(False) # Check non-finite
    # copysign
    strict_assert(False)

    ## Quaternion-quaternion or quaternion-scalar binary quaternion returners
    # multiply
    for q in Qs_finite: # General quaternion mult. would use inf*0.0
        assert_(q*q_1==q)
    for q in Qs_finite: # General quaternion mult. would use inf*0.0
        assert_(q*1.0==q)
    for s in [-2.3,-1.2,-1.0,1.0,1.2,2.3]:
        for q in Qs_finite:
            assert_(q*s==quaternion.quaternion(s*q.w,s*q.x,s*q.y,s*q.z))
            assert_(s*q==q*s)
    for q in Qs_finite:
        assert_(0.0*q==q_0)
        assert_(0.0*q==q*0.0)
    strict_assert(False) # Right-multiplication by non-unit scalar
    strict_assert(False) # Left-multiplication by scalar
    strict_assert(False) # Each of the 16 basic products
    # divide
    for q in Qs_finitenonzero:
        assert_( ((q/q)-q_1).abs()<np.finfo(float).eps )
    for q in Qs_nonnan:
        assert_( q/1.0==q )
    strict_assert(False) # Division by non-unit scalar
    strict_assert(False) # Each of the 16 basic products
    # power
    qpower_precision = 4.e-14
    for q in Qs:
        if(q.isfinite() and q.nonzero()):
            assert_( (q**1.0-q).abs()<qpower_precision )
            assert_( (q**2.0-q*q).abs()<qpower_precision )
    qinverse_precision = 5.e-16
    for q in Qs:
        if(q.isfinite() and q.nonzero()):
            assert_( ((q**-1.0)*q - q_1).abs()<qinverse_precision )
    for q in Qs:
        if(q.isfinite() and q.nonzero()):
            assert_( ((q**q_1)-q).abs()<qpower_precision )
    strict_assert(False)


def test_numpy_array_conversion():
    pass

if __name__=='__main__':
    test_quaternion_members()
    test_quaternion_methods()
    test_numpy_array_conversion()
