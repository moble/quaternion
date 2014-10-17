#!/usr/bin/env python

from __future__ import division
from numpy import *
import numpy as np
import quaternion
from numpy.testing import assert_


def test_quaternion_methods():
    q_nan1 = quaternion.quaternion(np.nan,0.,0.,0.)
    q_inf1 = quaternion.quaternion(np.inf,0.,0.,0.)
    q_0    = quaternion.quaternion(0.,0.,0.,0.)
    q_1    = quaternion.quaternion(1.,0.,0.,0.)
    x      = quaternion.quaternion(0.,1.,0.,0.)
    y      = quaternion.quaternion(0.,0.,1.,0.)
    z      = quaternion.quaternion(0.,0.,0.,1.)
    Q      = quaternion.quaternion(1.1,2.2,3.3,4.4)
    Qbar   = quaternion.quaternion(1.1,-2.2,-3.3,-4.4)


    ## bool returners
    # nonzero
    for q in [q_0]:
        assert_(not q_0.nonzero())
    for q in [q_nan1, q_inf1, q_1, x, y, z, Q]:
        assert_(q.nonzero())
    # isnan
    for q in [q_nan1]:
        assert_(q.isnan())
    for q in [q_inf1, q_0, q_1, x, y, z, Q]:
        assert_(not q.isnan())
    # isinf
    for q in [q_inf1]:
        assert_(q.isinf())
    for q in [q_nan1, q_0, q_1, x, y, z, Q]:
        assert_(not q.isinf())
    # isfinite
    for q in [q_nan1, q_inf1]:
        assert_(not q.isfinite())
    for q in [q_0, q_1, x, y, z, Q]:
        assert_(q.isfinite())


    ## float returners
    # absolute
    assert_(np.isnan(q_nan1.abs()))
    for q,a in [(q_inf1, np.inf), (q_0,0.0), (q_1, 1.0), (x,1.0), (y,1.0), (z,1.0),
                (Q,np.sqrt(Q.w**2+Q.x**2+Q.y**2+Q.z**2))]:
        assert_(q.absolute() == a)
        assert_(q.abs() == a)


    ## quaternion returners
    # conjugate
    assert_(Q.conjugate() == Qbar);
    assert_(Q.conjugate().conjugate() == Q);


def test_quaternion_members():
    Q = quaternion.quaternion(1.1,2.2,3.3,4.4)
    assert_(Q.real==1.1)
    assert_(Q.w==1.1)
    assert_(Q.x==2.2)
    assert_(Q.y==3.3)
    assert_(Q.z==4.4)

def test_numpy_array_conversion():
    pass

if __name__=='__main__':
    test_quaternion_methods()
    test_quaternion_members()
    test_numpy_array_conversion()
