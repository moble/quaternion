#!/usr/bin/env python

from __future__ import division
from numpy import *
import quaternion
from numpy.testing import assert_


def test_member_funcs():
    q_inf1 = quaternion.quaternion(inf,0.,0.,0.)
    q_0   = quaternion.quaternion(0.,0.,0.,0.)
    q_1      = quaternion.quaternion(1.,0.,0.,0.)
    x        = quaternion.quaternion(0.,1.,0.,0.)
    y        = quaternion.quaternion(0.,0.,1.,0.)
    z        = quaternion.quaternion(0.,0.,0.,1.)
    Q = quaternion.quaternion(1.,2.,3.,4.)
    #
    for q in [q_0
    assert_(not zero.nonzero())
    for q in [q_1, x, y, z, Q]:
        assert_(q.nonzero())
    for q in [q_1, x, y, z, Q]:
        assert_(not q.isnan())
    for q in [q_1, x, y, z, Q]:
        assert_(not q.isinf())
    for q in [q_1, x, y, z, Q]:
        assert_(q.isfinite())
    # assert_(quaternion.quaternion_isnonzero(q1))
    # assert_()
    # assert_()
    # assert_()
    # assert_()
    # assert_()


def test_numpy_array_conversion():
    pass

if __name__=='__main__':
    test_member_funcs()
    test_numpy_array_conversion()
