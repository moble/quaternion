#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

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

eps = np.finfo(float).eps


@pytest.mark.skipif(not has_scipy, reason="Scipy is not installed")
def test_subset_interpolation():
    from quaternion.calculus import spline
    t = np.linspace(0, 10, 100)
    f = np.sin(t)

    for i1, i2 in [[0, 100], [0, -1], [10, -10], [11, -10], [10, -11], [11, -11], [21, -21], [31, -31]]:
        f_out = spline(f, t, t_out=t[i1:i2])
        f_sub = f[i1:i2]
        assert np.allclose(f_out, f_sub, atol=2*eps, rtol=2*eps)


@pytest.mark.skipif(not has_scipy, reason="Scipy is not installed")
def test_differentiation():
    from quaternion.calculus import spline
    t = np.linspace(0, 10, 1000)

    f = np.sin(t)
    fprime = np.cos(t)
    assert np.allclose(spline(f, t, spline_degree=5, derivative_order=1), fprime, atol=1e-11, rtol=1e-11)

    f = np.exp(1j*t)
    fprime = 1j*np.exp(1j*t)
    assert np.allclose(spline(f, t, spline_degree=5, derivative_order=1), fprime, atol=1e-11, rtol=1e-11)


@pytest.mark.skipif(not has_scipy, reason="Scipy is not installed")
def test_antiderivative():
    from quaternion.calculus import spline
    t = np.linspace(0, 10, 1000)

    f = np.cos(t)
    fint = np.sin(t)
    assert np.allclose(spline(f, t, spline_degree=5, derivative_order=-1), fint, atol=1e-11, rtol=1e-11)

    f = np.exp(1j*t)
    fint = 1j*(1-np.exp(1j*t))
    assert np.allclose(spline(f, t, spline_degree=5, derivative_order=-1), fint, atol=1e-11, rtol=1e-11)


@pytest.mark.skipif(not has_scipy, reason="Scipy is not installed")
def test_integral():
    from quaternion.calculus import spline
    t = np.linspace(0, 10, 1000)

    f = np.cos(t)
    fint = np.sin(t)
    assert np.allclose(spline(f, t, spline_degree=5, definite_integral_bounds=(t[0], t[-1])),
                       fint[-1]-fint[0], atol=1e-11, rtol=1e-11)

    f = np.exp(1j*t)
    fint = 1j*(1-np.exp(1j*t))
    # print(max(abs(spline(f, t, spline_degree=5, derivative_order=-1)-fint)))
    assert np.allclose(spline(f, t, spline_degree=5, definite_integral_bounds=(t[0], t[-1])),
                       fint[-1]-fint[0], atol=1e-11, rtol=1e-11)
