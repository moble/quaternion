from __future__ import division, print_function, absolute_import
from .numba import njit, IS_PY3

if IS_PY3:
    xrange = range

@njit('void(f8[:],f8[:],f8[:])')
def derivative(f, t, dfdt):
    """Fourth-order finite-differencing with non-uniform time steps

    The formula for this finite difference comes from Eq. (A 5b) of
    "Derivative formulas and errors for non-uniformly spaced points"
    by M. K. Bowen and Ronald Smith.  As explained in their Eqs. (B
    9b) and (B 10b), this is a fourth-order formula.  If there are
    fewer than five points, the function reverts to simpler
    (lower-order) formulas.

    """

    for i in xrange(2):
        t_i = t[i]
        f1 = f[0]
        f2 = f[1]
        f3 = f[2]
        f4 = f[3]
        f5 = f[4]
        t1 = t[0]
        t2 = t[1]
        t3 = t[2]
        t4 = t[3]
        t5 = t[4]
        h1 = t1 - t_i
        h2 = t2 - t_i
        h3 = t3 - t_i
        h4 = t4 - t_i
        h5 = t5 - t_i
        h12 = t1 - t2
        h13 = t1 - t3
        h14 = t1 - t4
        h15 = t1 - t5
        h23 = t2 - t3
        h24 = t2 - t4
        h25 = t2 - t5
        h34 = t3 - t4
        h35 = t3 - t5
        h45 = t4 - t5
        dfdt[i] = (-(h2*h3*h4 +h2*h3*h5 +h2*h4*h5 +h3*h4*h5)*f1/((h12)*(h13)*(h14)*(h15))
                   +(h1*h3*h4 + h1*h3*h5 + h1*h4*h5 + h3*h4*h5)*f2/((h12)*(h23)*(h24)*(h25))
                   -(h1*h2*h4 + h1*h2*h5 + h1*h4*h5 + h2*h4*h5)*f3/((h13)*(h23)*(h34)*(h35))
                   +(h1*h2*h3 + h1*h2*h5 + h1*h3*h5 + h2*h3*h5)*f4/((h14)*(h24)*(h34)*(h45))
                   -(h1*h2*h3 + h1*h2*h4 + h1*h3*h4 + h2*h3*h4)*f5/((h15)*(h25)*(h35)*(h45)))

    for i in xrange(2,len(t)-2):
        f1 = f[i-2]
        f2 = f[i-1]
        f3 = f[i]
        f4 = f[i+1]
        f5 = f[i+2]
        t1 = t[i-2]
        t2 = t[i-1]
        t3 = t[i]
        t4 = t[i+1]
        t5 = t[i+2]
        h1 = t1 - t3
        h2 = t2 - t3
        h4 = t4 - t3
        h5 = t5 - t3
        h12 = t1 - t2
        h13 = t1 - t3
        h14 = t1 - t4
        h15 = t1 - t5
        h23 = t2 - t3
        h24 = t2 - t4
        h25 = t2 - t5
        h34 = t3 - t4
        h35 = t3 - t5
        h45 = t4 - t5
        dfdt[i] = (-(h2*h4*h5)*f1/(h12*h13*h14*h15)
                   +(h1*h4*h5)*f2/(h12*h23*h24*h25)
                   -(h1*h2*h4 + h1*h2*h5 + h1*h4*h5 + h2*h4*h5)*f3/((h13)*(h23)*(h34)*(h35))
                   +(h1*h2*h5)*f4/(h14*h24*h34*h45)
                   -(h1*h2*h4)*f5/(h15*h25*h35*h45))

    for i in xrange(len(t)-2,len(t)):
        t_i = t[i]
        f1 = f[-5]
        f2 = f[-4]
        f3 = f[-3]
        f4 = f[-2]
        f5 = f[-1]
        t1 = t[-5]
        t2 = t[-4]
        t3 = t[-3]
        t4 = t[-2]
        t5 = t[-1]
        h1 = t1 - t_i
        h2 = t2 - t_i
        h3 = t3 - t_i
        h4 = t4 - t_i
        h5 = t5 - t_i
        h12 = t1 - t2
        h13 = t1 - t3
        h14 = t1 - t4
        h15 = t1 - t5
        h23 = t2 - t3
        h24 = t2 - t4
        h25 = t2 - t5
        h34 = t3 - t4
        h35 = t3 - t5
        h45 = t4 - t5
        dfdt[i] = (-(h2*h3*h4 +h2*h3*h5 +h2*h4*h5 +h3*h4*h5)*f1/((h12)*(h13)*(h14)*(h15))
                   +(h1*h3*h4 + h1*h3*h5 + h1*h4*h5 + h3*h4*h5)*f2/((h12)*(h23)*(h24)*(h25))
                   -(h1*h2*h4 + h1*h2*h5 + h1*h4*h5 + h2*h4*h5)*f3/((h13)*(h23)*(h34)*(h35))
                   +(h1*h2*h3 + h1*h2*h5 + h1*h3*h5 + h2*h3*h5)*f4/((h14)*(h24)*(h34)*(h45))
                   -(h1*h2*h3 + h1*h2*h4 + h1*h3*h4 + h2*h3*h4)*f5/((h15)*(h25)*(h35)*(h45)))

    return


@njit('void(f8[:,:], f8[:], f8[:,:])')
def indefinite_integral(f, t, Sfdt):
    Sfdt[0] = 0.0
    for i in xrange(1,len(t)):
        for j in xrange(f.shape[1]):
            Sfdt[i,j] = Sfdt[i-1,j] + (f[i,j]+f[i-1,j])*((t[i]-t[i-1])/2.0)
    return


@njit('void(f8[:,:], f8[:], f8[:])')
def definite_integral(f, t, Sfdt):
    for i in xrange(len(Sfdt)):
        Sfdt[i] = 0.0
    for i in xrange(1,f.shape[0]):
        for j in xrange(f.shape[1]):
            Sfdt[j] += (f[i,j]+f[i-1,j])*((t[i]-t[i-1])/2.0)
    return
