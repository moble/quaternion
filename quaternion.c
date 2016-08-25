// Copyright (c) 2016, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
  #include "math_msvc_compatibility.h"
#else
  #include <math.h>
#endif

#include <stdio.h>

#include "quaternion.h"

quaternion
quaternion_create_from_spherical_coords(double vartheta, double varphi) {
  double ct = cos(vartheta/2.);
  double cp = cos(varphi/2.);
  double st = sin(vartheta/2.);
  double sp = sin(varphi/2.);
  quaternion r = {cp*ct, -sp*st, st*cp, sp*ct};
  return r;
}

quaternion
quaternion_create_from_euler_angles(double alpha, double beta, double gamma) {
  double ca = cos(alpha/2.);
  double cb = cos(beta/2.);
  double cc = cos(gamma/2.);
  double sa = sin(alpha/2.);
  double sb = sin(beta/2.);
  double sc = sin(gamma/2.);
  quaternion r = {ca*cb*cc-sa*cb*sc, ca*sb*sc-sa*sb*cc, ca*sb*cc+sa*sb*sc, sa*cb*cc+ca*cb*sc};
  return r;
}

quaternion
quaternion_sqrt(quaternion q)
{
  double absolute = quaternion_absolute(q);
  if(fabs(1+q.w/absolute)<_QUATERNION_EPS*absolute) {
    quaternion r = {0.0, 1.0, 0.0, 0.0};
    return r;
  } else {
    double c = sqrt(absolute/(2+2*q.w/absolute));
    quaternion r = {(1.0+q.w/absolute)*c, q.x*c/absolute, q.y*c/absolute, q.z*c/absolute};
    return r;
  }
}

quaternion
quaternion_log(quaternion q)
{
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _QUATERNION_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      // fprintf(stderr, "Input quaternion(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
      if(fabs(q.w+1)>_QUATERNION_EPS) {
        quaternion r = {log(-q.w), M_PI, 0., 0.};
        return r;
      } else {
        quaternion r = {0., M_PI, 0., 0.};
        return r;
      }
    } else {
      quaternion r = {log(q.w), 0., 0., 0.};
      return r;
    }
  } else {
    double v = atan2(b, q.w);
    double f = v/b;
    quaternion r = { log(q.w*q.w+b*b)/2.0, f*q.x, f*q.y, f*q.z };
    return r;
  }
}

double
_quaternion_scalar_log(double s) { return log(s); }

quaternion
quaternion_scalar_power(double s, quaternion q)
{
  /* Unlike the quaternion^quaternion power, this is unambiguous. */
  if(s==0.0) { /* log(s)=-inf */
    if(! quaternion_nonzero(q)) {
      quaternion r = {1.0, 0.0, 0.0, 0.0}; /* consistent with python */
      return r;
    } else {
      quaternion r = {0.0, 0.0, 0.0, 0.0}; /* consistent with python */
      return r;
    }
  } else if(s<0.0) { /* log(s)=nan */
    // fprintf(stderr, "Input scalar (%.15g) has no unique logarithm; returning one arbitrarily.", s);
    quaternion t = {log(-s), M_PI, 0, 0};
    return quaternion_exp(quaternion_multiply(q, t));
  }
  return quaternion_exp(quaternion_multiply_scalar(q, log(s)));
}

quaternion
quaternion_exp(quaternion q)
{
  double vnorm = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if (vnorm > _QUATERNION_EPS) {
    double s = sin(vnorm) / vnorm;
    double e = exp(q.w);
    quaternion r = {e*cos(vnorm), e*s*q.x, e*s*q.y, e*s*q.z};
    return r;
  } else {
    quaternion r = {exp(q.w), 0, 0, 0};
    return r;
  }
}

#ifdef __cplusplus
}
#endif
