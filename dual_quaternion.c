// Copyright (c) 2017, Michael Boyle
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
#include <float.h>

#include "dual_quaternion.h"

dual_quaternion
dual_quaternion_create_from_spherical_coords(double vartheta, double varphi) {
  double ct = cos(vartheta/2.);
  double cp = cos(varphi/2.);
  double st = sin(vartheta/2.);
  double sp = sin(varphi/2.);
  dual_quaternion r = {cp*ct, -sp*st, st*cp, sp*ct};
  return r;
}

dual_quaternion
dual_quaternion_create_from_euler_angles(double alpha, double beta, double gamma) {
  double ca = cos(alpha/2.);
  double cb = cos(beta/2.);
  double cc = cos(gamma/2.);
  double sa = sin(alpha/2.);
  double sb = sin(beta/2.);
  double sc = sin(gamma/2.);
  dual_quaternion r = {ca*cb*cc-sa*cb*sc, ca*sb*sc-sa*sb*cc, ca*sb*cc+sa*sb*sc, sa*cb*cc+ca*cb*sc};
  return r;
}

dual_quaternion
dual_quaternion_sqrt(dual_quaternion q)
{
  double absolute = dual_quaternion_norm(q);  // pre-square-root
  if(absolute<=DBL_MIN) {
      dual_quaternion r = {0.0, 0.0, 0.0, 0.0};
      return r;
  }
  absolute = sqrt(absolute);
  if(fabs(absolute+q.w)<_dual_quaternion_EPS*absolute) {
    dual_quaternion r = {0.0, sqrt(absolute), 0.0, 0.0};
    return r;
  } else {
    double c = sqrt(0.5/(absolute+q.w));
    dual_quaternion r = {(absolute+q.w)*c, q.x*c, q.y*c, q.z*c};
    return r;
  }
}

dual_quaternion
dual_quaternion_log(dual_quaternion q)
{
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _dual_quaternion_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      // fprintf(stderr, "Input dual_quaternion(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
      if(fabs(q.w+1)>_dual_quaternion_EPS) {
        dual_quaternion r = {log(-q.w), M_PI, 0., 0.};
        return r;
      } else {
        dual_quaternion r = {0., M_PI, 0., 0.};
        return r;
      }
    } else {
      dual_quaternion r = {log(q.w), 0., 0., 0.};
      return r;
    }
  } else {
    double v = atan2(b, q.w);
    double f = v/b;
    dual_quaternion r = { log(q.w*q.w+b*b)/2.0, f*q.x, f*q.y, f*q.z };
    return r;
  }
}

double
_dual_quaternion_scalar_log(double s) { return log(s); }

dual_quaternion
dual_quaternion_scalar_power(double s, dual_quaternion q)
{
  /* Unlike the dual_quaternion^dual_quaternion power, this is unambiguous. */
  if(s==0.0) { /* log(s)=-inf */
    if(! dual_quaternion_nonzero(q)) {
      dual_quaternion r = {1.0, 0.0, 0.0, 0.0}; /* consistent with python */
      return r;
    } else {
      dual_quaternion r = {0.0, 0.0, 0.0, 0.0}; /* consistent with python */
      return r;
    }
  } else if(s<0.0) { /* log(s)=nan */
    // fprintf(stderr, "Input scalar (%.15g) has no unique logarithm; returning one arbitrarily.", s);
    dual_quaternion t = {log(-s), M_PI, 0, 0};
    return dual_quaternion_exp(dual_quaternion_multiply(q, t));
  }
  return dual_quaternion_exp(dual_quaternion_multiply_scalar(q, log(s)));
}

dual_quaternion
dual_quaternion_exp(dual_quaternion q)
{
  double vnorm = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if (vnorm > _dual_quaternion_EPS) {
    double s = sin(vnorm) / vnorm;
    double e = exp(q.w);
    dual_quaternion r = {e*cos(vnorm), e*s*q.x, e*s*q.y, e*s*q.z};
    return r;
  } else {
    dual_quaternion r = {exp(q.w), 0, 0, 0};
    return r;
  }
}

#ifdef __cplusplus
}
#endif
