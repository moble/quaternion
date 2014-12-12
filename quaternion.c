// Copyright (c) 2014, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#include <math.h>
#include <stdio.h>

#include "quaternion.h"

#define _QUAT_EPS 1e-14

quaternion
quaternion_create_from_spherical_coords(double vartheta, double varphi) {
  double ct = cos(vartheta/2.);
  double cp = cos(varphi/2.);
  double st = sin(vartheta/2.);
  double sp = sin(varphi/2.);
  return (quaternion) {cp*ct, -sp*st, st*cp, sp*ct};
}

quaternion
quaternion_create_from_euler_angles(double alpha, double beta, double gamma) {
  double ca = cos(alpha/2.);
  double cb = cos(beta/2.);
  double cc = cos(gamma/2.);
  double sa = sin(alpha/2.);
  double sb = sin(beta/2.);
  double sc = sin(gamma/2.);
  return (quaternion) {ca*cb*cc-sa*cb*sc, ca*sb*sc-sa*sb*cc, ca*sb*cc+sa*sb*sc, sa*cb*cc+ca*cb*sc};
}

quaternion
quaternion_sqrt(quaternion q)
{
  double c;
  double absolute = quaternion_absolute(q);
  if(fabs(1+q.w/absolute)<_QUAT_EPS*absolute) {
    return (quaternion) {0.0, 1.0, 0.0, 0.0};
  }
  c = sqrt(absolute/(2+2*q.w/absolute));
  return (quaternion) {(1.0+q.w/absolute)*c, q.x*c/absolute, q.y*c/absolute, q.z*c/absolute};
}

quaternion
quaternion_log(quaternion q)
{
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _QUAT_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      // fprintf(stderr, "Input quaternion(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
      if(fabs(q.w+1)>_QUAT_EPS) {
        return (quaternion) {log(-q.w), M_PI, 0., 0.};
      }
      return (quaternion) {0., M_PI, 0., 0.};
    }
    return (quaternion) {log(q.w), 0., 0., 0.};
  } else {
    double v = atan2(b, q.w);
    double f = v/b;
    return (quaternion) { log(q.w*q.w+b*b)/2.0, f*q.x, f*q.y, f*q.z };
  }
}

quaternion
quaternion_exp(quaternion q)
{
  double vnorm = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if (vnorm > _QUAT_EPS) {
    double s = sin(vnorm) / vnorm;
    double e = exp(q.w);
    return (quaternion) {e*cos(vnorm), e*s*q.x, e*s*q.y, e*s*q.z};
  } else {
    return (quaternion) {exp(q.w), 0, 0, 0};
  }
}
