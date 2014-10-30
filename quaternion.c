/*
 * Quaternion math implementation
 * Copyright (c) 2011 Martin Ling
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NumPy Developers nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "quaternion.h"
#include <math.h>
#include <stdio.h>

#define _QUAT_EPS 1e-14

quaternion quaternion_create_from_spherical_coords(double vartheta, double varphi) {
  double ct = cos(vartheta/2.);
  double cp = cos(varphi/2.);
  double st = sin(vartheta/2.);
  double sp = sin(varphi/2.);
  return (quaternion) {cp*ct, -sp*st, st*cp, sp*ct};
}

quaternion quaternion_create_from_euler_angles(double alpha, double beta, double gamma) {
  double ca = cos(alpha/2.);
  double cb = cos(beta/2.);
  double cc = cos(gamma/2.);
  double sa = sin(alpha/2.);
  double sb = sin(beta/2.);
  double sc = sin(gamma/2.);
  return (quaternion) {ca*cb*cc-sa*cb*sc, ca*sb*sc-sa*sb*cc, ca*sb*cc+sa*sb*sc, sa*cb*cc+ca*cb*sc};
}


int
quaternion_nonzero(quaternion q)
{
  if(quaternion_isnan(q)) { return 1; }
  return ! (q.w == 0 && q.x == 0 && q.y == 0 && q.z == 0);
}

int
quaternion_isnan(quaternion q)
{
  return isnan(q.w) || isnan(q.x) || isnan(q.y) || isnan(q.z);
}

int
quaternion_isinf(quaternion q)
{
  return isinf(q.w) || isinf(q.x) || isinf(q.y) || isnan(q.z);
}

int
quaternion_isfinite(quaternion q)
{
  return isfinite(q.w) && isfinite(q.x) && isfinite(q.y) && isfinite(q.z);
}

double
quaternion_norm(quaternion q)
{
  return q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
}

double
quaternion_absolute(quaternion q)
{
  return sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

quaternion
quaternion_add(quaternion q1, quaternion q2)
{
  return (quaternion) {
    q1.w+q2.w,
      q1.x+q2.x,
      q1.y+q2.y,
      q1.z+q2.z,
      };
}

void quaternion_inplace_add(quaternion* q1, quaternion q2) {
  q1->w += q2.w;
  q1->x += q2.x;
  q1->y += q2.y;
  q1->z += q2.z;
  return;
}

quaternion
quaternion_subtract(quaternion q1, quaternion q2)
{
  return (quaternion) {
    q1.w-q2.w,
      q1.x-q2.x,
      q1.y-q2.y,
      q1.z-q2.z,
      };
}

void quaternion_inplace_subtract(quaternion* q1, quaternion q2) {
  q1->w -= q2.w;
  q1->x -= q2.x;
  q1->y -= q2.y;
  q1->z -= q2.z;
  return;
}

quaternion
quaternion_multiply(quaternion q1, quaternion q2)
{
  return (quaternion) {
    q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
      q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
      q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
      q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
      };
}

void quaternion_inplace_multiply(quaternion* q1a, quaternion q2) {
  quaternion q1 = *q1a;
  q1a->w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
  q1a->x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
  q1a->y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
  q1a->z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
  return;
}

quaternion
quaternion_divide(quaternion q1, quaternion q2)
{
  double q2norm = q2.w*q2.w + q2.x*q2.x + q2.y*q2.y + q2.z*q2.z;
  return (quaternion) {
    (  q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z) / q2norm,
    (- q1.w*q2.x + q1.x*q2.w - q1.y*q2.z + q1.z*q2.y) / q2norm,
    (- q1.w*q2.y + q1.x*q2.z + q1.y*q2.w - q1.z*q2.x) / q2norm,
    (- q1.w*q2.z - q1.x*q2.y + q1.y*q2.x + q1.z*q2.w) / q2norm
  };
}

void quaternion_inplace_divide(quaternion* q1a, quaternion q2) {
  double q2norm;
  quaternion q1 = *q1a;
  q2norm = q2.w*q2.w + q2.x*q2.x + q2.y*q2.y + q2.z*q2.z;
  q1a->w = ( q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z)/q2norm;
  q1a->x = (-q1.w*q2.x + q1.x*q2.w - q1.y*q2.z + q1.z*q2.y)/q2norm;
  q1a->y = (-q1.w*q2.y + q1.x*q2.z + q1.y*q2.w - q1.z*q2.x)/q2norm;
  q1a->z = (-q1.w*q2.z - q1.x*q2.y + q1.y*q2.x + q1.z*q2.w)/q2norm;
  return;
}

quaternion
quaternion_multiply_scalar(quaternion q, double s)
{
  return (quaternion) {s*q.w, s*q.x, s*q.y, s*q.z};
}

void quaternion_inplace_multiply_scalar(quaternion* q, double s) {
  q->w *= s;
  q->x *= s;
  q->y *= s;
  q->z *= s;
  return;
}

quaternion
quaternion_divide_scalar(quaternion q, double s)
{
  return (quaternion) {q.w/s, q.x/s, q.y/s, q.z/s};
}

void quaternion_inplace_divide_scalar(quaternion* q, double s) {
  q->w /= s;
  q->x /= s;
  q->y /= s;
  q->z /= s;
  return;
}

quaternion
quaternion_log(quaternion q)
{
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _QUAT_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      fprintf(stderr, "Input quaternion(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
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

quaternion
quaternion_normalized(quaternion q)
{
  double q_abs = quaternion_absolute(q);
  return (quaternion) {q.w/q_abs, q.x/q_abs, q.y/q_abs, q.z/q_abs};
}

quaternion
quaternion_power(quaternion q, quaternion p)
{
  return quaternion_exp(quaternion_multiply(quaternion_log(q), p));
}

void quaternion_inplace_power(quaternion* q1, quaternion q2) {
  quaternion q3 = quaternion_power(*q1,q2);
  *q1 = q3;
  return;
}

quaternion
quaternion_power_scalar(quaternion q, double p)
{
  return quaternion_exp(quaternion_multiply_scalar(quaternion_log(q), p));
}

void quaternion_inplace_power_scalar(quaternion* q, double s) {
  quaternion q2 = quaternion_power_scalar(*q,s);
  *q = q2;
  return;
}

quaternion
quaternion_negative(quaternion q)
{
  return (quaternion) {-q.w, -q.x, -q.y, -q.z};
}

quaternion
quaternion_conjugate(quaternion q)
{
  return (quaternion) {q.w, -q.x, -q.y, -q.z};
}

quaternion
quaternion_copysign(quaternion q1, quaternion q2)
{
  return (quaternion) {
    copysign(q1.w, q2.w),
      copysign(q1.x, q2.x),
      copysign(q1.y, q2.y),
      copysign(q1.z, q2.z)
      };
}

int
quaternion_equal(quaternion q1, quaternion q2)
{
  return
    !quaternion_isnan(q1) &&
    !quaternion_isnan(q2) &&
    q1.w == q2.w &&
    q1.x == q2.x &&
    q1.y == q2.y &&
    q1.z == q2.z;
}

int
quaternion_not_equal(quaternion q1, quaternion q2)
{
  return !quaternion_equal(q1, q2);
}

int
quaternion_less(quaternion q1, quaternion q2)
{
  return
    (!quaternion_isnan(q1) && !quaternion_isnan(q2))
    &&
    (q1.w != q2.w ? q1.w < q2.w :
     q1.x != q2.x ? q1.x < q2.x :
     q1.y != q2.y ? q1.y < q2.y :
     q1.z != q2.z ? q1.z < q2.z : 0);
}

int
quaternion_greater(quaternion q1, quaternion q2)
{
  return
    (!quaternion_isnan(q1) && !quaternion_isnan(q2))
    &&
    (q1.w != q2.w ? q1.w > q2.w :
     q1.x != q2.x ? q1.x > q2.x :
     q1.y != q2.y ? q1.y > q2.y :
     q1.z != q2.z ? q1.z > q2.z : 0);
}

int
quaternion_less_equal(quaternion q1, quaternion q2)
{
  return
    (!quaternion_isnan(q1) && !quaternion_isnan(q2))
    &&
    (q1.w != q2.w ? q1.w < q2.w :
     q1.x != q2.x ? q1.x < q2.x :
     q1.y != q2.y ? q1.y < q2.y :
     q1.z != q2.z ? q1.z < q2.z : 1);
  // Note that the final possibility is 1, whereas in
  // `quaternion_less` it was 0.  This distinction correctly
  // accounts for equality.
}

int
quaternion_greater_equal(quaternion q1, quaternion q2)
{
  return
    (!quaternion_isnan(q1) && !quaternion_isnan(q2))
    &&
    (q1.w != q2.w ? q1.w > q2.w :
     q1.x != q2.x ? q1.x > q2.x :
     q1.y != q2.y ? q1.y > q2.y :
     q1.z != q2.z ? q1.z > q2.z : 1);
  // Note that the final possibility is 1, whereas in
  // `quaternion_greater` it was 0.  This distinction correctly
  // accounts for equality.
}



double
rotor_intrinsic_distance(quaternion q1, quaternion q2)
{
  return 2*quaternion_absolute(quaternion_log(quaternion_divide(q1,q2)));
}

double
rotor_chordal_distance(quaternion q1, quaternion q2)
{
  return quaternion_absolute(quaternion_subtract(q1,q2));
}

#define SQRT_TWO 1.414213562373096

double
rotation_intrinsic_distance(quaternion q1, quaternion q2)
{
  if(rotor_chordal_distance(q1,q2)<=SQRT_TWO) {
    return 2*quaternion_absolute(quaternion_log(quaternion_divide(q1,q2)));
  } else {
    return 2*quaternion_absolute(quaternion_log(quaternion_divide(q1,quaternion_negative(q2))));
  }
}

double
rotation_chordal_distance(quaternion q1, quaternion q2)
{
  if(rotor_chordal_distance(q1,q2)<=SQRT_TWO) {
    return quaternion_absolute(quaternion_subtract(q1,q2));
  } else {
    return quaternion_absolute(quaternion_add(q1,q2));
  }
}

quaternion
slerp(quaternion q1, quaternion q2, double tau)
{
  if(rotor_chordal_distance(q1,q2)<=SQRT_TWO) {
    return quaternion_multiply( quaternion_power_scalar(quaternion_divide(q2,q1), tau), q1);
  } else {
    return quaternion_multiply( quaternion_power_scalar(quaternion_divide(quaternion_negative(q2),q1), tau), q1);
  }
}
