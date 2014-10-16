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
#include "math.h"

#define _QUAT_EPS 1e-6

int
quaternion_isnonzero(quaternion q)
{
  return q.w != 0 || q.x != 0 || q.y != 0 || q.z != 0;
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

quaternion
quaternion_divide(quaternion q1, quaternion q2)
{
  double s = q2.w*q2.w + q2.x*q2.x + q2.y*q2.y + q2.z*q2.z;
  return (quaternion) {
    (  q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z) / s,
      (- q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y) / s,
      (- q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x) / s,
      (- q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w) / s
      };
}

quaternion
quaternion_multiply_scalar(quaternion q, double s)
{
  return (quaternion) {s*q.w, s*q.x, s*q.y, s*q.z};
}

quaternion
quaternion_divide_scalar(quaternion q, double s)
{
  return (quaternion) {q.w/s, q.x/s, q.y/s, q.z/s};
}

quaternion
quaternion_log(quaternion q)
{
  double sumvsq = q.x*q.x + q.y*q.y + q.z*q.z;
  double vnorm = sqrt(sumvsq);
  if (vnorm > _QUAT_EPS) {
    double m = sqrt(q.w*q.w + sumvsq);
    double s = acos(q.w/m) / vnorm;
    return (quaternion) {log(m), s*q.x, s*q.y, s*q.z};
  } else {
    return (quaternion) {0, 0, 0, 0};
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
quaternion_power(quaternion q, quaternion p)
{
  return quaternion_exp(quaternion_multiply(quaternion_log(q), p));
}

quaternion
quaternion_power_scalar(quaternion q, double p)
{
  return quaternion_exp(quaternion_multiply_scalar(quaternion_log(q), p));
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
    (!quaternion_isnan(q1) &&
     !quaternion_isnan(q2)) && (
                                q1.w != q2.w ? q1.w < q2.w :
                                q1.x != q2.x ? q1.x < q2.x :
                                q1.y != q2.y ? q1.y < q2.y :
                                q1.z != q2.z ? q1.z < q2.z : 0);
}

int
quaternion_less_equal(quaternion q1, quaternion q2)
{
  return
    (!quaternion_isnan(q1) &&
     !quaternion_isnan(q2)) && (
                                q1.w != q2.w ? q1.w < q2.w :
                                q1.x != q2.x ? q1.x < q2.x :
                                q1.y != q2.y ? q1.y < q2.y :
                                q1.z != q2.z ? q1.z < q2.z : 1);
}
