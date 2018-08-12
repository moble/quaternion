// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#ifndef __DUAL_QUATERNION_H__
#define __DUAL_QUATERNION_H__

#ifdef __cplusplus
extern "C" {
#endif

  #if defined(_MSC_VER)
    #include "math_msvc_compatibility.h"
  #else
    #include <math.h>
  #endif

  #include <Python.h>

  #define _DUAL_QUATERNION_EPS 1e-14

  #if defined(_MSC_VER)
    #define NPY_INLINE __inline
  #elif defined(__GNUC__)
    #if defined(__STRICT_ANSI__)
      #define NPY_INLINE __inline__
    #else
      #define NPY_INLINE inline
    #endif
  #else
    #define NPY_INLINE
  #endif

  typedef struct {
    double w;
    double x;
    double y;
    double z;
    double er;
    double ei;
    double ej;
    double ek;
  } dual_quaternion;

  // Unary bool returners
  static NPY_INLINE int dual_quaternion_isnan(dual_quaternion q) {
    return (isnan(q.w) || isnan(q.x) || isnan(q.y) || isnan(q.z)
            || isnan(q.er) || isnan(q.ei) || isnan(q.ej) || isnan(q.ek));

  }
  static NPY_INLINE int dual_quaternion_nonzero(dual_quaternion q) {
    if(dual_quaternion_isnan(q)) { return 1; }
    return ! (q.w == 0 && q.x == 0 && q.y == 0 && q.z == 0
              && q.er == 0 && q.ei == 0 && q.ej == 0 && q.ek == 0);
  }
  static NPY_INLINE int dual_quaternion_isinf(dual_quaternion q) {
    return (isinf(q.w) || isinf(q.x) || isinf(q.y) || isinf(q.z)
            || isinf(q.er) || isinf(q.ei) || isinf(q.ej) || isinf(q.ek));
  }
  static NPY_INLINE int dual_quaternion_isfinite(dual_quaternion q) {
    return (isfinite(q.w) && isfinite(q.x) && isfinite(q.y) && isfinite(q.z) &&
            isfinite(q.er) && isfinite(q.ei) && isfinite(q.ej) && isfinite(q.ek));
  }

  // Binary bool returners
  static NPY_INLINE int dual_quaternion_equal(dual_quaternion q1, dual_quaternion q2) {
    return
      !dual_quaternion_isnan(q1) &&
      !dual_quaternion_isnan(q2) &&
      q1.w == q2.w &&
      q1.x == q2.x &&
      q1.y == q2.y &&
      q1.z == q2.z &&
      q1.er == q2.er &&
      q1.ei == q2.ei &&
      q1.ej == q2.ej &&
      q1.ek == q2.ek;
  }
  static NPY_INLINE int dual_quaternion_not_equal(dual_quaternion q1, dual_quaternion q2) {
    return !dual_quaternion_equal(q1, q2);
  }
  static NPY_INLINE int dual_quaternion_less(dual_quaternion q1, dual_quaternion q2) {
    return
      (!dual_quaternion_isnan(q1) && !dual_quaternion_isnan(q2))
      &&
      (q1.w != q2.w ? q1.w < q2.w :
       q1.x != q2.x ? q1.x < q2.x :
       q1.y != q2.y ? q1.y < q2.y :
       q1.z != q2.z ? q1.z < q2.z :
       q1.er != q2.er ? q1.er < q2.er :
       q1.ei != q2.ei ? q1.ei < q2.ei :
       q1.ej != q2.ej ? q1.ej < q2.ej :
       q1.ek != q2.ek ? q1.ek < q2.ek : 0);
  }
  static NPY_INLINE int dual_quaternion_greater(dual_quaternion q1, dual_quaternion q2) {
    return
      (!dual_quaternion_isnan(q1) && !dual_quaternion_isnan(q2))
      &&
      (q1.w != q2.w ? q1.w > q2.w :
       q1.x != q2.x ? q1.x > q2.x :
       q1.y != q2.y ? q1.y > q2.y :
       q1.z != q2.z ? q1.z > q2.z :
       q1.er != q2.er ? q1.er > q2.er :
       q1.ei != q2.ei ? q1.ei > q2.ei :
       q1.ej != q2.ej ? q1.ej > q2.ej :
       q1.ek != q2.ek ? q1.ek > q2.ek : 0);
  }
  static NPY_INLINE int dual_quaternion_less_equal(dual_quaternion q1, dual_quaternion q2) {
    return
      (!dual_quaternion_isnan(q1) && !dual_quaternion_isnan(q2))
      &&
      (q1.w != q2.w ? q1.w < q2.w :
       q1.x != q2.x ? q1.x < q2.x :
       q1.y != q2.y ? q1.y < q2.y :
       q1.z != q2.z ? q1.z < q2.z :
       q1.er != q2.er ? q1.er < q2.er :
       q1.ei != q2.ei ? q1.ei < q2.ei :
       q1.ej != q2.ej ? q1.ej < q2.ej :
       q1.ek != q2.ek ? q1.ek < q2.ek : 1);
    // Note that the final possibility is 1, whereas in
    // `dual_quaternion_less` it was 0.  This distinction correctly
    // accounts for equality.
  }
  static NPY_INLINE int dual_quaternion_greater_equal(dual_quaternion q1, dual_quaternion q2) {
    return
      (!dual_quaternion_isnan(q1) && !dual_quaternion_isnan(q2))
      &&
      (q1.w != q2.w ? q1.w > q2.w :
       q1.x != q2.x ? q1.x > q2.x :
       q1.y != q2.y ? q1.y > q2.y :
       q1.z != q2.z ? q1.z > q2.z :
       q1.er != q2.er ? q1.er > q2.er :
       q1.ei != q2.ei ? q1.ei > q2.ei :
       q1.ej != q2.ej ? q1.ej > q2.ej :
       q1.ek != q2.ek ? q1.ek > q2.ek : 1);
    // Note that the final possibility is 1, whereas in
    // `dual_quaternion_greater` it was 0.  This distinction correctly
    // accounts for equality.
  }

  // Unary float returners
  static NPY_INLINE PyObject * dual_quaternion_norm(dual_quaternion q) {
    dual_quaternion q_conj = {q.w, -q.x, -q.y, -q.z, q.er, -q.ei, -q.ej, -q.ek};
    double scalar = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    double w = q_conj.w*q.er - q_conj.x*q.ei - q_conj.y*q.ej - q_conj.z*q.ek + q_conj.er*q.w - q_conj.ei*q.x - q_conj.ej*q.y - q_conj.ek*q.z;
    double vector = (w)/(2*scalar);
    return PyComplex_FromDoubles(scalar, vector);
  }
  static NPY_INLINE double dual_quaternion_absolute(dual_quaternion q) {
    PyErr_SetNone(PyExc_NotImplementedError);
    return 0;
  }
  static NPY_INLINE double dual_quaternion_angle(dual_quaternion q) {
    PyErr_SetNone(PyExc_NotImplementedError);
    return 0;
  }
  dual_quaternion dual_quaternion_sqrt(dual_quaternion q);
  dual_quaternion dual_quaternion_log(dual_quaternion q);
  dual_quaternion dual_quaternion_exp(dual_quaternion q);
  static NPY_INLINE dual_quaternion dual_quaternion_normalized(dual_quaternion q) {
    PyErr_SetNone(PyExc_NotImplementedError);
    dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_negative(dual_quaternion q) {
    dual_quaternion r = {-q.w, -q.x, -q.y, -q.z, -q.er, -q.ei, -q.ej, -q.ek};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_conjugate(dual_quaternion q) {
    dual_quaternion r = {q.w, -q.x, -q.y, -q.z, q.er, -q.ei, -q.ej, -q.ek};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_inverse(dual_quaternion q) {
    PyErr_SetNone(PyExc_NotImplementedError);
    dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
  }

  // dual_quaternion-dual_quaternion binary dual_quaternion returners
  static NPY_INLINE dual_quaternion dual_quaternion_copysign(dual_quaternion q1, dual_quaternion q2) {
    dual_quaternion r = {
      copysign(q1.w, q2.w),
      copysign(q1.x, q2.x),
      copysign(q1.y, q2.y),
      copysign(q1.z, q2.z),
      copysign(q1.er, q2.er),
      copysign(q1.ei, q2.ei),
      copysign(q1.ej, q2.ej),
      copysign(q1.ek, q2.ek)
    };
    return r;
  }

  // dual_quaternion-dual_quaternion/dual_quaternion-scalar binary dual_quaternion returners
  static NPY_INLINE dual_quaternion dual_quaternion_add(dual_quaternion q1, dual_quaternion q2) {
    dual_quaternion r = {
      q1.w+q2.w,
      q1.x+q2.x,
      q1.y+q2.y,
      q1.z+q2.z,
      q1.er+q2.er,
      q1.ei+q2.ei,
      q1.ej+q2.ej,
      q1.ek+q2.ek,
    };
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_add(dual_quaternion* q1, dual_quaternion q2) {
    q1->w += q2.w;
    q1->x += q2.x;
    q1->y += q2.y;
    q1->z += q2.z;
    q1->er += q2.er;
    q1->ei += q2.ei;
    q1->ej += q2.ej;
    q1->ek += q2.ek;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_scalar_add(double s, dual_quaternion q) {
    dual_quaternion r = {s+q.w, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_scalar_add(double s, dual_quaternion* q) {
    q->w += s;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_add_scalar(dual_quaternion q, double s) {
    dual_quaternion r = {s+q.w, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_add_scalar(dual_quaternion* q, double s) {
    q->w += s;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_subtract(dual_quaternion q1, dual_quaternion q2) {
    dual_quaternion r = {
      q1.w-q2.w,
      q1.x-q2.x,
      q1.y-q2.y,
      q1.z-q2.z,
      q1.er-q2.er,
      q1.ei-q2.ei,
      q1.ej-q2.ej,
      q1.ek-q2.ek,
    };
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_subtract(dual_quaternion* q1, dual_quaternion q2) {
    q1->w -= q2.w;
    q1->x -= q2.x;
    q1->y -= q2.y;
    q1->z -= q2.z;
    q1->er -= q2.er;
    q1->ei -= q2.ei;
    q1->ej -= q2.ej;
    q1->ek -= q2.ek;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_scalar_subtract(double s, dual_quaternion q) {
    dual_quaternion r = {s-q.w, -q.x, -q.y, -q.z, -q.er, -q.ei, -q.ej, -q.ek};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_subtract_scalar(dual_quaternion q, double s) {
    dual_quaternion r = {q.w-s, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_subtract_scalar(dual_quaternion* q, double s) {
    q->w -= s;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_multiply(dual_quaternion q1, dual_quaternion q2) {
    dual_quaternion r = {
      q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
      q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
      q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
      q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
      (q1.w*q2.er - q1.x*q2.ei - q1.y*q2.ej - q1.z*q2.ek)+(q1.er*q2.w - q1.ei*q2.x - q1.ej*q2.y - q1.ek*q2.z),
      (q1.w*q2.ei + q1.x*q2.er + q1.y*q2.ek - q1.z*q2.ej)+(q1.er*q2.x + q1.ei*q2.w + q1.ej*q2.z - q1.ek*q2.y),
      (q1.w*q2.ej - q1.x*q2.ek + q1.y*q2.er + q1.z*q2.ei)+(q1.er*q2.y - q1.ei*q2.z + q1.ej*q2.w + q1.ek*q2.x),
      (q1.w*q2.ek + q1.x*q2.ej - q1.y*q2.ei + q1.z*q2.er)+(q1.er*q2.z + q1.ei*q2.y - q1.ej*q2.x + q1.ek*q2.w),
    };
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_multiply(dual_quaternion* q1a, dual_quaternion q2) {
    dual_quaternion q1 = {q1a->w, q1a->x, q1a->y, q1a->z};
    q1a->w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
    q1a->x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
    q1a->y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
    q1a->z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
    q1a->er = (q1.w*q2.er - q1.x*q2.ei - q1.y*q2.ej - q1.z*q2.ek)+(q1.er*q2.w - q1.ei*q2.x - q1.ej*q2.y - q1.ek*q2.z);
    q1a->ei = (q1.w*q2.ei + q1.x*q2.er + q1.y*q2.ek - q1.z*q2.ej)+(q1.er*q2.x + q1.ei*q2.w + q1.ej*q2.z - q1.ek*q2.y);
    q1a->ej = (q1.w*q2.ej - q1.x*q2.ek + q1.y*q2.er + q1.z*q2.ei)+(q1.er*q2.y - q1.ei*q2.z + q1.ej*q2.w + q1.ek*q2.x);
    q1a->ek = (q1.w*q2.ek + q1.x*q2.ej - q1.y*q2.ei + q1.z*q2.er)+(q1.er*q2.z + q1.ei*q2.y - q1.ej*q2.x + q1.ek*q2.w);
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_scalar_multiply(double s, dual_quaternion q) {
    dual_quaternion r = {s*q.w, s*q.x, s*q.y, s*q.z, s*q.er, s*q.ei, s*q.ej, s*q.ek};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_scalar_multiply(double s, dual_quaternion* q) {
    q->w *= s;
    q->x *= s;
    q->y *= s;
    q->z *= s;
    q->er *= s;
    q->ei *= s;
    q->ej *= s;
    q->ek *= s;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_multiply_scalar(dual_quaternion q, double s) {
    dual_quaternion r = {s*q.w, s*q.x, s*q.y, s*q.z, s*q.er, s*q.ei, s*q.ej, s*q.ek};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_multiply_scalar(dual_quaternion* q, double s) {
    q->w *= s;
    q->x *= s;
    q->y *= s;
    q->z *= s;
    q->er *= s;
    q->ei *= s;
    q->ej *= s;
    q->ek *= s;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_divide(dual_quaternion q1, dual_quaternion q2) {
    PyErr_SetNone(PyExc_NotImplementedError);
    dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_divide(dual_quaternion* q1a, dual_quaternion q2) {
    PyErr_SetNone(PyExc_NotImplementedError);
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_scalar_divide(double s, dual_quaternion q) {
    PyErr_SetNone(PyExc_NotImplementedError);
    dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
  }
  /* The following function is impossible, but listed for completeness: */
  /* static NPY_INLINE void dual_quaternion_inplace_scalar_divide(double* sa, dual_quaternion q2) { } */
  static NPY_INLINE dual_quaternion dual_quaternion_divide_scalar(dual_quaternion q, double s) {
    dual_quaternion r = {q.w/s, q.x/s, q.y/s, q.z/s, q.er/s, q.ei/s, q.ej/s, q.ek/s};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_divide_scalar(dual_quaternion* q, double s) {
    q->w /= s;
    q->x /= s;
    q->y /= s;
    q->z /= s;
    q->er /= s;
    q->ei /= s;
    q->ej /= s;
    q->ek /= s;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_power(dual_quaternion q, dual_quaternion p) {
    PyErr_SetNone(PyExc_NotImplementedError);
    dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_power(dual_quaternion* q1, dual_quaternion q2) {
    PyErr_SetNone(PyExc_NotImplementedError);
    return;
  }
  dual_quaternion dual_quaternion_scalar_power(double s, dual_quaternion q);
  static NPY_INLINE void dual_quaternion_inplace_scalar_power(double s, dual_quaternion* q) {
    PyErr_SetNone(PyExc_NotImplementedError);
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_power_scalar(dual_quaternion q, double s) {
    PyErr_SetNone(PyExc_NotImplementedError);
    dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_power_scalar(dual_quaternion* q, double s) {
    PyErr_SetNone(PyExc_NotImplementedError);
    return;
  }


#ifdef __cplusplus
}
#endif

#endif // __DUAL_QUATERNION_H__
