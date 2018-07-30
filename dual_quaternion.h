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

  // Constructor-ish
  dual_quaternion dual_quaternion_create_from_spherical_coords(double vartheta, double varphi);
  dual_quaternion dual_quaternion_create_from_euler_angles(double alpha, double beta, double gamma);

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
  dual_quaternion dual_quaternion_log(dual_quaternion q); // Pre-declare; declared again below, in its rightful place
  static NPY_INLINE PyObject * dual_quaternion_norm(dual_quaternion q) {
    dual_quaternion q_conj = {q.w, -q.x, -q.y, -q.z, q.er, -q.ei, -q.ej, -q.ek};
    double scalar = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    double w = q_conj.w*q.er - q_conj.x*q.ei - q_conj.y*q.ej - q_conj.z*q.ek - q_conj.er*q.w - q_conj.ei*q.x - q_conj.ej*q.y - q_conj.ek*q.z;
    double x = q_conj.w*q.ei + q_conj.x*q.er + q_conj.y*q.ek - q_conj.z*q.ej - q_conj.er*q.x + q_conj.ei*q.w + q_conj.ej*q.z - q_conj.ek*q.y;
    double y = q_conj.w*q.ej - q_conj.x*q.ek + q_conj.y*q.er + q_conj.z*q.ei - q_conj.er*q.y - q_conj.ei*q.z + q_conj.ej*q.w + q_conj.ek*q.x;
    double z = q_conj.w*q.ek + q_conj.x*q.ej - q_conj.y*q.ei + q_conj.z*q.er - q_conj.er*q.z + q_conj.ei*q.y - q_conj.ej*q.x + q_conj.ek*q.w;

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

  // Unary dual_quaternion returners
  dual_quaternion dual_quaternion_sqrt(dual_quaternion q);
  dual_quaternion dual_quaternion_log(dual_quaternion q);
  dual_quaternion dual_quaternion_exp(dual_quaternion q);
  static NPY_INLINE dual_quaternion dual_quaternion_normalized(dual_quaternion q) {
    double q_abs = dual_quaternion_absolute(q);
    dual_quaternion r = {q.w/q_abs, q.x/q_abs, q.y/q_abs, q.z/q_abs};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_x_parity_conjugate(dual_quaternion q) {
    dual_quaternion r = {q.w, q.x, -q.y, -q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_x_parity_symmetric_part(dual_quaternion q) {
    dual_quaternion r = {q.w, q.x, 0.0, 0.0};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_x_parity_antisymmetric_part(dual_quaternion q) {
    dual_quaternion r = {0.0, 0.0, q.y, q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_y_parity_conjugate(dual_quaternion q) {
    dual_quaternion r = {q.w, -q.x, q.y, -q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_y_parity_symmetric_part(dual_quaternion q) {
    dual_quaternion r = {q.w, 0.0, q.y, 0.0};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_y_parity_antisymmetric_part(dual_quaternion q) {
    dual_quaternion r = {0.0, q.x, 0.0, q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_z_parity_conjugate(dual_quaternion q) {
    dual_quaternion r = {q.w, -q.x, -q.y, q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_z_parity_symmetric_part(dual_quaternion q) {
    dual_quaternion r = {q.w, 0.0, 0.0, q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_z_parity_antisymmetric_part(dual_quaternion q) {
    dual_quaternion r = {0.0, q.x, q.y, 0.0};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_parity_conjugate(dual_quaternion q) {
    dual_quaternion r = {q.w, q.x, q.y, q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_parity_symmetric_part(dual_quaternion q) {
    dual_quaternion r = {q.w, q.x, q.y, q.z};
    return r;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_parity_antisymmetric_part(dual_quaternion q) {
    dual_quaternion r = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    (void) q; // This q parameter is unused, but here for consistency with similar functions
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

  // dual_quaternion-vector binary void returner
//  static inline void _cross(double a[], double b[], double c[]) {
//    c[0] = a[1]*b[2] - a[2]*b[1];
//    c[1] = a[2]*b[0] - a[0]*b[2];
//    c[2] = a[0]*b[1] - a[1]*b[0];
//    return;
//  }
//  static inline void _cross_times_scalar(double s, double a[], double b[], double c[]) {
//    c[0] = s*(a[1]*b[2] - a[2]*b[1]);
//    c[1] = s*(a[2]*b[0] - a[0]*b[2]);
//    c[2] = s*(a[0]*b[1] - a[1]*b[0]);
//    return;
//  }
  static NPY_INLINE void _sv_plus_rxv(dual_quaternion q, double v[], double w[]) {
    w[0] = q.w * v[0] + q.y*v[2] - q.z*v[1];
    w[1] = q.w * v[1] + q.z*v[0] - q.x*v[2];
    w[2] = q.w * v[2] + q.x*v[1] - q.y*v[0];
    return;
  }
  static NPY_INLINE void _v_plus_2rxvprime_over_m(dual_quaternion q, double v[], double w[], double two_over_m, double vprime[]) {
    vprime[0] = v[0] + two_over_m * (q.y*w[2] - q.z*w[1]);
    vprime[1] = v[1] + two_over_m * (q.z*w[0] - q.x*w[2]);
    vprime[2] = v[2] + two_over_m * (q.x*w[1] - q.y*w[0]);
    return;
  }
  static NPY_INLINE void dual_quaternion_rotate_vector(dual_quaternion q, double v[], double vprime[]) {
    // The most efficient formula I know of for rotating a vector by a dual_quaternion is
    //     v' = v + 2 * r x (s * v + r x v) / m
    // where x represents the cross product, s and r are the scalar and vector parts of the dual_quaternion, respectively,
    // and m is the sum of the squares of the components of the dual_quaternion.  This requires 22 multiplications and
    // 14 additions, as opposed to 32 and 24 for naive application of `q*v*q.conj()`.  In this function, I will further
    // reduce the operation count to 18 and 12 by skipping the normalization by `m`.  The full version will be
    // implemented in another function.
    double w[3];
    _sv_plus_rxv(q, v, w);
    _v_plus_2rxvprime_over_m(q, v, w, 2, vprime);
    return;
  }
  static NPY_INLINE void dual_quaternion_rotate_vector_and_normalize(dual_quaternion q, double v[], double vprime[]) {
    // This applies the algorithm described above, but also includes normalization of the dual_quaternion.
    double w[3];
    double m = q.x*q.x+q.y*q.y+q.z*q.z;
    _sv_plus_rxv(q, v, w);
    _v_plus_2rxvprime_over_m(q, v, w, 2/m, vprime);
    return;
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
    double q2norm = q2.w*q2.w + q2.x*q2.x + q2.y*q2.y + q2.z*q2.z;
    dual_quaternion r = {
      (  q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z) / q2norm,
      (- q1.w*q2.x + q1.x*q2.w - q1.y*q2.z + q1.z*q2.y) / q2norm,
      (- q1.w*q2.y + q1.x*q2.z + q1.y*q2.w - q1.z*q2.x) / q2norm,
      (- q1.w*q2.z - q1.x*q2.y + q1.y*q2.x + q1.z*q2.w) / q2norm
    };
    return r;
  }
  static NPY_INLINE void dual_quaternion_inplace_divide(dual_quaternion* q1a, dual_quaternion q2) {
    double q2norm;
    dual_quaternion q1 = *q1a;
    q2norm = q2.w*q2.w + q2.x*q2.x + q2.y*q2.y + q2.z*q2.z;
    q1a->w = ( q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z)/q2norm;
    q1a->x = (-q1.w*q2.x + q1.x*q2.w - q1.y*q2.z + q1.z*q2.y)/q2norm;
    q1a->y = (-q1.w*q2.y + q1.x*q2.z + q1.y*q2.w - q1.z*q2.x)/q2norm;
    q1a->z = (-q1.w*q2.z - q1.x*q2.y + q1.y*q2.x + q1.z*q2.w)/q2norm;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_scalar_divide(double s, dual_quaternion q) {
    double qnorm = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    dual_quaternion r = {
      ( s*q.w) / qnorm,
      (-s*q.x) / qnorm,
      (-s*q.y) / qnorm,
      (-s*q.z) / qnorm
    };
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
    /* Note that the following is just my chosen definition of the power. */
    /* Other definitions may disagree due to non-commutativity. */
    if(! dual_quaternion_nonzero(q)) { /* log(q)=-inf */
      if(! dual_quaternion_nonzero(p)) {
        dual_quaternion r = {1.0, 0.0, 0.0, 0.0}; /* consistent with python */
        return r;
      } else {
        dual_quaternion r = {0.0, 0.0, 0.0, 0.0}; /* consistent with python */
        return r;
      }
    }
    return dual_quaternion_exp(dual_quaternion_multiply(dual_quaternion_log(q), p));
  }
  static NPY_INLINE void dual_quaternion_inplace_power(dual_quaternion* q1, dual_quaternion q2) {
    /* Not overly useful as an in-place operator, but here for completeness. */
    dual_quaternion q3 = dual_quaternion_power(*q1,q2);
    *q1 = q3;
    return;
  }
  dual_quaternion dual_quaternion_scalar_power(double s, dual_quaternion q);
  static NPY_INLINE void dual_quaternion_inplace_scalar_power(double s, dual_quaternion* q) {
    /* Not overly useful as an in-place operator, but here for completeness. */
    dual_quaternion q2 = dual_quaternion_scalar_power(s, *q);
    *q = q2;
    return;
  }
  static NPY_INLINE dual_quaternion dual_quaternion_power_scalar(dual_quaternion q, double s) {
    /* Unlike the dual_quaternion^dual_quaternion power, this is unambiguous. */
    if(! dual_quaternion_nonzero(q)) { /* log(q)=-inf */
      if(s==0) {
        dual_quaternion r = {1.0, 0.0, 0.0, 0.0}; /* consistent with python */
        return r;
      } else {
        dual_quaternion r = {0.0, 0.0, 0.0, 0.0}; /* consistent with python */
        return r;
      }
    }
    return dual_quaternion_exp(dual_quaternion_multiply_scalar(dual_quaternion_log(q), s));
  }
  static NPY_INLINE void dual_quaternion_inplace_power_scalar(dual_quaternion* q, double s) {
    /* Not overly useful as an in-place operator, but here for completeness. */
    dual_quaternion q2 = dual_quaternion_power_scalar(*q, s);
    *q = q2;
    return;
  }

  // Associated functions
  static NPY_INLINE double dual_quaternion_rotor_intrinsic_distance(dual_quaternion q1, dual_quaternion q2) {
    return 2*dual_quaternion_absolute(dual_quaternion_log(dual_quaternion_divide(q1,q2)));
  }
  static NPY_INLINE double dual_quaternion_rotor_chordal_distance(dual_quaternion q1, dual_quaternion q2) {
    return dual_quaternion_absolute(dual_quaternion_subtract(q1,q2));
  }
  static NPY_INLINE double dual_quaternion_rotation_intrinsic_distance(dual_quaternion q1, dual_quaternion q2) {
    if(dual_quaternion_rotor_chordal_distance(q1,q2)<=1.414213562373096) {
      return 2*dual_quaternion_absolute(dual_quaternion_log(dual_quaternion_divide(q1,q2)));
    } else {
      return 2*dual_quaternion_absolute(dual_quaternion_log(dual_quaternion_divide(q1,dual_quaternion_negative(q2))));
    }
  }
  static NPY_INLINE double dual_quaternion_rotation_chordal_distance(dual_quaternion q1, dual_quaternion q2) {
    if(dual_quaternion_rotor_chordal_distance(q1,q2)<=1.414213562373096) {
      return dual_quaternion_absolute(dual_quaternion_subtract(q1,q2));
    } else {
      return dual_quaternion_absolute(dual_quaternion_add(q1,q2));
    }
  }
  static NPY_INLINE dual_quaternion slerp(dual_quaternion q1, dual_quaternion q2, double tau) {
    if(dual_quaternion_rotor_chordal_distance(q1,q2)<=1.414213562373096) {
      return dual_quaternion_multiply( dual_quaternion_power_scalar(dual_quaternion_divide(q2,q1), tau), q1);
    } else {
      return dual_quaternion_multiply( dual_quaternion_power_scalar(dual_quaternion_divide(dual_quaternion_negative(q2),q1), tau), q1);
    }
  }
  static NPY_INLINE dual_quaternion squad_evaluate(double tau_i, dual_quaternion q_i, dual_quaternion a_i, dual_quaternion b_ip1, dual_quaternion q_ip1) {
    return slerp(slerp(q_i, q_ip1, tau_i),
                 slerp(a_i, b_ip1, tau_i),
                 2*tau_i*(1-tau_i));
  }


#ifdef __cplusplus
}
#endif

#endif // __DUAL_QUATERNION_H__
