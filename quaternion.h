// Copyright (c) 2014, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#ifndef __QUATERNION_H__
#define __QUATERNION_H__

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {
    double w;
    double x;
    double y;
    double z;
  } quaternion;

  // Constructor-ish
  quaternion quaternion_create_from_spherical_coords(double vartheta, double varphi);
  quaternion quaternion_create_from_euler_angles(double alpha, double beta, double gamma);

  // Unary bool returners
  int quaternion_nonzero(quaternion q);
  int quaternion_isnan(quaternion q);
  int quaternion_isinf(quaternion q);
  int quaternion_isfinite(quaternion q);
  // Binary bool returners
  int quaternion_equal(quaternion q1, quaternion q2);
  int quaternion_not_equal(quaternion q1, quaternion q2);
  int quaternion_less(quaternion q1, quaternion q2);
  int quaternion_greater(quaternion q1, quaternion q2);
  int quaternion_less_equal(quaternion q1, quaternion q2);
  int quaternion_greater_equal(quaternion q1, quaternion q2);
  // Unary float returners
  double quaternion_absolute(quaternion q);
  double quaternion_norm(quaternion q);
  double quaternion_angle(quaternion q);
  // Unary quaternion returners
  quaternion quaternion_negative(quaternion q);
  quaternion quaternion_conjugate(quaternion q);
  quaternion quaternion_inverse(quaternion q);
  quaternion quaternion_sqrt(quaternion q);
  quaternion quaternion_log(quaternion q);
  quaternion quaternion_exp(quaternion q);
  quaternion quaternion_normalized(quaternion q);
  quaternion quaternion_x_parity_conjugate(quaternion q);
  quaternion quaternion_y_parity_conjugate(quaternion q);
  quaternion quaternion_z_parity_conjugate(quaternion q);
  quaternion quaternion_parity_conjugate(quaternion q);
  // Quaternion-quaternion binary quaternion returners
  quaternion quaternion_add(quaternion q1, quaternion q2);
  quaternion quaternion_subtract(quaternion q1, quaternion q2);
  quaternion quaternion_copysign(quaternion q1, quaternion q2);
  // Quaternion-quaternion/quaternion-scalar binary quaternion returners
  quaternion quaternion_multiply(quaternion q1, quaternion q2);
  quaternion quaternion_multiply_scalar(quaternion q, double s);
  quaternion quaternion_divide(quaternion q1, quaternion q2);
  quaternion quaternion_divide_scalar(quaternion q, double s);
  quaternion quaternion_power(quaternion q, quaternion p);
  quaternion quaternion_power_scalar(quaternion q, double p);

  // In-place operations
  void quaternion_inplace_add(quaternion* q1, quaternion q2);
  void quaternion_inplace_subtract(quaternion* q1, quaternion q2);
  void quaternion_inplace_multiply(quaternion* q1, quaternion q2);
  void quaternion_inplace_multiply_scalar(quaternion* q, double s);
  void quaternion_inplace_divide(quaternion* q1, quaternion q2);
  void quaternion_inplace_divide_scalar(quaternion* q, double s);
  void quaternion_inplace_power(quaternion* q1, quaternion q2);
  void quaternion_inplace_power_scalar(quaternion* q, double s);

  // Associated functions
  double rotor_intrinsic_distance(quaternion q1, quaternion q2);
  double rotor_chordal_distance(quaternion q1, quaternion q2);
  double rotation_intrinsic_distance(quaternion q1, quaternion q2);
  double rotation_chordal_distance(quaternion q1, quaternion q2);
  quaternion slerp(quaternion q1, quaternion q2, double tau);
  quaternion squad_evaluate(double tau_i, quaternion q_i, quaternion a_i, quaternion b_ip1, quaternion q_ip1);

#ifdef __cplusplus
}
#endif

#endif // __QUATERNION_H__
