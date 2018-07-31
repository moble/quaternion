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
dual_quaternion_sqrt(dual_quaternion q)
{
  PyErr_SetNone(PyExc_NotImplementedError);
  dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
  return r;
}

dual_quaternion
dual_quaternion_log(dual_quaternion q)
{
  PyErr_SetNone(PyExc_NotImplementedError);
  dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
  return r;
}

double
_dual_quaternion_scalar_log(double s) { return log(s); }

dual_quaternion
dual_quaternion_scalar_power(double s, dual_quaternion q)
{
  PyErr_SetNone(PyExc_NotImplementedError);
  dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
  return r;
}

dual_quaternion
dual_quaternion_exp(dual_quaternion q)
{
  PyErr_SetNone(PyExc_NotImplementedError);
  dual_quaternion r = {0, 0, 0, 0, 0, 0, 0, 0};
  return r;
}

#ifdef __cplusplus
}
#endif
