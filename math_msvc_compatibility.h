// Copyright (c) 2016, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#ifndef __MATH_MSVC_COMPATIBILITY_H__
#define __MATH_MSVC_COMPATIBILITY_H__

#ifdef __cplusplus
extern "C" {
#endif

#define _USE_MATH_DEFINES
#include <float.h>
#include <math.h>

#ifdef isnan
#undef isnan
#endif

#ifdef isinf
#undef isinf
#endif

#ifdef isfinite
#undef isfinite
#endif

#ifdef copysign
#undef copysign
#endif

static __inline int isnan(double x) {
  return _isnan(x);
}

static __inline int isinf(double x) {
  return !_finite(x);
}

static __inline int isfinite(double x) {
  return _finite(x);
}

static __inline double copysign(double x, double y) {
  return _copysign(x, y);
}

#ifdef __cplusplus
}
#endif

#endif // __MATH_MSVC_COMPATIBILITY_H__
