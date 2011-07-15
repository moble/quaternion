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

int quaternion_isnonzero(quaternion q);
int quaternion_isnan(quaternion q);
int quaternion_isinf(quaternion q);
int quaternion_isfinite(quaternion q);
double quaternion_absolute(quaternion q);
quaternion quaternion_add(quaternion q1, quaternion q2);
quaternion quaternion_subtract(quaternion q1, quaternion q2);
quaternion quaternion_multiply(quaternion q1, quaternion q2);
quaternion quaternion_divide(quaternion q1, quaternion q2);
quaternion quaternion_multiply_scalar(quaternion q, double s);
quaternion quaternion_divide_scalar(quaternion q, double s);
quaternion quaternion_log(quaternion q);
quaternion quaternion_exp(quaternion q);
quaternion quaternion_power(quaternion q, quaternion p);
quaternion quaternion_power_scalar(quaternion q, double p);
quaternion quaternion_negative(quaternion q);
quaternion quaternion_conjugate(quaternion q);
quaternion quaternion_copysign(quaternion q1, quaternion q2);
int quaternion_equal(quaternion q1, quaternion q2);
int quaternion_not_equal(quaternion q1, quaternion q2);
int quaternion_less(quaternion q1, quaternion q2);
int quaternion_less_equal(quaternion q1, quaternion q2);

#ifdef __cplusplus
}
#endif

#endif
