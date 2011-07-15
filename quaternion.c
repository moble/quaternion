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

int
quaternion_isnonzero(quaternion q)
{
    return q.w != 0 && q.x != 0 && q.y != 0 && q.z != 0;
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
   return sqrt(pow(q.w, 2) + pow(q.x, 2) + pow(q.y, 2) + pow(q.z, 2));
}

quaternion
quaternion_copysign(quaternion x, quaternion y)
{
    return (quaternion) {
        copysign(x.w, y.w),
        copysign(x.x, y.x),
        copysign(x.y, y.y),
        copysign(x.z, y.z)
    };
}

int
quaternion_eq(quaternion q1, quaternion q2)
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
quaternion_ne(quaternion q1, quaternion q2)
{
    return !quaternion_eq(q1, q2);
}

int
quaternion_lt(quaternion q1, quaternion q2)
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
quaternion_gt(quaternion q1, quaternion q2)
{
    return quaternion_lt(q2, q1);
}

int
quaternion_le(quaternion q1, quaternion q2)
{
   return
        (!quaternion_isnan(q1) &&
        !quaternion_isnan(q2)) && (
            q1.w != q2.w ? q1.w < q2.w :
            q1.x != q2.x ? q1.x < q2.x :
            q1.y != q2.y ? q1.y < q2.y :
            q1.z != q2.z ? q1.z < q2.z : 1);
}

int
quaternion_ge(quaternion q1, quaternion q2)
{
    return quaternion_le(q2, q1);
}
