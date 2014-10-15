/*
 * Quaternion type for NumPy
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

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"

#include "quaternion.h"

typedef struct {
  PyObject_HEAD
  quaternion obval;
} PyQuaternionScalarObject;

PyMemberDef PyQuaternionArrType_members[] = {
  {"real", T_DOUBLE, offsetof(PyQuaternionScalarObject, obval.w), READONLY,
   "The real component of the quaternion"},
  {"w", T_DOUBLE, offsetof(PyQuaternionScalarObject, obval.w), READONLY,
   "The real component of the quaternion"},
  {"x", T_DOUBLE, offsetof(PyQuaternionScalarObject, obval.x), READONLY,
   "The first imaginary component of the quaternion"},
  {"y", T_DOUBLE, offsetof(PyQuaternionScalarObject, obval.y), READONLY,
   "The second imaginary component of the quaternion"},
  {"z", T_DOUBLE, offsetof(PyQuaternionScalarObject, obval.z), READONLY,
   "The third imaginary component of the quaternion"},
  {NULL}
};

static PyObject *
PyQuaternionArrType_get_components(PyObject *self, void *closure)
{
  quaternion *q = &((PyQuaternionScalarObject *)self)->obval;
  PyObject *tuple = PyTuple_New(4);
  PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q->w));
  PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q->x));
  PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(q->y));
  PyTuple_SET_ITEM(tuple, 3, PyFloat_FromDouble(q->z));
  return tuple;
}

static PyObject *
PyQuaternionArrType_get_imag(PyObject *self, void *closure)
{
  quaternion *q = &((PyQuaternionScalarObject *)self)->obval;
  PyObject *tuple = PyTuple_New(3);
  PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q->x));
  PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q->y));
  PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(q->z));
  return tuple;
}

PyGetSetDef PyQuaternionArrType_getset[] = {
  {"components", PyQuaternionArrType_get_components, NULL,
   "The components of the quaternion as a (w,x,y,z) tuple", NULL},
  {"imag", PyQuaternionArrType_get_imag, NULL,
   "The imaginary part of the quaternion as an (x,y,z) tuple", NULL},
  {NULL}
};

PyTypeObject PyQuaternionArrType_Type = {
  #if defined(NPY_PY3K)
  PyVarObject_HEAD_INIT(NULL, 0)
  #else
  PyObject_HEAD_INIT(NULL)
  0,                                          /* ob_size */
  #endif
  "quaternion.quaternion",                    /* tp_name*/
  sizeof(PyQuaternionScalarObject),           /* tp_basicsize*/
  0,                                          /* tp_itemsize */
  0,                                          /* tp_dealloc */
  0,                                          /* tp_print */
  0,                                          /* tp_getattr */
  0,                                          /* tp_setattr */
  #if defined(NPY_PY3K)
  0,                                          /* tp_reserved */
  #else
  0,                                          /* tp_compare */
  #endif
  0,                                          /* tp_repr */
  0,                                          /* tp_as_number */
  0,                                          /* tp_as_sequence */
  0,                                          /* tp_as_mapping */
  0,                                          /* tp_hash */
  0,                                          /* tp_call */
  0,                                          /* tp_str */
  0,                                          /* tp_getattro */
  0,                                          /* tp_setattro */
  0,                                          /* tp_as_buffer */
  0,                                          /* tp_flags */
  0,                                          /* tp_doc */
  0,                                          /* tp_traverse */
  0,                                          /* tp_clear */
  0,                                          /* tp_richcompare */
  0,                                          /* tp_weaklistoffset */
  0,                                          /* tp_iter */
  0,                                          /* tp_iternext */
  0,                                          /* tp_methods */
  PyQuaternionArrType_members,                /* tp_members */
  PyQuaternionArrType_getset,                 /* tp_getset */
  0,                                          /* tp_base */
  0,                                          /* tp_dict */
  0,                                          /* tp_descr_get */
  0,                                          /* tp_descr_set */
  0,                                          /* tp_dictoffset */
  0,                                          /* tp_init */
  0,                                          /* tp_alloc */
  0,                                          /* tp_new */
  0,                                          /* tp_free */
  0,                                          /* tp_is_gc */
  0,                                          /* tp_bases */
  0,                                          /* tp_mro */
  0,                                          /* tp_cache */
  0,                                          /* tp_subclasses */
  0,                                          /* tp_weaklist */
  0,                                          /* tp_del */
  #if PY_VERSION_HEX >= 0x02060000
  0,                                          /* tp_version_tag */
  #endif
};

static PyArray_ArrFuncs _PyQuaternion_ArrFuncs;
PyArray_Descr *quaternion_descr;

static PyObject *
QUATERNION_getitem(char *ip, PyArrayObject *ap)
{
  quaternion q;
  PyObject *tuple;

  if ((ap == NULL) || PyArray_ISBEHAVED_RO(ap)) {
    q = *((quaternion *)ip);
  }
  else {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswap(&q.w, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.x, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.y, ip+16, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.z, ip+24, !PyArray_ISNOTSWAPPED(ap), NULL);
    Py_DECREF(descr);
  }

  tuple = PyTuple_New(4);
  PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble(q.w));
  PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(q.x));
  PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(q.y));
  PyTuple_SET_ITEM(tuple, 3, PyFloat_FromDouble(q.z));

  return tuple;
}

static int QUATERNION_setitem(PyObject *op, char *ov, PyArrayObject *ap)
{
  quaternion q;

  if (PyArray_IsScalar(op, Quaternion)) {
    q = ((PyQuaternionScalarObject *)op)->obval;
  }
  else {
    q.w = PyFloat_AsDouble(PyTuple_GetItem(op, 0));
    q.x = PyFloat_AsDouble(PyTuple_GetItem(op, 1));
    q.y = PyFloat_AsDouble(PyTuple_GetItem(op, 2));
    q.z = PyFloat_AsDouble(PyTuple_GetItem(op, 3));
  }
  if (PyErr_Occurred()) {
    if (PySequence_Check(op)) {
      PyErr_Clear();
      PyErr_SetString(PyExc_ValueError,
                      "setting an array element with a sequence.");
    }
    return -1;
  }
  if (ap == NULL || PyArray_ISBEHAVED(ap))
    *((quaternion *)ov)=q;
  else {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswap(ov, &q.w, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(ov+8, &q.x, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(ov+16, &q.y, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(ov+24, &q.z, !PyArray_ISNOTSWAPPED(ap), NULL);
    Py_DECREF(descr);
  }

  return 0;
}

static void
QUATERNION_copyswap(quaternion *dst, quaternion *src,
                    int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(dst, sizeof(double), src, sizeof(double), 4, swap, NULL);
  Py_DECREF(descr);
}

static void
QUATERNION_copyswapn(quaternion *dst, npy_intp dstride,
                     quaternion *src, npy_intp sstride,
                     npy_intp n, int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(&dst->w, dstride, &src->w, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->x, dstride, &src->x, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->y, dstride, &src->y, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->z, dstride, &src->z, sstride, n, swap, NULL);
  Py_DECREF(descr);
}

static int
QUATERNION_compare (quaternion *pa, quaternion *pb, PyArrayObject *NPY_UNUSED(ap))
{
  quaternion a = *pa, b = *pb;
  npy_bool anan, bnan;
  int ret;

  anan = quaternion_isnan(a);
  bnan = quaternion_isnan(b);

  if (anan) {
    ret = bnan ? 0 : -1;
  } else if (bnan) {
    ret = 1;
  } else if(quaternion_less(a, b)) {
    ret = -1;
  } else if(quaternion_less(b, a)) {
    ret = 1;
  } else {
    ret = 0;
  }

  return ret;
}

static int
QUATERNION_argmax(quaternion *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
  npy_intp i;
  quaternion mp = *ip;

  *max_ind = 0;

  if (quaternion_isnan(mp)) {
    /* nan encountered; it's maximal */
    return 0;
  }

  for (i = 1; i < n; i++) {
    ip++;
    /*
     * Propagate nans, similarly as max() and min()
     */
    if (!(quaternion_less_equal(*ip, mp))) {  /* negated, for correct nan handling */
      mp = *ip;
      *max_ind = i;
      if (quaternion_isnan(mp)) {
        /* nan encountered, it's maximal */
        break;
      }
    }
  }
  return 0;
}

static npy_bool
QUATERNION_nonzero (char *ip, PyArrayObject *ap)
{
  quaternion q;
  if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
    q = *(quaternion *)ip;
  }
  else {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswap(&q.w, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.x, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.y, ip+16, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.z, ip+24, !PyArray_ISNOTSWAPPED(ap), NULL);
    Py_DECREF(descr);
  }
  return (npy_bool) !quaternion_equal(q, (quaternion) {0,0,0,0});
}

static void
QUATERNION_fillwithscalar(quaternion *buffer, npy_intp length, quaternion *value, void *NPY_UNUSED(ignored))
{
  npy_intp i;
  quaternion val = *value;

  for (i = 0; i < length; ++i) {
    buffer[i] = val;
  }
}

#define MAKE_T_TO_QUATERNION(TYPE, type)                                \
  static void                                                           \
  TYPE ## _to_quaternion(type *ip, quaternion *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      op->w = (double)(*ip++);                                          \
      op->x = 0;                                                        \
      op->y = 0;                                                        \
      op->z = 0;                                                        \
    }                                                                   \
  }

MAKE_T_TO_QUATERNION(FLOAT, npy_float);
MAKE_T_TO_QUATERNION(DOUBLE, npy_double);
MAKE_T_TO_QUATERNION(LONGDOUBLE, npy_longdouble);
MAKE_T_TO_QUATERNION(BOOL, npy_bool);
MAKE_T_TO_QUATERNION(BYTE, npy_byte);
MAKE_T_TO_QUATERNION(UBYTE, npy_ubyte);
MAKE_T_TO_QUATERNION(SHORT, npy_short);
MAKE_T_TO_QUATERNION(USHORT, npy_ushort);
MAKE_T_TO_QUATERNION(INT, npy_int);
MAKE_T_TO_QUATERNION(UINT, npy_uint);
MAKE_T_TO_QUATERNION(LONG, npy_long);
MAKE_T_TO_QUATERNION(ULONG, npy_ulong);
MAKE_T_TO_QUATERNION(LONGLONG, npy_longlong);
MAKE_T_TO_QUATERNION(ULONGLONG, npy_ulonglong);

#define MAKE_CT_TO_QUATERNION(TYPE, type)                               \
  static void                                                           \
  TYPE ## _to_quaternion(type *ip, quaternion *op, npy_intp n,          \
                           PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
   while (n--) {                                                        \
                op->w = (double)(*ip++);                                \
                op->x = (double)(*ip++);                                \
                op->y = 0;                                              \
                op->z = 0;                                              \
                }                                                       \
   }

MAKE_CT_TO_QUATERNION(CFLOAT, npy_uint32);
MAKE_CT_TO_QUATERNION(CDOUBLE, npy_uint64);
MAKE_CT_TO_QUATERNION(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
  PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
  PyArray_RegisterCastFunc(descr, destType, castfunc);
  PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
  Py_DECREF(descr);
}

static PyObject *
quaternion_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  quaternion q;

  if (!PyArg_ParseTuple(args, "dddd", &q.w, &q.x, &q.y, &q.z))
    return NULL;

  return PyArray_Scalar(&q, quaternion_descr, NULL);
}

static PyObject *
gentype_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
  PyObject *arr, *ret;

  arr = PyArray_FromScalar(self, NULL);
  if (arr == NULL) {
    return NULL;
  }
  ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
  Py_DECREF(arr);
  return ret;
}

static long
quaternion_arrtype_hash(PyObject *o)
{
  quaternion q = ((PyQuaternionScalarObject *)o)->obval;
  long value = 0x456789;
  value = (10000004 * value) ^ _Py_HashDouble(q.w);
  value = (10000004 * value) ^ _Py_HashDouble(q.x);
  value = (10000004 * value) ^ _Py_HashDouble(q.y);
  value = (10000004 * value) ^ _Py_HashDouble(q.z);
  if (value == -1)
    value = -2;
  return value;
}

static PyObject *
quaternion_arrtype_repr(PyObject *o)
{
  char str[128];
  quaternion q = ((PyQuaternionScalarObject *)o)->obval;
  sprintf(str, "quaternion(%g, %g, %g, %g)", q.w, q.x, q.y, q.z);
  return PyString_FromString(str);
}

static PyObject *
quaternion_arrtype_str(PyObject *o)
{
  char str[128];
  quaternion q = ((PyQuaternionScalarObject *)o)->obval;
  sprintf(str, "quaternion(%g, %g, %g, %g)", q.w, q.x, q.y, q.z);
  return PyString_FromString(str);
}

static PyMethodDef QuaternionMethods[] = {
  {NULL, NULL, 0, NULL}
};

#define UNARY_UFUNC(name, ret_type)                             \
  static void                                                   \
  quaternion_##name##_ufunc(char** args, npy_intp* dimensions,  \
                            npy_intp* steps, void* data) {      \
    char *ip1 = args[0], *op1 = args[1];                        \
    npy_intp is1 = steps[0], os1 = steps[1];                    \
    npy_intp n = dimensions[0];                                 \
    npy_intp i;                                                 \
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){             \
      const quaternion in1 = *(quaternion *)ip1;                \
      *((ret_type *)op1) = quaternion_##name(in1);};}

UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(log, quaternion)
UNARY_UFUNC(exp, quaternion)
UNARY_UFUNC(negative, quaternion)
UNARY_UFUNC(conjugate, quaternion)

#define BINARY_GEN_UFUNC(name, func_name, arg_type, ret_type)           \
  static void                                                           \
  quaternion_##func_name##_ufunc(char** args, npy_intp* dimensions,     \
                                   npy_intp* steps, void* data) {       \
                                                                 char *ip1 = args[0], *ip2 = args[1], *op1 = args[2]; \
                                                                 npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2]; \
                                                                 npy_intp n = dimensions[0]; \
                                                                 npy_intp i; \
                                                                 for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1){ \
                                                                                                                            const quaternion in1 = *(quaternion *)ip1; \
                                                                                                                            const arg_type in2 = *(arg_type *)ip2; \
                                                                                                                            *((ret_type *)op1) = quaternion_##func_name(in1, in2);};};

#define BINARY_UFUNC(name, ret_type)                    \
  BINARY_GEN_UFUNC(name, name, quaternion, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)                     \
  BINARY_GEN_UFUNC(name, name##_scalar, npy_double, ret_type)

BINARY_UFUNC(add, quaternion)
BINARY_UFUNC(subtract, quaternion)
BINARY_UFUNC(multiply, quaternion)
BINARY_UFUNC(divide, quaternion)
BINARY_UFUNC(power, quaternion)
BINARY_UFUNC(copysign, quaternion)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
     BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)

BINARY_SCALAR_UFUNC(multiply, quaternion)
     BINARY_SCALAR_UFUNC(divide, quaternion)
BINARY_SCALAR_UFUNC(power, quaternion)

PyMODINIT_FUNC initnumpy_quaternion(void)
{
  PyObject *m;
  int quaternionNum;
  PyObject* numpy = PyImport_ImportModule("numpy");
  PyObject* numpy_dict = PyModule_GetDict(numpy);
  int arg_types[3];

  m = Py_InitModule("numpy_quaternion", QuaternionMethods);
  if (m == NULL) {
    return;
  }

  /* Make sure NumPy is initialized */
  import_array();
  import_umath();

  /* Register the quaternion array scalar type */
  #if defined(NPY_PY3K)
  PyQuaternionArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  #else
  PyQuaternionArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
  #endif
  PyQuaternionArrType_Type.tp_new = quaternion_arrtype_new;
  PyQuaternionArrType_Type.tp_richcompare = gentype_richcompare;
  PyQuaternionArrType_Type.tp_hash = quaternion_arrtype_hash;
  PyQuaternionArrType_Type.tp_repr = quaternion_arrtype_repr;
  PyQuaternionArrType_Type.tp_str = quaternion_arrtype_str;
  PyQuaternionArrType_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&PyQuaternionArrType_Type) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "could not initialize PyQuaternionArrType_Type");
    return;
  }

  /* The array functions */
  PyArray_InitArrFuncs(&_PyQuaternion_ArrFuncs);
  _PyQuaternion_ArrFuncs.getitem = (PyArray_GetItemFunc*)QUATERNION_getitem;
  _PyQuaternion_ArrFuncs.setitem = (PyArray_SetItemFunc*)QUATERNION_setitem;
  _PyQuaternion_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)QUATERNION_copyswap;
  _PyQuaternion_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)QUATERNION_copyswapn;
  _PyQuaternion_ArrFuncs.compare = (PyArray_CompareFunc*)QUATERNION_compare;
  _PyQuaternion_ArrFuncs.argmax = (PyArray_ArgFunc*)QUATERNION_argmax;
  _PyQuaternion_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)QUATERNION_nonzero;
  _PyQuaternion_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)QUATERNION_fillwithscalar;

  /* The quaternion array descr */
  quaternion_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  quaternion_descr->typeobj = &PyQuaternionArrType_Type;
  quaternion_descr->kind = 'q';
  quaternion_descr->type = 'j';
  quaternion_descr->byteorder = '=';
  quaternion_descr->type_num = 0; /* assigned at registration */
  quaternion_descr->elsize = 8*4;
  quaternion_descr->alignment = 8;
  quaternion_descr->subarray = NULL;
  quaternion_descr->fields = NULL;
  quaternion_descr->names = NULL;
  quaternion_descr->f = &_PyQuaternion_ArrFuncs;

  Py_INCREF(&PyQuaternionArrType_Type);
  quaternionNum = PyArray_RegisterDataType(quaternion_descr);

  if (quaternionNum < 0)
    return;

  register_cast_function(NPY_BOOL, quaternionNum, (PyArray_VectorUnaryFunc*)BOOL_to_quaternion);
  register_cast_function(NPY_BYTE, quaternionNum, (PyArray_VectorUnaryFunc*)BYTE_to_quaternion);
  register_cast_function(NPY_UBYTE, quaternionNum, (PyArray_VectorUnaryFunc*)UBYTE_to_quaternion);
  register_cast_function(NPY_SHORT, quaternionNum, (PyArray_VectorUnaryFunc*)SHORT_to_quaternion);
  register_cast_function(NPY_USHORT, quaternionNum, (PyArray_VectorUnaryFunc*)USHORT_to_quaternion);
  register_cast_function(NPY_INT, quaternionNum, (PyArray_VectorUnaryFunc*)INT_to_quaternion);
  register_cast_function(NPY_UINT, quaternionNum, (PyArray_VectorUnaryFunc*)UINT_to_quaternion);
  register_cast_function(NPY_LONG, quaternionNum, (PyArray_VectorUnaryFunc*)LONG_to_quaternion);
  register_cast_function(NPY_ULONG, quaternionNum, (PyArray_VectorUnaryFunc*)ULONG_to_quaternion);
  register_cast_function(NPY_LONGLONG, quaternionNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_quaternion);
  register_cast_function(NPY_ULONGLONG, quaternionNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_quaternion);
  register_cast_function(NPY_FLOAT, quaternionNum, (PyArray_VectorUnaryFunc*)FLOAT_to_quaternion);
  register_cast_function(NPY_DOUBLE, quaternionNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_quaternion);
  register_cast_function(NPY_LONGDOUBLE, quaternionNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_quaternion);
  register_cast_function(NPY_CFLOAT, quaternionNum, (PyArray_VectorUnaryFunc*)CFLOAT_to_quaternion);
  register_cast_function(NPY_CDOUBLE, quaternionNum, (PyArray_VectorUnaryFunc*)CDOUBLE_to_quaternion);
  register_cast_function(NPY_CLONGDOUBLE, quaternionNum, (PyArray_VectorUnaryFunc*)CLONGDOUBLE_to_quaternion);

  #define REGISTER_UFUNC(name)                                          \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                quaternion_descr->type_num, quaternion_##name##_ufunc, arg_types, NULL)

  #define REGISTER_SCALAR_UFUNC(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                quaternion_descr->type_num, quaternion_##name##_scalar_ufunc, arg_types, NULL)

  /* quat -> bool */
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = NPY_BOOL;
  REGISTER_UFUNC(isnan);
  REGISTER_UFUNC(isinf);
  REGISTER_UFUNC(isfinite);

  /* quat -> double */
  arg_types[1] = NPY_DOUBLE;
  REGISTER_UFUNC(absolute);

  /* quat -> quat */
  arg_types[1] = quaternion_descr->type_num;
  REGISTER_UFUNC(log);
  REGISTER_UFUNC(exp);
  REGISTER_UFUNC(negative);
  REGISTER_UFUNC(conjugate);

  /* quat, quat -> bool */
  arg_types[2] = NPY_BOOL;
  REGISTER_UFUNC(equal);
  REGISTER_UFUNC(not_equal);
  REGISTER_UFUNC(less);
  REGISTER_UFUNC(less_equal);

  /* quat, double -> quat */
  arg_types[1] = NPY_DOUBLE;
  arg_types[2] = quaternion_descr->type_num;
  REGISTER_SCALAR_UFUNC(multiply);
  REGISTER_SCALAR_UFUNC(divide);
  REGISTER_SCALAR_UFUNC(power);

  /* quat, quat -> quat */
  arg_types[1] = quaternion_descr->type_num;
  REGISTER_UFUNC(add);
  REGISTER_UFUNC(subtract);
  REGISTER_UFUNC(multiply);
  REGISTER_UFUNC(divide);
  REGISTER_UFUNC(power);
  REGISTER_UFUNC(copysign);

  PyModule_AddObject(m, "quaternion", (PyObject *)&PyQuaternionArrType_Type);
}
