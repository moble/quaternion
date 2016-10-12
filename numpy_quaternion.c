// Copyright (c) 2016, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"

#include "quaternion.h"


// The following definitions, along with `#define NPY_PY3K 1`, can
// also be found in the header <numpy/npy_3kcompat.h>.
#if PY_MAJOR_VERSION >= 3
#define PyUString_FromString PyUnicode_FromString
static NPY_INLINE int PyInt_Check(PyObject *op) {
    int overflow = 0;
    if (!PyLong_Check(op)) {
        return 0;
    }
    PyLong_AsLongAndOverflow(op, &overflow);
    return (overflow == 0);
}
#define PyInt_AsLong PyLong_AsLong
#else
#define PyUString_FromString PyString_FromString
#endif


// The basic python object holding a quaternion
typedef struct {
  PyObject_HEAD
  quaternion obval;
} PyQuaternion;

static PyTypeObject PyQuaternion_Type;

// This is the crucial feature that will make a quaternion into a
// built-in numpy data type.  We will describe its features below.
PyArray_Descr* quaternion_descr;


static NPY_INLINE int
PyQuaternion_Check(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&PyQuaternion_Type);
}

static PyObject*
PyQuaternion_FromQuaternion(quaternion q) {
  PyQuaternion* p = (PyQuaternion*)PyQuaternion_Type.tp_alloc(&PyQuaternion_Type,0);
  if (p) { p->obval = q; }
  return (PyObject*)p;
}

// TODO: Add list/tuple conversions
#define PyQuaternion_AsQuaternion(q, o)                                 \
  /* fprintf (stderr, "file %s, line %d., PyQuaternion_AsQuaternion\n", __FILE__, __LINE__); */ \
  if(PyQuaternion_Check(o)) {                                           \
    q = ((PyQuaternion*)o)->obval;                                      \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not a quaternion.");               \
    return NULL;                                                        \
  }

#define PyQuaternion_AsQuaternionPointer(q, o)                          \
  /* fprintf (stderr, "file %s, line %d, PyQuaternion_AsQuaternionPointer.\n", __FILE__, __LINE__); */ \
  if(PyQuaternion_Check(o)) {                                           \
    q = &((PyQuaternion*)o)->obval;                                     \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not a quaternion.");               \
    return NULL;                                                        \
  }

static PyObject *
pyquaternion_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  PyQuaternion* self;
  self = (PyQuaternion *)type->tp_alloc(type, 0);
  return (PyObject *)self;
}

static int
pyquaternion_init(PyObject *self, PyObject *args, PyObject *kwds)
{
  // "A good rule of thumb is that for immutable types, all
  // initialization should take place in `tp_new`, while for mutable
  // types, most initialization should be deferred to `tp_init`."
  // ---Python 2.7.8 docs
  quaternion* q;
  q = &(((PyQuaternion*)self)->obval);
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError,
                    "quaternion constructor takes no keyword arguments");
    return -1;
  }

  if (!PyArg_ParseTuple(args, "dddd", &q->w, &q->x, &q->y, &q->z)) {
    PyErr_SetString(PyExc_TypeError,
                    "quaternion constructor takes four double (float) arguments");
    return -1;
  }

  return 0;
}

#define UNARY_BOOL_RETURNER(name)                                       \
  static PyObject*                                                      \
  pyquaternion_##name(PyObject* a, PyObject* b) {                       \
    quaternion q = {0};                                                 \
    PyQuaternion_AsQuaternion(q, a);                                    \
    return PyBool_FromLong(quaternion_##name(q));                       \
  }
UNARY_BOOL_RETURNER(nonzero)
UNARY_BOOL_RETURNER(isnan)
UNARY_BOOL_RETURNER(isinf)
UNARY_BOOL_RETURNER(isfinite)

#define BINARY_BOOL_RETURNER(name)                                      \
  static PyObject*                                                      \
  pyquaternion_##name(PyObject* a, PyObject* b) {                       \
    quaternion p = {0};                                                 \
    quaternion q = {0};                                                 \
    PyQuaternion_AsQuaternion(p, a);                                    \
    PyQuaternion_AsQuaternion(q, b);                                    \
    return PyBool_FromLong(quaternion_##name(p,q));                     \
  }
BINARY_BOOL_RETURNER(equal)
BINARY_BOOL_RETURNER(not_equal)
BINARY_BOOL_RETURNER(less)
BINARY_BOOL_RETURNER(greater)
BINARY_BOOL_RETURNER(less_equal)
BINARY_BOOL_RETURNER(greater_equal)

#define UNARY_FLOAT_RETURNER(name)                                      \
  static PyObject*                                                      \
  pyquaternion_##name(PyObject* a, PyObject* b) {                       \
    quaternion q = {0};                                                 \
    PyQuaternion_AsQuaternion(q, a);                                    \
    return PyFloat_FromDouble(quaternion_##name(q));                    \
  }
UNARY_FLOAT_RETURNER(absolute)
UNARY_FLOAT_RETURNER(norm)
UNARY_FLOAT_RETURNER(angle)

#define UNARY_QUATERNION_RETURNER(name)                                 \
  static PyObject*                                                      \
  pyquaternion_##name(PyObject* a, PyObject* b) {                       \
    quaternion q = {0};                                                 \
    PyQuaternion_AsQuaternion(q, a);                                    \
    return PyQuaternion_FromQuaternion(quaternion_##name(q));           \
  }
UNARY_QUATERNION_RETURNER(negative)
UNARY_QUATERNION_RETURNER(conjugate)
UNARY_QUATERNION_RETURNER(inverse)
UNARY_QUATERNION_RETURNER(sqrt)
UNARY_QUATERNION_RETURNER(log)
UNARY_QUATERNION_RETURNER(exp)
UNARY_QUATERNION_RETURNER(normalized)
UNARY_QUATERNION_RETURNER(x_parity_conjugate)
UNARY_QUATERNION_RETURNER(x_parity_symmetric_part)
UNARY_QUATERNION_RETURNER(x_parity_antisymmetric_part)
UNARY_QUATERNION_RETURNER(y_parity_conjugate)
UNARY_QUATERNION_RETURNER(y_parity_symmetric_part)
UNARY_QUATERNION_RETURNER(y_parity_antisymmetric_part)
UNARY_QUATERNION_RETURNER(z_parity_conjugate)
UNARY_QUATERNION_RETURNER(z_parity_symmetric_part)
UNARY_QUATERNION_RETURNER(z_parity_antisymmetric_part)
UNARY_QUATERNION_RETURNER(parity_conjugate)
UNARY_QUATERNION_RETURNER(parity_symmetric_part)
UNARY_QUATERNION_RETURNER(parity_antisymmetric_part)
static PyObject*
pyquaternion_positive(PyObject* self, PyObject* b) {
    Py_INCREF(self);
    return self;
}

#define QQ_BINARY_QUATERNION_RETURNER(name)                             \
  static PyObject*                                                      \
  pyquaternion_##name(PyObject* a, PyObject* b) {                       \
    quaternion p = {0};                                                 \
    quaternion q = {0};                                                 \
    if(PyArray_Check(b)) { fprintf (stderr, "\nfile %s, line %d, pyquaternion_%s(PyObject* a, PyObject* b).\n", __FILE__, __LINE__, #name); } /*return pyquaternion_##name##_array_operator(a, b); }*/ \
    PyQuaternion_AsQuaternion(p, a);                                    \
    PyQuaternion_AsQuaternion(q, b);                                    \
    return PyQuaternion_FromQuaternion(quaternion_##name(p,q));         \
  }
/* QQ_BINARY_QUATERNION_RETURNER(add) */
/* QQ_BINARY_QUATERNION_RETURNER(subtract) */
QQ_BINARY_QUATERNION_RETURNER(copysign)

#define QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(fake_name, name)       \
  static PyObject*                                                      \
  pyquaternion_##fake_name##_array_operator(PyObject* a, PyObject* b) { \
    /* fprintf (stderr, "\nfile %s, line %d, pyquaternion_%s_array_operator(PyObject* a, PyObject* b).\n", __FILE__, __LINE__, #fake_name); */ \
    PyArrayObject *in_array = (PyArrayObject*) b;                       \
    PyObject      *out_array;                                           \
    NpyIter *in_iter;                                                   \
    NpyIter *out_iter;                                                  \
    NpyIter_IterNextFunc *in_iternext;                                  \
    NpyIter_IterNextFunc *out_iternext;                                 \
    quaternion p = {0};                                                 \
    quaternion ** out_dataptr;                                          \
    PyQuaternion_AsQuaternion(p, a);                                    \
    out_array = PyArray_NewLikeArray(in_array, NPY_ANYORDER, quaternion_descr, 0); \
    if (out_array == NULL) return NULL;                                 \
    in_iter = NpyIter_New(in_array, NPY_ITER_READONLY, NPY_KEEPORDER,   \
                          NPY_NO_CASTING, NULL);                        \
    if (in_iter == NULL) goto fail;                                     \
    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE, \
                           NPY_KEEPORDER, NPY_NO_CASTING, NULL);        \
    if (out_iter == NULL) {                                             \
      NpyIter_Deallocate(in_iter);                                      \
      goto fail;                                                        \
    }                                                                   \
    in_iternext = NpyIter_GetIterNext(in_iter, NULL);                   \
    out_iternext = NpyIter_GetIterNext(out_iter, NULL);                 \
    if (in_iternext == NULL || out_iternext == NULL) {                  \
      NpyIter_Deallocate(in_iter);                                      \
      NpyIter_Deallocate(out_iter);                                     \
      goto fail;                                                        \
    }                                                                   \
    out_dataptr = (quaternion **) NpyIter_GetDataPtrArray(out_iter);    \
    if(PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*) b), quaternion_descr)) { \
      quaternion ** in_dataptr = (quaternion **) NpyIter_GetDataPtrArray(in_iter); \
      do {                                                              \
        **out_dataptr = quaternion_##name(p, **in_dataptr);             \
      } while(in_iternext(in_iter) && out_iternext(out_iter));          \
    } else if(PyArray_ISFLOAT((PyArrayObject*) b)) {                    \
      double ** in_dataptr = (double **) NpyIter_GetDataPtrArray(in_iter); \
      do {                                                              \
        **out_dataptr = quaternion_##name##_scalar(p, **in_dataptr);    \
      } while(in_iternext(in_iter) && out_iternext(out_iter));          \
    } else if(PyArray_ISINTEGER((PyArrayObject*) b)) {                  \
      int ** in_dataptr = (int **) NpyIter_GetDataPtrArray(in_iter);    \
      do {                                                              \
        **out_dataptr = quaternion_##name##_scalar(p, **in_dataptr);    \
      } while(in_iternext(in_iter) && out_iternext(out_iter));          \
    } else {                                                            \
      NpyIter_Deallocate(in_iter);                                      \
      NpyIter_Deallocate(out_iter);                                     \
      goto fail;                                                        \
    }                                                                   \
    NpyIter_Deallocate(in_iter);                                        \
    NpyIter_Deallocate(out_iter);                                       \
    Py_INCREF(out_array);                                               \
    return out_array;                                                   \
  fail:                                                                 \
    Py_XDECREF(out_array);                                              \
    return NULL;                                                        \
  }                                                                     \
  static PyObject*                                                      \
  pyquaternion_##fake_name(PyObject* a, PyObject* b) {                  \
    /* PyObject *a_type, *a_repr, *b_type, *b_repr, *a_repr2, *b_repr2;    \ */ \
    /* char* a_char, b_char, a_char2, b_char2;                             \ */ \
    npy_int64 val64;                                                    \
    npy_int32 val32;                                                    \
    quaternion p = {0};                                                 \
    if(PyArray_Check(b)) { return pyquaternion_##fake_name##_array_operator(a, b); } \
    if(PyFloat_Check(a) && PyQuaternion_Check(b)) {                     \
      return PyQuaternion_FromQuaternion(quaternion_scalar_##name(PyFloat_AsDouble(a), ((PyQuaternion*)b)->obval)); \
    }                                                                   \
    if(PyInt_Check(a) && PyQuaternion_Check(b)) {                       \
      return PyQuaternion_FromQuaternion(quaternion_scalar_##name(PyInt_AsLong(a), ((PyQuaternion*)b)->obval)); \
    }                                                                   \
    PyQuaternion_AsQuaternion(p, a);                                    \
    if(PyQuaternion_Check(b)) {                                         \
      return PyQuaternion_FromQuaternion(quaternion_##name(p,((PyQuaternion*)b)->obval)); \
    } else if(PyFloat_Check(b)) {                                       \
      return PyQuaternion_FromQuaternion(quaternion_##name##_scalar(p,PyFloat_AsDouble(b))); \
    } else if(PyInt_Check(b)) {                                         \
      return PyQuaternion_FromQuaternion(quaternion_##name##_scalar(p,PyInt_AsLong(b))); \
    } else if(PyObject_TypeCheck(b, &PyInt64ArrType_Type)) {            \
      PyArray_ScalarAsCtype(b, &val64);                                 \
      return PyQuaternion_FromQuaternion(quaternion_##name##_scalar(p, val64)); \
    } else if(PyObject_TypeCheck(b, &PyInt32ArrType_Type)) {            \
      PyArray_ScalarAsCtype(b, &val32);                                 \
      return PyQuaternion_FromQuaternion(quaternion_##name##_scalar(p, val32)); \
    }                                                                   \
    /* a_type = PyObject_Type(a);                                          \ */ \
    /* a_repr = PyObject_Repr(a_type);                                     \ */ \
    /* a_char = PyString_AsString(a_repr);                                 \ */ \
    /* b_type = PyObject_Type(b);                                          \ */ \
    /* b_repr = PyObject_Repr(b_type);                                     \ */ \
    /* b_char = PyString_AsString(b_repr);                                 \ */ \
    /* a_repr2 = PyObject_Repr(a);                                         \ */ \
    /* a_char2 = PyString_AsString(a_repr2);                               \ */ \
    /* b_repr2 = PyObject_Repr(b);                                         \ */ \
    /* b_char2 = PyString_AsString(b_repr2);                               \ */ \
    /* fprintf (stderr, "\nfile %s, line %d, pyquaternion_%s(PyObject* a, PyObject* b).\n", __FILE__, __LINE__, #fake_name); \ */ \
    /* fprintf (stderr, "\na: '%s'\tb: '%s'", a_char, b_char);             \ */ \
    /* fprintf (stderr, "\na: '%s'\tb: '%s'", a_char2, b_char2);           \ */ \
    /* Py_DECREF(a_type);                                                  \ */ \
    /* Py_DECREF(a_repr);                                                  \ */ \
    /* Py_DECREF(b_type);                                                  \ */ \
    /* Py_DECREF(b_repr);                                                  \ */ \
    /* Py_DECREF(a_repr2);                                                 \ */ \
    /* Py_DECREF(b_repr2);                                                 \ */ \
    PyErr_SetString(PyExc_TypeError, "Binary operation involving quaternion and \\neither float nor quaternion."); \
    return NULL;                                                        \
  }
#define QQ_QS_SQ_BINARY_QUATERNION_RETURNER(name) QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(name, name)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(add)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(subtract)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(multiply)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(divide)
/* QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(true_divide, divide) */
/* QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(floor_divide, divide) */
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(power)

#define QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(fake_name, name)        \
  static PyObject*                                                      \
  pyquaternion_inplace_##fake_name(PyObject* a, PyObject* b) {          \
    quaternion* p = {0};                                                \
    /* fprintf (stderr, "file %s, line %d, pyquaternion_inplace_"#fake_name"(PyObject* a, PyObject* b).\n", __FILE__, __LINE__); \ */ \
    if(PyFloat_Check(a) || PyInt_Check(a)) {                            \
      PyErr_SetString(PyExc_TypeError, "Cannot in-place "#fake_name" a scalar by a quaternion; should be handled by python."); \
      return NULL;                                                      \
    }                                                                   \
    PyQuaternion_AsQuaternionPointer(p, a);                             \
    if(PyQuaternion_Check(b)) {                                         \
      quaternion_inplace_##name(p,((PyQuaternion*)b)->obval);           \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyFloat_Check(b)) {                                       \
      quaternion_inplace_##name##_scalar(p,PyFloat_AsDouble(b));        \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyInt_Check(b)) {                                         \
      quaternion_inplace_##name##_scalar(p,PyInt_AsLong(b));            \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    }                                                                   \
    PyErr_SetString(PyExc_TypeError, "Binary in-place operation involving quaternion and neither float nor quaternion."); \
    return NULL;                                                        \
  }
#define QQ_QS_SQ_BINARY_QUATERNION_INPLACE(name) QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(name, name)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(add)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(subtract)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(multiply)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(divide)
/* QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(true_divide, divide) */
/* QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(floor_divide, divide) */
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(power)

static PyObject *
pyquaternion__reduce(PyQuaternion* self)
{
  /* printf("\n\n\nI'm trying, most of all!\n\n\n"); */
  return Py_BuildValue("O(OOOO)", Py_TYPE(self),
                       PyFloat_FromDouble(self->obval.w), PyFloat_FromDouble(self->obval.x),
                       PyFloat_FromDouble(self->obval.y), PyFloat_FromDouble(self->obval.z));
}

static PyObject *
pyquaternion_getstate(PyQuaternion* self, PyObject* args)
{
  /* printf("\n\n\nI'm Trying, OKAY?\n\n\n"); */
  if (!PyArg_ParseTuple(args, ":getstate"))
    return NULL;
  return Py_BuildValue("OOOO",
                       PyFloat_FromDouble(self->obval.w), PyFloat_FromDouble(self->obval.x),
                       PyFloat_FromDouble(self->obval.y), PyFloat_FromDouble(self->obval.z));
}

static PyObject *
pyquaternion_setstate(PyQuaternion* self, PyObject* args)
{
  /* printf("\n\n\nI'm Trying, TOO!\n\n\n"); */
  quaternion* q;
  q = &(self->obval);

  if (!PyArg_ParseTuple(args, "dddd:setstate", &q->w, &q->x, &q->y, &q->z)) {
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}


// This is an array of methods (member functions) that will be
// available to use on the quaternion objects in python.  This is
// packaged up here, and will be used in the `tp_methods` field when
// definining the PyQuaternion_Type below.
PyMethodDef pyquaternion_methods[] = {
  // Unary bool returners
  {"nonzero", pyquaternion_nonzero, METH_NOARGS,
   "True if the quaternion has all zero components"},
  {"isnan", pyquaternion_isnan, METH_NOARGS,
   "True if the quaternion has any NAN components"},
  {"isinf", pyquaternion_isinf, METH_NOARGS,
   "True if the quaternion has any INF components"},
  {"isfinite", pyquaternion_isfinite, METH_NOARGS,
   "True if the quaternion has all finite components"},

  // Binary bool returners
  {"equal", pyquaternion_equal, METH_O,
   "True if the quaternions are PRECISELY equal"},
  {"not_equal", pyquaternion_not_equal, METH_O,
   "True if the quaternions are not PRECISELY equal"},
  {"less", pyquaternion_less, METH_O,
   "Strict dictionary ordering"},
  {"greater", pyquaternion_greater, METH_O,
   "Strict dictionary ordering"},
  {"less_equal", pyquaternion_less_equal, METH_O,
   "Dictionary ordering"},
  {"greater_equal", pyquaternion_greater_equal, METH_O,
   "Dictionary ordering"},

  // Unary float returners
  {"absolute", pyquaternion_absolute, METH_NOARGS,
   "Absolute value of quaternion"},
  {"abs", pyquaternion_absolute, METH_NOARGS,
   "Absolute value of quaternion"},
  {"norm", pyquaternion_norm, METH_NOARGS,
   "Norm (square of the absolute value) of quaternion"},
  {"angle", pyquaternion_angle, METH_NOARGS,
   "Angle through which rotor rotates"},

  // Unary quaternion returners
  // {"negative", pyquaternion_negative, METH_NOARGS,
  //  "Return the negated quaternion"},
  // {"positive", pyquaternion_positive, METH_NOARGS,
  //  "Return the quaternion itself"},
  {"conjugate", pyquaternion_conjugate, METH_NOARGS,
   "Return the complex conjugate of the quaternion"},
  {"conj", pyquaternion_conjugate, METH_NOARGS,
   "Return the complex conjugate of the quaternion"},
  {"inverse", pyquaternion_inverse, METH_NOARGS,
   "Return the inverse of the quaternion"},
  {"sqrt", pyquaternion_sqrt, METH_NOARGS,
   "Return the square-root of the quaternion"},
  {"log", pyquaternion_log, METH_NOARGS,
   "Return the logarithm (base e) of the quaternion"},
  {"exp", pyquaternion_exp, METH_NOARGS,
   "Return the exponential of the quaternion (e**q)"},
  {"normalized", pyquaternion_normalized, METH_NOARGS,
   "Return a normalized copy of the quaternion"},
  {"x_parity_conjugate", pyquaternion_x_parity_conjugate, METH_NOARGS,
   "Reflect across y-z plane (note spinorial character)"},
  {"x_parity_symmetric_part", pyquaternion_x_parity_symmetric_part, METH_NOARGS,
   "Part invariant under reflection across y-z plane (note spinorial character)"},
  {"x_parity_antisymmetric_part", pyquaternion_x_parity_antisymmetric_part, METH_NOARGS,
   "Part anti-invariant under reflection across y-z plane (note spinorial character)"},
  {"y_parity_conjugate", pyquaternion_y_parity_conjugate, METH_NOARGS,
   "Reflect across x-z plane (note spinorial character)"},
  {"y_parity_symmetric_part", pyquaternion_y_parity_symmetric_part, METH_NOARGS,
   "Part invariant under reflection across x-z plane (note spinorial character)"},
  {"y_parity_antisymmetric_part", pyquaternion_y_parity_antisymmetric_part, METH_NOARGS,
   "Part anti-invariant under reflection across x-z plane (note spinorial character)"},
  {"z_parity_conjugate", pyquaternion_z_parity_conjugate, METH_NOARGS,
   "Reflect across x-y plane (note spinorial character)"},
  {"z_parity_symmetric_part", pyquaternion_z_parity_symmetric_part, METH_NOARGS,
   "Part invariant under reflection across x-y plane (note spinorial character)"},
  {"z_parity_antisymmetric_part", pyquaternion_z_parity_antisymmetric_part, METH_NOARGS,
   "Part anti-invariant under reflection across x-y plane (note spinorial character)"},
  {"parity_conjugate", pyquaternion_parity_conjugate, METH_NOARGS,
   "Reflect all dimensions (note spinorial character)"},
  {"parity_symmetric_part", pyquaternion_parity_symmetric_part, METH_NOARGS,
   "Part invariant under negation of all vectors (note spinorial character)"},
  {"parity_antisymmetric_part", pyquaternion_parity_antisymmetric_part, METH_NOARGS,
   "Part anti-invariant under negation of all vectors (note spinorial character)"},

  // Quaternion-quaternion binary quaternion returners
  // {"add", pyquaternion_add, METH_O,
  //  "Componentwise addition"},
  // {"subtract", pyquaternion_subtract, METH_O,
  //  "Componentwise subtraction"},
  {"copysign", pyquaternion_copysign, METH_O,
   "Componentwise copysign"},

  // Quaternion-quaternion or quaternion-scalar binary quaternion returners
  // {"multiply", pyquaternion_multiply, METH_O,
  //  "Standard (geometric) quaternion product"},
  // {"divide", pyquaternion_divide, METH_O,
  //  "Standard (geometric) quaternion division"},
  // {"power", pyquaternion_power, METH_O,
  //  "q.power(p) = (q.log() * p).exp()"},

  {"__reduce__", (PyCFunction)pyquaternion__reduce, METH_NOARGS,
   "Return state information for pickling."},
  {"__getstate__", (PyCFunction)pyquaternion_getstate, METH_VARARGS,
   "Return state information for pickling."},
  {"__setstate__", (PyCFunction)pyquaternion_setstate, METH_VARARGS,
   "Reconstruct state information from pickle."},

  {NULL}
};

static PyObject* pyquaternion_num_power(PyObject* a, PyObject* b, PyObject *c) { return pyquaternion_power(a,b); }
static PyObject* pyquaternion_num_inplace_power(PyObject* a, PyObject* b, PyObject *c) { return pyquaternion_inplace_power(a,b); }
static PyObject* pyquaternion_num_negative(PyObject* a) { return pyquaternion_negative(a,NULL); }
static PyObject* pyquaternion_num_positive(PyObject* a) { return pyquaternion_positive(a,NULL); }
static PyObject* pyquaternion_num_absolute(PyObject* a) { return pyquaternion_absolute(a,NULL); }
static PyObject* pyquaternion_num_inverse(PyObject* a) { return pyquaternion_inverse(a,NULL); }
static int pyquaternion_num_nonzero(PyObject* a) {
  quaternion q = ((PyQuaternion*)a)->obval;
  return quaternion_nonzero(q);
}

static PyNumberMethods pyquaternion_as_number = {
  pyquaternion_add,               // nb_add
  pyquaternion_subtract,          // nb_subtract
  pyquaternion_multiply,          // nb_multiply
  #if PY_MAJOR_VERSION < 3
  pyquaternion_divide,            // nb_divide
  #endif
  0,                              // nb_remainder
  0,                              // nb_divmod
  pyquaternion_num_power,         // nb_power
  pyquaternion_num_negative,      // nb_negative
  pyquaternion_num_positive,      // nb_positive
  pyquaternion_num_absolute,      // nb_absolute
  pyquaternion_num_nonzero,       // nb_nonzero
  pyquaternion_num_inverse,       // nb_invert
  0,                              // nb_lshift
  0,                              // nb_rshift
  0,                              // nb_and
  0,                              // nb_xor
  0,                              // nb_or
  #if PY_MAJOR_VERSION < 3
  0,                              // nb_coerce
  #endif
  0,                              // nb_int
  #if PY_MAJOR_VERSION >= 3
  0,                              // nb_reserved
  #else
  0,                              // nb_long
  #endif
  0,                              // nb_float
  #if PY_MAJOR_VERSION < 3
  0,                              // nb_oct
  0,                              // nb_hex
  #endif
  pyquaternion_inplace_add,       // nb_inplace_add
  pyquaternion_inplace_subtract,  // nb_inplace_subtract
  pyquaternion_inplace_multiply,  // nb_inplace_multiply
  #if PY_MAJOR_VERSION < 3
  pyquaternion_inplace_divide,    // nb_inplace_divide
  #endif
  0,                              // nb_inplace_remainder
  pyquaternion_num_inplace_power, // nb_inplace_power
  0,                              // nb_inplace_lshift
  0,                              // nb_inplace_rshift
  0,                              // nb_inplace_and
  0,                              // nb_inplace_xor
  0,                              // nb_inplace_or
  pyquaternion_divide,            // nb_floor_divide
  pyquaternion_divide,            // nb_true_divide
  pyquaternion_inplace_divide,    // nb_inplace_floor_divide
  pyquaternion_inplace_divide,    // nb_inplace_true_divide
  0,                              // nb_index
  #if PY_MAJOR_VERSION >= 3
  #if PY_MINOR_VERSION >= 5
  0,                              // nb_matrix_multiply
  0,                              //  nb_inplace_matrix_multiply
  #endif
  #endif
};


// This is an array of members (member data) that will be available to
// use on the quaternion objects in python.  This is packaged up here,
// and will be used in the `tp_members` field when definining the
// PyQuaternion_Type below.
PyMemberDef pyquaternion_members[] = {
  {"real", T_DOUBLE, offsetof(PyQuaternion, obval.w), READONLY,
   "The real component of the quaternion"},
  {"w", T_DOUBLE, offsetof(PyQuaternion, obval.w), READONLY,
   "The real component of the quaternion"},
  {"x", T_DOUBLE, offsetof(PyQuaternion, obval.x), READONLY,
   "The first imaginary component of the quaternion"},
  {"y", T_DOUBLE, offsetof(PyQuaternion, obval.y), READONLY,
   "The second imaginary component of the quaternion"},
  {"z", T_DOUBLE, offsetof(PyQuaternion, obval.z), READONLY,
   "The third imaginary component of the quaternion"},
  {NULL}
};

// The quaternion can be conveniently separated into two complex
// numbers, which we call 'part a' and 'part b'.  These are useful in
// writing Wigner's D matrices directly in terms of quaternions.  This
// is essentially the column-vector presentation of spinors.
static PyObject *
pyquaternion_get_part_a(PyObject *self, void *closure)
{
  return (PyObject*) PyComplex_FromDoubles(((PyQuaternion *)self)->obval.w, ((PyQuaternion *)self)->obval.z);
}
static PyObject *
pyquaternion_get_part_b(PyObject *self, void *closure)
{
  return (PyObject*) PyComplex_FromDoubles(((PyQuaternion *)self)->obval.y, ((PyQuaternion *)self)->obval.x);
}

// This will be defined as a member function on the quaternion
// objects, so that calling "vec" will return a numpy array
// with the last three components of the quaternion.
static PyObject *
pyquaternion_get_vec(PyObject *self, void *closure)
{
  quaternion *q = &((PyQuaternion *)self)->obval;
  int nd = 1;
  npy_intp dims[1] = { 3 };
  int typenum = NPY_DOUBLE;
  PyObject* components = PyArray_SimpleNewFromData(nd, dims, typenum, &(q->x));
  Py_INCREF(self);
  PyArray_SetBaseObject((PyArrayObject*)components, self);
  return components;
}

// This will be defined as a member function on the quaternion
// objects, so that calling `q.vec = [1,2,3]`, for example,
// will set the vector components appropriately.
static int
pyquaternion_set_vec(PyObject *self, PyObject *value, void *closure)
{
  PyObject *element;
  quaternion *q = &((PyQuaternion *)self)->obval;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot set quaternion to empty value");
    return -1;
  }
  if (! (PySequence_Check(value) && PySequence_Size(value)==3) ) {
    PyErr_SetString(PyExc_TypeError,
                    "A quaternion's vector components must be set to something of length 3");
    return -1;
  }
  /* PySequence_GetItem INCREFs element. */
  element = PySequence_GetItem(value, 0);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->x = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 1);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->y = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 2);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->z = PyFloat_AsDouble(element);
  Py_DECREF(element);
  return 0;
}

// This will be defined as a member function on the quaternion
// objects, so that calling "components" will return a numpy array
// with the components of the quaternion.
static PyObject *
pyquaternion_get_components(PyObject *self, void *closure)
{
  quaternion *q = &((PyQuaternion *)self)->obval;
  int nd = 1;
  npy_intp dims[1] = { 4 };
  int typenum = NPY_DOUBLE;
  PyObject* components = PyArray_SimpleNewFromData(nd, dims, typenum, &(q->w));
  Py_INCREF(self);
  PyArray_SetBaseObject((PyArrayObject*)components, self);
  return components;
}

// This will be defined as a member function on the quaternion
// objects, so that calling `q.components = [1,2,3,4]`, for example,
// will set the components appropriately.
static int
pyquaternion_set_components(PyObject *self, PyObject *value, void *closure)
{
  PyObject *element;
  quaternion *q = &((PyQuaternion *)self)->obval;
  if (value == NULL) {
    PyErr_SetString(PyExc_ValueError, "Cannot set quaternion to empty value");
    return -1;
  }
  if (! (PySequence_Check(value) && PySequence_Size(value)==4) ) {
    PyErr_SetString(PyExc_TypeError,
                    "A quaternion's components must be set to something of length 4");
    return -1;
  }
  element = PySequence_GetItem(value, 0);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->w = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 1);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->x = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 2);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->y = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 3);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->z = PyFloat_AsDouble(element);
  Py_DECREF(element);
  return 0;
}

// This collects the methods for getting and setting elements of the
// quaternion.  This is packaged up here, and will be used in the
// `tp_getset` field when definining the PyQuaternion_Type
// below.
PyGetSetDef pyquaternion_getset[] = {
  {"a", pyquaternion_get_part_a, NULL,
   "The complex number (w+i*z)", NULL},
  {"b", pyquaternion_get_part_b, NULL,
   "The complex number (y+i*x)", NULL},
  {"vec", pyquaternion_get_vec, pyquaternion_set_vec,
   "The vector part (x,y,z) of the quaternion as a numpy array", NULL},
  {"components", pyquaternion_get_components, pyquaternion_set_components,
   "The components (w,x,y,z) of the quaternion as a numpy array", NULL},
  {NULL}
};



static PyObject*
pyquaternion_richcompare(PyObject* a, PyObject* b, int op)
{
  quaternion x = {0};
  quaternion y = {0};
  int result = 0;
  PyQuaternion_AsQuaternion(x,a);
  PyQuaternion_AsQuaternion(y,b);
  #define COMPARISONOP(py,op) case py: result = quaternion_##op(x,y); break;
  switch (op) {
    COMPARISONOP(Py_LT,less)
    COMPARISONOP(Py_LE,less_equal)
    COMPARISONOP(Py_EQ,equal)
    COMPARISONOP(Py_NE,not_equal)
    COMPARISONOP(Py_GT,greater)
    COMPARISONOP(Py_GE,greater_equal)
  };
  #undef COMPARISONOP
  return PyBool_FromLong(result);
}


static long
pyquaternion_hash(PyObject *o)
{
  quaternion q = ((PyQuaternion *)o)->obval;
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
pyquaternion_repr(PyObject *o)
{
  char str[128];
  quaternion q = ((PyQuaternion *)o)->obval;
  sprintf(str, "quaternion(%.15g, %.15g, %.15g, %.15g)", q.w, q.x, q.y, q.z);
  return PyUString_FromString(str);
}

static PyObject *
pyquaternion_str(PyObject *o)
{
  char str[128];
  quaternion q = ((PyQuaternion *)o)->obval;
  sprintf(str, "quaternion(%.15g, %.15g, %.15g, %.15g)", q.w, q.x, q.y, q.z);
  return PyUString_FromString(str);
}


// This establishes the quaternion as a python object (not yet a numpy
// scalar type).  The name may be a little counterintuitive; the idea
// is that this will be a type that can be used as an array dtype.
// Note that many of the slots below will be filled later, after the
// corresponding functions are defined.
static PyTypeObject PyQuaternion_Type = {
#if PY_MAJOR_VERSION >= 3
  PyVarObject_HEAD_INIT(NULL, 0)
#else
  PyObject_HEAD_INIT(NULL)
  0,                                          // ob_size
#endif
  "quaternion",                               // tp_name
  sizeof(PyQuaternion),                       // tp_basicsize
  0,                                          // tp_itemsize
  0,                                          // tp_dealloc
  0,                                          // tp_print
  0,                                          // tp_getattr
  0,                                          // tp_setattr
#if PY_MAJOR_VERSION >= 3
  0,                                          // tp_reserved
#else
  0,                                          // tp_compare
#endif
  pyquaternion_repr,                          // tp_repr
  &pyquaternion_as_number,                    // tp_as_number
  0,                                          // tp_as_sequence
  0,                                          // tp_as_mapping
  pyquaternion_hash,                          // tp_hash
  0,                                          // tp_call
  pyquaternion_str,                           // tp_str
  0,                                          // tp_getattro
  0,                                          // tp_setattro
  0,                                          // tp_as_buffer
#if PY_MAJOR_VERSION >= 3
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   // tp_flags
#else
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, // tp_flags
#endif
  0,                                          // tp_doc
  0,                                          // tp_traverse
  0,                                          // tp_clear
  pyquaternion_richcompare,                   // tp_richcompare
  0,                                          // tp_weaklistoffset
  0,                                          // tp_iter
  0,                                          // tp_iternext
  pyquaternion_methods,                       // tp_methods
  pyquaternion_members,                       // tp_members
  pyquaternion_getset,                        // tp_getset
  0,                                          // tp_base; will be reset to &PyGenericArrType_Type after numpy import
  0,                                          // tp_dict
  0,                                          // tp_descr_get
  0,                                          // tp_descr_set
  0,                                          // tp_dictoffset
  pyquaternion_init,                          // tp_init
  0,                                          // tp_alloc
  pyquaternion_new,                           // tp_new
  0,                                          // tp_free
  0,                                          // tp_is_gc
  0,                                          // tp_bases
  0,                                          // tp_mro
  0,                                          // tp_cache
  0,                                          // tp_subclasses
  0,                                          // tp_weaklist
  0,                                          // tp_del
#if PY_VERSION_HEX >= 0x02060000
  0,                                          // tp_version_tag
#endif
};

// Functions implementing internal features. Not all of these function
// pointers must be defined for a given type. The required members are
// nonzero, copyswap, copyswapn, setitem, getitem, and cast.
static PyArray_ArrFuncs _PyQuaternion_ArrFuncs;

static npy_bool
QUATERNION_nonzero (char *ip, PyArrayObject *ap)
{
  quaternion q;
  quaternion zero = {0,0,0,0};
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
  return (npy_bool) !quaternion_equal(q, zero);
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

static int QUATERNION_setitem(PyObject* item, void* data, void* ap)
{
  PyObject *element;
  quaternion q = {0};
  if(PyQuaternion_Check(item)) {
    memcpy(data,&(((PyQuaternion *)item)->obval),sizeof(quaternion));
  } else if(PySequence_Check(item) && PySequence_Length(item)==4) {
    element = PySequence_GetItem(item, 0);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    q.w = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 1);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    q.x = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 2);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    q.y = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 3);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    q.z = PyFloat_AsDouble(element);
    Py_DECREF(element);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown input to QUATERNION_setitem");
    return -1;
  }
  return 0;
}

// When a numpy array of dtype=quaternion is indexed, this function is
// called, returning a new quaternion object with a copy of the
// data... sometimes...
static PyObject *
QUATERNION_getitem(void* data, void* arr)
{
  quaternion q;
  memcpy(&q,data,sizeof(quaternion));
  return PyQuaternion_FromQuaternion(q);
}

static int
QUATERNION_compare(quaternion *pa, quaternion *pb, PyArrayObject *NPY_UNUSED(ap))
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
    // nan encountered; it's maximal
    return 0;
  }

  for (i = 1; i < n; i++) {
    ip++;
    //Propagate nans, similarly as max() and min()
    if (!(quaternion_less_equal(*ip, mp))) {  // negated, for correct nan handling
      mp = *ip;
      *max_ind = i;
      if (quaternion_isnan(mp)) {
        // nan encountered, it's maximal
        break;
      }
    }
  }
  return 0;
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

// This is a macro (followed by applications of the macro) that cast
// the input types to standard quaternions with only a nonzero scalar
// part.
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

// This is a macro (followed by applications of the macro) that cast
// the input complex types to standard quaternions with only the first
// two components nonzero.  This doesn't make a whole lot of sense to
// me, and may be removed in the future.
#define MAKE_CT_TO_QUATERNION(TYPE, type)                               \
  static void                                                           \
  TYPE ## _to_quaternion(type *ip, quaternion *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      op->w = (double)(*ip++);                                          \
      op->x = (double)(*ip++);                                          \
      op->y = 0;                                                        \
      op->z = 0;                                                        \
    }                                                                   \
  }
MAKE_CT_TO_QUATERNION(CFLOAT, npy_float);
MAKE_CT_TO_QUATERNION(CDOUBLE, npy_double);
MAKE_CT_TO_QUATERNION(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
  PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
  PyArray_RegisterCastFunc(descr, destType, castfunc);
  PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
  Py_DECREF(descr);
}


// This is a macro that will be used to define the various basic unary
// quaternion functions, so that they can be applied quickly to a
// numpy array of quaternions.
#define UNARY_GEN_UFUNC(ufunc_name, func_name, ret_type)        \
  static void                                                           \
  quaternion_##ufunc_name##_ufunc(char** args, npy_intp* dimensions,    \
                                  npy_intp* steps, void* data) {        \
    /* fprintf (stderr, "file %s, line %d, quaternion_%s_ufunc.\n", __FILE__, __LINE__, #ufunc_name); */ \
    char *ip1 = args[0], *op1 = args[1];                                \
    npy_intp is1 = steps[0], os1 = steps[1];                            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){                     \
      const quaternion in1 = *(quaternion *)ip1;                        \
      *((ret_type *)op1) = quaternion_##func_name(in1);};}
#define UNARY_UFUNC(name, ret_type) \
  UNARY_GEN_UFUNC(name, name, ret_type)
// And these all do the work mentioned above, using the macro
UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(norm, npy_double)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(angle, npy_double)
UNARY_UFUNC(sqrt, quaternion)
UNARY_UFUNC(log, quaternion)
UNARY_UFUNC(exp, quaternion)
UNARY_UFUNC(negative, quaternion)
UNARY_UFUNC(conjugate, quaternion)
UNARY_GEN_UFUNC(invert, inverse, quaternion)
UNARY_UFUNC(normalized, quaternion)
UNARY_UFUNC(x_parity_conjugate, quaternion)
UNARY_UFUNC(x_parity_symmetric_part, quaternion)
UNARY_UFUNC(x_parity_antisymmetric_part, quaternion)
UNARY_UFUNC(y_parity_conjugate, quaternion)
UNARY_UFUNC(y_parity_symmetric_part, quaternion)
UNARY_UFUNC(y_parity_antisymmetric_part, quaternion)
UNARY_UFUNC(z_parity_conjugate, quaternion)
UNARY_UFUNC(z_parity_symmetric_part, quaternion)
UNARY_UFUNC(z_parity_antisymmetric_part, quaternion)
UNARY_UFUNC(parity_conjugate, quaternion)
UNARY_UFUNC(parity_symmetric_part, quaternion)
UNARY_UFUNC(parity_antisymmetric_part, quaternion)

// This is a macro that will be used to define the various basic binary
// quaternion functions, so that they can be applied quickly to a
// numpy array of quaternions.
#define BINARY_GEN_UFUNC(ufunc_name, func_name, arg_type1, arg_type2, ret_type) \
  static void                                                           \
  quaternion_##ufunc_name##_ufunc(char** args, npy_intp* dimensions,    \
                                  npy_intp* steps, void* data) {        \
    /* fprintf (stderr, "file %s, line %d, quaternion_%s_ufunc.\n", __FILE__, __LINE__, #ufunc_name); */ \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];                \
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {        \
      const arg_type1 in1 = *(arg_type1 *)ip1;                          \
      const arg_type2 in2 = *(arg_type2 *)ip2;                          \
      *((ret_type *)op1) = quaternion_##func_name(in1, in2);            \
    };                                                                  \
  };
// A couple special-case versions of the above
#define BINARY_UFUNC(name, ret_type)                    \
  BINARY_GEN_UFUNC(name, name, quaternion, quaternion, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)                             \
  BINARY_GEN_UFUNC(name##_scalar, name##_scalar, quaternion, npy_double, ret_type) \
  BINARY_GEN_UFUNC(scalar_##name, scalar_##name, npy_double, quaternion, ret_type)
// And these all do the work mentioned above, using the macros
BINARY_UFUNC(add, quaternion)
BINARY_UFUNC(subtract, quaternion)
BINARY_UFUNC(multiply, quaternion)
BINARY_UFUNC(divide, quaternion)
BINARY_GEN_UFUNC(true_divide, divide, quaternion, quaternion, quaternion)
BINARY_GEN_UFUNC(floor_divide, divide, quaternion, quaternion, quaternion)
BINARY_UFUNC(power, quaternion)
BINARY_UFUNC(copysign, quaternion)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)
BINARY_SCALAR_UFUNC(add, quaternion)
BINARY_SCALAR_UFUNC(subtract, quaternion)
BINARY_SCALAR_UFUNC(multiply, quaternion)
BINARY_SCALAR_UFUNC(divide, quaternion)
BINARY_GEN_UFUNC(true_divide_scalar, divide_scalar, quaternion, npy_double, quaternion)
BINARY_GEN_UFUNC(floor_divide_scalar, divide_scalar, quaternion, npy_double, quaternion)
BINARY_GEN_UFUNC(scalar_true_divide, scalar_divide, npy_double, quaternion, quaternion)
BINARY_GEN_UFUNC(scalar_floor_divide, scalar_divide, npy_double, quaternion, quaternion)
BINARY_SCALAR_UFUNC(power, quaternion)


// Used to create unit rotor from spherical coordinates, this can be
// imported directly from quaternion.numpy_quaternion
static PyObject*
quaternion_from_spherical_coords(PyObject *self, PyObject *args )
{
  double vartheta, varphi;
  PyQuaternion* Q = (PyQuaternion*)PyQuaternion_Type.tp_alloc(&PyQuaternion_Type,0);
  if (!PyArg_ParseTuple(args, "dd", &vartheta, &varphi)) {
    return NULL;
  }
  Q->obval = quaternion_create_from_spherical_coords(vartheta, varphi);
  return (PyObject*)Q;
}

// Used to create unit rotor from Euler angles, this can be imported
// directly from quaternion.numpy_quaternion
static PyObject*
quaternion_from_euler_angles(PyObject *self, PyObject *args )
{
  double alpha, beta, gamma;
  PyQuaternion* Q = (PyQuaternion*)PyQuaternion_Type.tp_alloc(&PyQuaternion_Type,0);
  if (!PyArg_ParseTuple(args, "ddd", &alpha, &beta, &gamma)) {
    return NULL;
  }
  Q->obval = quaternion_create_from_euler_angles(alpha, beta, gamma);
  return (PyObject*)Q;
}

static PyObject*
pyquaternion_rotor_intrinsic_distance(PyObject *self, PyObject *args)
{
  PyObject* Q1 = {0};
  PyObject* Q2 = {0};
  if (!PyArg_ParseTuple(args, "OO", &Q1, &Q2)) {
    return NULL;
  }
  return PyFloat_FromDouble(rotor_intrinsic_distance(((PyQuaternion*)Q1)->obval, ((PyQuaternion*)Q2)->obval));
}

static PyObject*
pyquaternion_rotor_chordal_distance(PyObject *self, PyObject *args)
{
  PyObject* Q1 = {0};
  PyObject* Q2 = {0};
  if (!PyArg_ParseTuple(args, "OO", &Q1, &Q2)) {
    return NULL;
  }
  return PyFloat_FromDouble(rotor_chordal_distance(((PyQuaternion*)Q1)->obval, ((PyQuaternion*)Q2)->obval));
}

static PyObject*
pyquaternion_rotation_intrinsic_distance(PyObject *self, PyObject *args)
{
  PyObject* Q1 = {0};
  PyObject* Q2 = {0};
  if (!PyArg_ParseTuple(args, "OO", &Q1, &Q2)) {
    return NULL;
  }
  return PyFloat_FromDouble(rotation_intrinsic_distance(((PyQuaternion*)Q1)->obval, ((PyQuaternion*)Q2)->obval));
}

static PyObject*
pyquaternion_rotation_chordal_distance(PyObject *self, PyObject *args)
{
  PyObject* Q1 = {0};
  PyObject* Q2 = {0};
  if (!PyArg_ParseTuple(args, "OO", &Q1, &Q2)) {
    return NULL;
  }
  return PyFloat_FromDouble(rotation_chordal_distance(((PyQuaternion*)Q1)->obval, ((PyQuaternion*)Q2)->obval));
}

// Interface to the module-level slerp function
static PyObject*
pyquaternion_slerp_evaluate(PyObject *self, PyObject *args)
{
  double tau;
  PyObject* Q1 = {0};
  PyObject* Q2 = {0};
  PyQuaternion* Q = (PyQuaternion*)PyQuaternion_Type.tp_alloc(&PyQuaternion_Type,0);
  if (!PyArg_ParseTuple(args, "OOd", &Q1, &Q2, &tau)) {
    return NULL;
  }
  Q->obval = slerp(((PyQuaternion*)Q1)->obval, ((PyQuaternion*)Q2)->obval, tau);
  return (PyObject*)Q;
}

// Interface to the evaluate a squad interpolant at a particular time
static PyObject*
pyquaternion_squad_evaluate(PyObject *self, PyObject *args)
{
  double tau_i;
  PyObject* q_i = {0};
  PyObject* a_i = {0};
  PyObject* b_ip1 = {0};
  PyObject* q_ip1 = {0};
  PyQuaternion* Q = (PyQuaternion*)PyQuaternion_Type.tp_alloc(&PyQuaternion_Type,0);
  if (!PyArg_ParseTuple(args, "dOOOO", &tau_i, &q_i, &a_i, &b_ip1, &q_ip1)) {
    return NULL;
  }
  Q->obval = squad_evaluate(tau_i,
                        ((PyQuaternion*)q_i)->obval, ((PyQuaternion*)a_i)->obval,
                        ((PyQuaternion*)b_ip1)->obval, ((PyQuaternion*)q_ip1)->obval);
  return (PyObject*)Q;
}

// This will be used to create the ufunc needed for `slerp`, which
// evaluates the interpolant at a point.  The method for doing this
// was pieced together from examples given on the page
// <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>
static void
slerp_loop(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
  npy_intp i;
  double tau_i;
  quaternion *q_1, *q_2;

  npy_intp is1=steps[0];
  npy_intp is2=steps[1];
  npy_intp is3=steps[2];
  npy_intp os=steps[3];
  npy_intp n=dimensions[0];

  char *i1=args[0];
  char *i2=args[1];
  char *i3=args[2];
  char *op=args[3];

  for (i = 0; i < n; i++) {
    q_1 = (quaternion*)i1;
    q_2 = (quaternion*)i2;
    tau_i = *(double *)i3;

    *((quaternion *)op) = slerp(*q_1, *q_2, tau_i);

    i1 += is1;
    i2 += is2;
    i3 += is3;
    op += os;
  }
}

// This will be used to create the ufunc needed for `squad`, which
// evaluates the interpolant at a point.  The method for doing this
// was pieced together from examples given on the page
// <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>
static void
squad_loop(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
  npy_intp i;
  double tau_i;
  quaternion *q_i, *a_i, *b_ip1, *q_ip1;

  npy_intp is1=steps[0];
  npy_intp is2=steps[1];
  npy_intp is3=steps[2];
  npy_intp is4=steps[3];
  npy_intp is5=steps[4];
  npy_intp os=steps[5];
  npy_intp n=dimensions[0];

  char *i1=args[0];
  char *i2=args[1];
  char *i3=args[2];
  char *i4=args[3];
  char *i5=args[4];
  char *op=args[5];

  for (i = 0; i < n; i++) {
    tau_i = *(double *)i1;
    q_i = (quaternion*)i2;
    a_i = (quaternion*)i3;
    b_ip1 = (quaternion*)i4;
    q_ip1 = (quaternion*)i5;

    *((quaternion *)op) = squad_evaluate(tau_i, *q_i, *a_i, *b_ip1, *q_ip1);

    i1 += is1;
    i2 += is2;
    i3 += is3;
    i4 += is4;
    i5 += is5;
    op += os;
  }
}

// This contains assorted other top-level methods for the module
static PyMethodDef QuaternionMethods[] = {
  {"from_spherical_coords", quaternion_from_spherical_coords, METH_VARARGS,
   "Generate unit quaternion from spherical coordinates"},
  {"from_euler_angles", quaternion_from_euler_angles, METH_VARARGS,
   "Generate unit quaternion from Euler angles as `exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)`"},
  {"rotor_intrinsic_distance", pyquaternion_rotor_intrinsic_distance, METH_VARARGS,
   "Distance measure intrinsic to rotor manifold"},
  {"rotor_chordal_distance", pyquaternion_rotor_chordal_distance, METH_VARARGS,
   "Distance measure from embedding of rotor manifold"},
  {"rotation_intrinsic_distance", pyquaternion_rotation_intrinsic_distance, METH_VARARGS,
   "Distance measure intrinsic to rotation manifold"},
  {"rotation_chordal_distance", pyquaternion_rotation_chordal_distance, METH_VARARGS,
   "Distance measure from embedding of rotation manifold"},
  {"slerp_evaluate", pyquaternion_slerp_evaluate, METH_VARARGS,
   "Interpolate linearly along the geodesic between two rotors \n\n"
   "See also `numpy.slerp_vectorized` for a vectorized version of this function, and\n"
   "`quaternion.slerp` for the most useful form, which automatically finds the correct\n"
   "rotors to interpolate and the relative time to which they must be interpolated."},
  {"squad_evaluate", pyquaternion_squad_evaluate, METH_VARARGS,
   "Interpolate linearly along the geodesic between two rotors\n\n"
   "See also `numpy.squad_vectorized` for a vectorized version of this function, and\n"
   "`quaternion.squad` for the most useful form, which automatically finds the correct\n"
   "rotors to interpolate and the relative time to which they must be interpolated."},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_quaternion",
    NULL,
    -1,
    QuaternionMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

#define INITERROR return NULL

// This is the initialization function that does the setup
PyMODINIT_FUNC PyInit_numpy_quaternion(void) {

#else

#define INITERROR return

// This is the initialization function that does the setup
PyMODINIT_FUNC initnumpy_quaternion(void) {

#endif

  PyObject *module;
  PyObject *module_dict;
  PyObject *tmp_ufunc;
  PyObject *slerp_evaluate_ufunc;
  PyObject *squad_evaluate_ufunc;
  int quaternionNum;
  int arg_types[3];
  PyArray_Descr* arg_dtypes[6];
  PyObject* numpy;
  PyObject* numpy_dict;

  // Initialize a (for now, empty) module
#if PY_MAJOR_VERSION >= 3
  module = PyModule_Create(&moduledef);
#else
  module = Py_InitModule("numpy_quaternion", QuaternionMethods);
#endif

  if(module==NULL) {
    INITERROR;
  }

  module_dict = PyModule_GetDict(module);

  // Initialize numpy
  import_array();
  if (PyErr_Occurred()) {
    INITERROR;
  }
  import_umath();
  if (PyErr_Occurred()) {
    INITERROR;
  }
  numpy = PyImport_ImportModule("numpy");
  if (!numpy) {
    INITERROR;
  }
  numpy_dict = PyModule_GetDict(numpy);
  if (!numpy_dict) {
    INITERROR;
  }

  // Register the quaternion array base type.  Couldn't do this until
  // after we imported numpy (above)
  PyQuaternion_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&PyQuaternion_Type) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "Could not initialize PyQuaternion_Type.");
    INITERROR;
  }

  // The array functions, to be used below.  This InitArrFuncs
  // function is a convenient way to set all the fields to zero
  // initially, so we don't get undefined behavior.
  PyArray_InitArrFuncs(&_PyQuaternion_ArrFuncs);
  _PyQuaternion_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)QUATERNION_nonzero;
  _PyQuaternion_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)QUATERNION_copyswap;
  _PyQuaternion_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)QUATERNION_copyswapn;
  _PyQuaternion_ArrFuncs.setitem = (PyArray_SetItemFunc*)QUATERNION_setitem;
  _PyQuaternion_ArrFuncs.getitem = (PyArray_GetItemFunc*)QUATERNION_getitem;
  _PyQuaternion_ArrFuncs.compare = (PyArray_CompareFunc*)QUATERNION_compare;
  _PyQuaternion_ArrFuncs.argmax = (PyArray_ArgFunc*)QUATERNION_argmax;
  _PyQuaternion_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)QUATERNION_fillwithscalar;

  // The quaternion array descr
  quaternion_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  quaternion_descr->typeobj = &PyQuaternion_Type;
  quaternion_descr->kind = 'q';
  quaternion_descr->type = 'j';
  quaternion_descr->byteorder = '=';
  quaternion_descr->flags = 0;
  quaternion_descr->type_num = 0; // assigned at registration
  quaternion_descr->elsize = 8*4;
  quaternion_descr->alignment = 8;
  quaternion_descr->subarray = NULL;
  quaternion_descr->fields = NULL;
  quaternion_descr->names = NULL;
  quaternion_descr->f = &_PyQuaternion_ArrFuncs;
  quaternion_descr->metadata = NULL;
  quaternion_descr->c_metadata = NULL;

  Py_INCREF(&PyQuaternion_Type);
  quaternionNum = PyArray_RegisterDataType(quaternion_descr);

  if (quaternionNum < 0) {
    INITERROR;
  }

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

  // These macros will be used below
  #define REGISTER_UFUNC(name)                                          \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                quaternion_descr->type_num, quaternion_##name##_ufunc, arg_types, NULL)
  #define REGISTER_SCALAR_UFUNC(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                quaternion_descr->type_num, quaternion_scalar_##name##_ufunc, arg_types, NULL)
  #define REGISTER_UFUNC_SCALAR(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                quaternion_descr->type_num, quaternion_##name##_scalar_ufunc, arg_types, NULL)
  #define REGISTER_NEW_UFUNC_GENERAL(pyname, cname, nargin, nargout, doc) \
    tmp_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, nargin, nargout, \
                                        PyUFunc_None, #pyname, doc, 0); \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)tmp_ufunc,             \
                                quaternion_descr->type_num, quaternion_##cname##_ufunc, arg_types, NULL); \
    PyDict_SetItemString(numpy_dict, #pyname, tmp_ufunc);               \
    Py_DECREF(tmp_ufunc)
  #define REGISTER_NEW_UFUNC(name, nargin, nargout, doc)                \
    REGISTER_NEW_UFUNC_GENERAL(name, name, nargin, nargout, doc)

  // quat -> bool
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = NPY_BOOL;
  REGISTER_UFUNC(isnan);
  /* // Already works: REGISTER_UFUNC(nonzero); */
  REGISTER_UFUNC(isinf);
  REGISTER_UFUNC(isfinite);

  // quat -> double
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  REGISTER_NEW_UFUNC(norm, 1, 1,
                     "Return norm of each quaternion.\n");
  REGISTER_UFUNC(absolute);
  REGISTER_NEW_UFUNC_GENERAL(angle_of_rotor, angle, 1, 1,
                             "Return angle of rotation, assuming input is a unit rotor\n");

  // quat -> quat
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = quaternion_descr->type_num;
  REGISTER_NEW_UFUNC_GENERAL(sqrt_of_rotor, sqrt, 1, 1,
                             "Return square-root of rotor.  Assumes input has unit norm.\n");
  REGISTER_UFUNC(log);
  REGISTER_UFUNC(exp);
  REGISTER_NEW_UFUNC(normalized, 1, 1,
                     "Normalize all quaternions in this array\n");
  REGISTER_NEW_UFUNC(x_parity_conjugate, 1, 1,
                     "Reflect across y-z plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(x_parity_symmetric_part, 1, 1,
                     "Part invariant under reflection across y-z plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(x_parity_antisymmetric_part, 1, 1,
                     "Part anti-invariant under reflection across y-z plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(y_parity_conjugate, 1, 1,
                     "Reflect across x-z plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(y_parity_symmetric_part, 1, 1,
                     "Part invariant under reflection across x-z plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(y_parity_antisymmetric_part, 1, 1,
                     "Part anti-invariant under reflection across x-z plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(z_parity_conjugate, 1, 1,
                     "Reflect across x-y plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(z_parity_symmetric_part, 1, 1,
                     "Part invariant under reflection across x-y plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(z_parity_antisymmetric_part, 1, 1,
                     "Part anti-invariant under reflection across x-y plane (note spinorial character)\n");
  REGISTER_NEW_UFUNC(parity_conjugate, 1, 1,
                     "Reflect all dimensions (note spinorial character)\n");
  REGISTER_NEW_UFUNC(parity_symmetric_part, 1, 1,
                     "Part invariant under reversal of all vectors (note spinorial character)\n");
  REGISTER_NEW_UFUNC(parity_antisymmetric_part, 1, 1,
                     "Part anti-invariant under reversal of all vectors (note spinorial character)\n");
  REGISTER_UFUNC(negative);
  REGISTER_UFUNC(conjugate);
  REGISTER_UFUNC(invert);

  // quat, quat -> bool
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = quaternion_descr->type_num;
  arg_types[2] = NPY_BOOL;
  REGISTER_UFUNC(equal);
  REGISTER_UFUNC(not_equal);
  REGISTER_UFUNC(less);
  REGISTER_UFUNC(less_equal);

  // double, quat -> quat
  arg_types[0] = NPY_DOUBLE;
  arg_types[1] = quaternion_descr->type_num;
  arg_types[2] = quaternion_descr->type_num;
  REGISTER_SCALAR_UFUNC(add);
  REGISTER_SCALAR_UFUNC(subtract);
  REGISTER_SCALAR_UFUNC(multiply);
  REGISTER_SCALAR_UFUNC(divide);
  REGISTER_SCALAR_UFUNC(true_divide);
  REGISTER_SCALAR_UFUNC(floor_divide);
  REGISTER_SCALAR_UFUNC(power);

  // quat, double -> quat
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  arg_types[2] = quaternion_descr->type_num;
  REGISTER_UFUNC_SCALAR(add);
  REGISTER_UFUNC_SCALAR(subtract);
  REGISTER_UFUNC_SCALAR(multiply);
  REGISTER_UFUNC_SCALAR(divide);
  REGISTER_UFUNC_SCALAR(true_divide);
  REGISTER_UFUNC_SCALAR(floor_divide);
  REGISTER_UFUNC_SCALAR(power);

  // quat, quat -> quat
  arg_types[0] = quaternion_descr->type_num;
  arg_types[1] = quaternion_descr->type_num;
  arg_types[2] = quaternion_descr->type_num;
  REGISTER_UFUNC(add);
  REGISTER_UFUNC(subtract);
  REGISTER_UFUNC(multiply);
  REGISTER_UFUNC(divide);
  REGISTER_UFUNC(true_divide);
  REGISTER_UFUNC(floor_divide);
  REGISTER_UFUNC(power);
  REGISTER_UFUNC(copysign);

  /* I think before I do the following, I'll have to update numpy_dict
   * somehow, presumably with something related to
   * `PyUFunc_RegisterLoopForType`.  I should also do this for the
   * various other methods defined above. */

  // Create a custom ufunc and register it for loops.  The method for
  // doing this was pieced together from examples given on the page
  // <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>
  arg_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
  arg_dtypes[1] = quaternion_descr;
  arg_dtypes[2] = quaternion_descr;
  arg_dtypes[3] = quaternion_descr;
  arg_dtypes[4] = quaternion_descr;
  arg_dtypes[5] = quaternion_descr;
  squad_evaluate_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 5, 1,
                                                 PyUFunc_None, "squad_vectorized",
                                                 "Calculate squad from arrays of (tau, q_i, a_i, b_ip1, q_ip1)\n\n"
                                                 "See `quaternion.squad` for an easier-to-use version of this function",
                                                  0);
  PyUFunc_RegisterLoopForDescr((PyUFuncObject*)squad_evaluate_ufunc,
                               quaternion_descr,
                               &squad_loop,
                               arg_dtypes,
                               NULL);
  PyDict_SetItemString(numpy_dict, "squad_vectorized", squad_evaluate_ufunc);
  Py_DECREF(squad_evaluate_ufunc);

  // Create a custom ufunc and register it for loops.  The method for
  // doing this was pieced together from examples given on the page
  // <http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>
  arg_dtypes[0] = quaternion_descr;
  arg_dtypes[1] = quaternion_descr;
  arg_dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
  slerp_evaluate_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 3, 1,
                                                 PyUFunc_None, "slerp_vectorized",
                                                 "Calculate slerp from arrays of (q_1, q_2, tau)\n\n"
                                                 "See `quaternion.slerp` for an easier-to-use version of this function",
                                                  0);
  PyUFunc_RegisterLoopForDescr((PyUFuncObject*)slerp_evaluate_ufunc,
                               quaternion_descr,
                               &slerp_loop,
                               arg_dtypes,
                               NULL);
  PyDict_SetItemString(numpy_dict, "slerp_vectorized", slerp_evaluate_ufunc);
  Py_DECREF(slerp_evaluate_ufunc);

  // Add the constant `_QUATERNION_EPS` to the module as `quaternion._eps`
  PyModule_AddObject(module, "_eps", PyFloat_FromDouble(_QUATERNION_EPS));
 
  // Finally, add this quaternion object to the quaternion module itself
  PyModule_AddObject(module, "quaternion", (PyObject *)&PyQuaternion_Type);


#if PY_MAJOR_VERSION >= 3
    return module;
#else
    return;
#endif
}
