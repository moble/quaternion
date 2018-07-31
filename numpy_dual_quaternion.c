// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/dual_quaternion/blob/master/LICENSE>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"

#include "dual_quaternion.h"

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


// The basic python object holding a dual_quaternion
typedef struct {
  PyObject_HEAD
  dual_quaternion obval;
} PyDualQuaternion;

static PyTypeObject PyDualQuaternion_Type;

// This is the crucial feature that will make a dual_quaternion into a
// built-in numpy data type.  We will describe its features below.
PyArray_Descr* dual_quaternion_descr;


static NPY_INLINE int
PyDualQuaternion_Check(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&PyDualQuaternion_Type);
}

static PyObject*
PyDualQuaternion_FromDualQuaternion(dual_quaternion q) {
  PyDualQuaternion* p = (PyDualQuaternion*)PyDualQuaternion_Type.tp_alloc(&PyDualQuaternion_Type,0);
  if (p) { p->obval = q; }
  return (PyObject*)p;
}

// TODO: Add list/tuple conversions
#define PyDualQuaternion_AsDualQuaternion(q, o)                                 \
  /* fprintf (stderr, "file %s, line %d., PyDualQuaternion_AsDualQuaternion\n", __FILE__, __LINE__); */ \
  if(PyDualQuaternion_Check(o)) {                                           \
    q = ((PyDualQuaternion*)o)->obval;                                      \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not a dual_quaternion.");               \
    return NULL;                                                        \
  }

#define PyDualQuaternion_AsDualQuaternionPointer(q, o)                          \
  /* fprintf (stderr, "file %s, line %d, PyDualQuaternion_AsDualQuaternionPointer.\n", __FILE__, __LINE__); */ \
  if(PyDualQuaternion_Check(o)) {                                           \
    q = &((PyDualQuaternion*)o)->obval;                                     \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not a dual_quaternion.");               \
    return NULL;                                                        \
  }

static PyObject *
pydual_quaternion_new(PyTypeObject *type, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
  PyDualQuaternion* self;
  self = (PyDualQuaternion *)type->tp_alloc(type, 0);
  return (PyObject *)self;
}

static int
pydual_quaternion_init(PyObject *self, PyObject *args, PyObject *kwds)
{
  // "A good rule of thumb is that for immutable types, all
  // initialization should take place in `tp_new`, while for mutable
  // types, most initialization should be deferred to `tp_init`."
  // ---Python 2.7.8 docs

  Py_ssize_t size = PyTuple_Size(args);
  dual_quaternion* q;
  q = &(((PyDualQuaternion*)self)->obval);
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError,
                    "dual_quaternion constructor takes no keyword arguments");
    return -1;
  }
  if (((size == 8) && (!PyArg_ParseTuple(args, "dddddddd",
           &q->w, &q->x, &q->y, &q->z, &q->er, &q->ei, &q->ej, &q->ek)))
      || ((size<8) || (size>8))) {
    PyErr_SetString(PyExc_TypeError,
                    "dual_quaternion constructor takes eight float arguments");
    return -1;
  }

  return 0;
}

#define UNARY_BOOL_RETURNER(name)                                       \
  static PyObject*                                                      \
  pydual_quaternion_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {           \
    dual_quaternion q = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};       \
    PyDualQuaternion_AsDualQuaternion(q, a);                                    \
    return PyBool_FromLong(dual_quaternion_##name(q));                       \
  }
UNARY_BOOL_RETURNER(nonzero)
UNARY_BOOL_RETURNER(isnan)
UNARY_BOOL_RETURNER(isinf)
UNARY_BOOL_RETURNER(isfinite)

#define BINARY_BOOL_RETURNER(name)                                      \
  static PyObject*                                                      \
  pydual_quaternion_##name(PyObject* a, PyObject* b) {                       \
    dual_quaternion p = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};            \
    dual_quaternion q = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};            \
    PyDualQuaternion_AsDualQuaternion(p, a);                                    \
    PyDualQuaternion_AsDualQuaternion(q, b);                                    \
    return PyBool_FromLong(dual_quaternion_##name(p,q));                     \
  }
BINARY_BOOL_RETURNER(equal)
BINARY_BOOL_RETURNER(not_equal)
BINARY_BOOL_RETURNER(less)
BINARY_BOOL_RETURNER(greater)
BINARY_BOOL_RETURNER(less_equal)
BINARY_BOOL_RETURNER(greater_equal)

#define UNARY_FLOAT_RETURNER(name)                                      \
  static PyObject*                                                      \
  pydual_quaternion_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {           \
    dual_quaternion q = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};            \
    PyDualQuaternion_AsDualQuaternion(q, a);                                    \
    return PyFloat_FromDouble(dual_quaternion_##name(q));                    \
  }
UNARY_FLOAT_RETURNER(absolute)
UNARY_FLOAT_RETURNER(angle)

#define UNARY_DUAL_QUATERNION_RETURNER(name)                                 \
  static PyObject*                                                      \
  pydual_quaternion_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {           \
    dual_quaternion q = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};            \
    PyDualQuaternion_AsDualQuaternion(q, a);                                    \
    return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name(q));           \
  }
UNARY_DUAL_QUATERNION_RETURNER(negative)
UNARY_DUAL_QUATERNION_RETURNER(conjugate)
UNARY_DUAL_QUATERNION_RETURNER(inverse)
UNARY_DUAL_QUATERNION_RETURNER(sqrt)
UNARY_DUAL_QUATERNION_RETURNER(log)
UNARY_DUAL_QUATERNION_RETURNER(exp)
UNARY_DUAL_QUATERNION_RETURNER(normalized)
static PyObject*
pydual_quaternion_positive(PyObject* self, PyObject* NPY_UNUSED(b)) {
  Py_INCREF(self);
  return self;
}

#define QQ_BINARY_DUAL_QUATERNION_RETURNER(name)                             \
  static PyObject*                                                      \
  pydual_quaternion_##name(PyObject* a, PyObject* b) {                       \
    dual_quaternion p = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};            \
    dual_quaternion q = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};            \
    PyDualQuaternion_AsDualQuaternion(p, a);                                    \
    PyDualQuaternion_AsDualQuaternion(q, b);                                    \
    return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name(p,q));         \
  }
/* QQ_BINARY_DUAL_QUATERNION_RETURNER(add) */
/* QQ_BINARY_DUAL_QUATERNION_RETURNER(subtract) */
QQ_BINARY_DUAL_QUATERNION_RETURNER(copysign)

#define QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER_FULL(fake_name, name)       \
  static PyObject*                                                      \
  pydual_quaternion_##fake_name##_array_operator(PyObject* a, PyObject* b) { \
    /* fprintf (stderr, "\nfile %s, line %d, pydual_quaternion_%s_array_operator(PyObject* a, PyObject* b).\n", __FILE__, __LINE__, #fake_name); */ \
    PyArrayObject *in_array = (PyArrayObject*) b;                       \
    PyObject      *out_array;                                           \
    NpyIter *in_iter;                                                   \
    NpyIter *out_iter;                                                  \
    NpyIter_IterNextFunc *in_iternext;                                  \
    NpyIter_IterNextFunc *out_iternext;                                 \
    dual_quaternion p = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};       \
    dual_quaternion ** out_dataptr;                                          \
    PyDualQuaternion_AsDualQuaternion(p, a);                                    \
    out_array = PyArray_NewLikeArray(in_array, NPY_ANYORDER, dual_quaternion_descr, 0); \
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
    out_dataptr = (dual_quaternion **) NpyIter_GetDataPtrArray(out_iter);    \
    if(PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*) b), dual_quaternion_descr)) { \
      dual_quaternion ** in_dataptr = (dual_quaternion **) NpyIter_GetDataPtrArray(in_iter); \
      do {                                                              \
        **out_dataptr = dual_quaternion_##name(p, **in_dataptr);             \
      } while(in_iternext(in_iter) && out_iternext(out_iter));          \
    } else if(PyArray_ISFLOAT((PyArrayObject*) b)) {                    \
      double ** in_dataptr = (double **) NpyIter_GetDataPtrArray(in_iter); \
      do {                                                              \
        **out_dataptr = dual_quaternion_##name##_scalar(p, **in_dataptr);    \
      } while(in_iternext(in_iter) && out_iternext(out_iter));          \
    } else if(PyArray_ISINTEGER((PyArrayObject*) b)) {                  \
      int ** in_dataptr = (int **) NpyIter_GetDataPtrArray(in_iter);    \
      do {                                                              \
        **out_dataptr = dual_quaternion_##name##_scalar(p, **in_dataptr);    \
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
  pydual_quaternion_##fake_name(PyObject* a, PyObject* b) {                  \
    /* PyObject *a_type, *a_repr, *b_type, *b_repr, *a_repr2, *b_repr2;    \ */ \
    /* char* a_char, b_char, a_char2, b_char2;                             \ */ \
    npy_int64 val64;                                                    \
    npy_int32 val32;                                                    \
    dual_quaternion p = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};       \
    if(PyArray_Check(b)) { return pydual_quaternion_##fake_name##_array_operator(a, b); } \
    if(PyFloat_Check(a) && PyDualQuaternion_Check(b)) {                     \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_scalar_##name(PyFloat_AsDouble(a), ((PyDualQuaternion*)b)->obval)); \
    }                                                                   \
    if(PyInt_Check(a) && PyDualQuaternion_Check(b)) {                       \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_scalar_##name(PyInt_AsLong(a), ((PyDualQuaternion*)b)->obval)); \
    }                                                                   \
    PyDualQuaternion_AsDualQuaternion(p, a);                                    \
    if(PyDualQuaternion_Check(b)) {                                         \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name(p,((PyDualQuaternion*)b)->obval)); \
    } else if(PyFloat_Check(b)) {                                       \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name##_scalar(p,PyFloat_AsDouble(b))); \
    } else if(PyInt_Check(b)) {                                         \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name##_scalar(p,PyInt_AsLong(b))); \
    } else if(PyObject_TypeCheck(b, &PyInt64ArrType_Type)) {            \
      PyArray_ScalarAsCtype(b, &val64);                                 \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name##_scalar(p, val64)); \
    } else if(PyObject_TypeCheck(b, &PyInt32ArrType_Type)) {            \
      PyArray_ScalarAsCtype(b, &val32);                                 \
      return PyDualQuaternion_FromDualQuaternion(dual_quaternion_##name##_scalar(p, val32)); \
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
    /* fprintf (stderr, "\nfile %s, line %d, pydual_quaternion_%s(PyObject* a, PyObject* b).\n", __FILE__, __LINE__, #fake_name); \ */ \
    /* fprintf (stderr, "\na: '%s'\tb: '%s'", a_char, b_char);             \ */ \
    /* fprintf (stderr, "\na: '%s'\tb: '%s'", a_char2, b_char2);           \ */ \
    /* Py_DECREF(a_type);                                                  \ */ \
    /* Py_DECREF(a_repr);                                                  \ */ \
    /* Py_DECREF(b_type);                                                  \ */ \
    /* Py_DECREF(b_repr);                                                  \ */ \
    /* Py_DECREF(a_repr2);                                                 \ */ \
    /* Py_DECREF(b_repr2);                                                 \ */ \
    PyErr_SetString(PyExc_TypeError, "Binary operation involving dual_quaternion and \\neither float nor dual_quaternion."); \
    return NULL;                                                        \
  }
#define QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER(name) QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER_FULL(name, name)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER(add)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER(subtract)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER(multiply)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER(divide)
/* QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER_FULL(true_divide, divide) */
/* QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER_FULL(floor_divide, divide) */
QQ_QS_SQ_BINARY_DUAL_QUATERNION_RETURNER(power)

#define QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE_FULL(fake_name, name)        \
  static PyObject*                                                      \
  pydual_quaternion_inplace_##fake_name(PyObject* a, PyObject* b) {          \
    dual_quaternion* p = {0};                                                \
    /* fprintf (stderr, "file %s, line %d, pydual_quaternion_inplace_"#fake_name"(PyObject* a, PyObject* b).\n", __FILE__, __LINE__); \ */ \
    if(PyFloat_Check(a) || PyInt_Check(a)) {                            \
      PyErr_SetString(PyExc_TypeError, "Cannot in-place "#fake_name" a scalar by a dual_quaternion; should be handled by python."); \
      return NULL;                                                      \
    }                                                                   \
    PyDualQuaternion_AsDualQuaternionPointer(p, a);                             \
    if(PyDualQuaternion_Check(b)) {                                         \
      dual_quaternion_inplace_##name(p,((PyDualQuaternion*)b)->obval);           \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyFloat_Check(b)) {                                       \
      dual_quaternion_inplace_##name##_scalar(p,PyFloat_AsDouble(b));        \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyInt_Check(b)) {                                         \
      dual_quaternion_inplace_##name##_scalar(p,PyInt_AsLong(b));            \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    }                                                                   \
    PyErr_SetString(PyExc_TypeError, "Binary in-place operation involving dual_quaternion and neither float nor dual_quaternion."); \
    return NULL;                                                        \
  }
#define QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE(name) QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE_FULL(name, name)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE(add)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE(subtract)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE(multiply)
QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE(divide)
/* QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE_FULL(true_divide, divide) */
/* QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE_FULL(floor_divide, divide) */
QQ_QS_SQ_BINARY_DUAL_QUATERNION_INPLACE(power)

static PyObject*
pydual_quaternion_norm(PyObject* a, PyObject* NPY_UNUSED(b)) {
    dual_quaternion q = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    PyDualQuaternion_AsDualQuaternion(q, a);
    return dual_quaternion_norm(q);
}

static PyObject *
pydual_quaternion__reduce(PyDualQuaternion* self)
{
  /* printf("\n\n\nI'm trying, most of all!\n\n\n"); */
  return Py_BuildValue("O(OOOOOOOO)", Py_TYPE(self),
                       PyFloat_FromDouble(self->obval.w), PyFloat_FromDouble(self->obval.x),
                       PyFloat_FromDouble(self->obval.y), PyFloat_FromDouble(self->obval.z),
                       PyFloat_FromDouble(self->obval.er), PyFloat_FromDouble(self->obval.ei),
                       PyFloat_FromDouble(self->obval.ej), PyFloat_FromDouble(self->obval.ek));

}

static PyObject *
pydual_quaternion_getstate(PyDualQuaternion* self, PyObject* args)
{
  /* printf("\n\n\nI'm Trying, OKAY?\n\n\n"); */
  if (!PyArg_ParseTuple(args, ":getstate"))
    return NULL;
  return Py_BuildValue("OOOOOOOO",
                       PyFloat_FromDouble(self->obval.w), PyFloat_FromDouble(self->obval.x),
                       PyFloat_FromDouble(self->obval.y), PyFloat_FromDouble(self->obval.z),
                       PyFloat_FromDouble(self->obval.er), PyFloat_FromDouble(self->obval.ei),
                       PyFloat_FromDouble(self->obval.ej), PyFloat_FromDouble(self->obval.ek));
}

static PyObject *
pydual_quaternion_setstate(PyDualQuaternion* self, PyObject* args)
{
  /* printf("\n\n\nI'm Trying, TOO!\n\n\n"); */
  dual_quaternion* q;
  q = &(self->obval);

  if (!PyArg_ParseTuple(args, "dddddddd:setstate",
      &q->w, &q->x, &q->y, &q->z, &q->er, &q->ei, &q->ej, &q->ek)) {
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}


// This is an array of methods (member functions) that will be
// available to use on the dual_quaternion objects in python.  This is
// packaged up here, and will be used in the `tp_methods` field when
// definining the PyDualQuaternion_Type below.
PyMethodDef pydual_quaternion_methods[] = {
  // Unary bool returners
  {"nonzero", pydual_quaternion_nonzero, METH_NOARGS,
   "True if the dual_quaternion has all zero components"},
  {"isnan", pydual_quaternion_isnan, METH_NOARGS,
   "True if the dual_quaternion has any NAN components"},
  {"isinf", pydual_quaternion_isinf, METH_NOARGS,
   "True if the dual_quaternion has any INF components"},
  {"isfinite", pydual_quaternion_isfinite, METH_NOARGS,
   "True if the dual_quaternion has all finite components"},

  // Binary bool returners
  {"equal", pydual_quaternion_equal, METH_O,
   "True if the dual_quaternions are PRECISELY equal"},
  {"not_equal", pydual_quaternion_not_equal, METH_O,
   "True if the dual_quaternions are not PRECISELY equal"},
  {"less", pydual_quaternion_less, METH_O,
   "Strict dictionary ordering"},
  {"greater", pydual_quaternion_greater, METH_O,
   "Strict dictionary ordering"},
  {"less_equal", pydual_quaternion_less_equal, METH_O,
   "Dictionary ordering"},
  {"greater_equal", pydual_quaternion_greater_equal, METH_O,
   "Dictionary ordering"},

  // Unary float returners
  {"absolute", pydual_quaternion_absolute, METH_NOARGS,
   "Absolute value of dual_quaternion"},
  {"abs", pydual_quaternion_absolute, METH_NOARGS,
   "Absolute value (Euclidean norm) of dual_quaternion"},
  {"norm", pydual_quaternion_norm, METH_NOARGS,
   "Cayley norm (square of the absolute value) of dual_quaternion"},
  {"angle", pydual_quaternion_angle, METH_NOARGS,
   "Angle through which rotor rotates"},

  // Unary dual_quaternion returners
  // {"negative", pydual_quaternion_negative, METH_NOARGS,
  //  "Return the negated dual_quaternion"},
  // {"positive", pydual_quaternion_positive, METH_NOARGS,
  //  "Return the dual_quaternion itself"},
  {"conjugate", pydual_quaternion_conjugate, METH_NOARGS,
   "Return the complex conjugate of the dual_quaternion"},
  {"conj", pydual_quaternion_conjugate, METH_NOARGS,
   "Return the complex conjugate of the dual_quaternion"},
  {"inverse", pydual_quaternion_inverse, METH_NOARGS,
   "Return the inverse of the dual_quaternion"},
  {"sqrt", pydual_quaternion_sqrt, METH_NOARGS,
   "Return the square-root of the dual_quaternion"},
  {"log", pydual_quaternion_log, METH_NOARGS,
   "Return the logarithm (base e) of the dual_quaternion"},
  {"exp", pydual_quaternion_exp, METH_NOARGS,
   "Return the exponential of the dual_quaternion (e**q)"},
  {"normalized", pydual_quaternion_normalized, METH_NOARGS,
   "Return a normalized copy of the dual_quaternion"},

  // DualQuaternion-dual_quaternion binary dual_quaternion returners
  // {"add", pydual_quaternion_add, METH_O,
  //  "Componentwise addition"},
  // {"subtract", pydual_quaternion_subtract, METH_O,
  //  "Componentwise subtraction"},
  {"copysign", pydual_quaternion_copysign, METH_O,
   "Componentwise copysign"},

  // DualQuaternion-dual_quaternion or dual_quaternion-scalar binary dual_quaternion returners
  // {"multiply", pydual_quaternion_multiply, METH_O,
  //  "Standard (geometric) dual_quaternion product"},
  // {"divide", pydual_quaternion_divide, METH_O,
  //  "Standard (geometric) dual_quaternion division"},
  // {"power", pydual_quaternion_power, METH_O,
  //  "q.power(p) = (q.log() * p).exp()"},

  {"__reduce__", (PyCFunction)pydual_quaternion__reduce, METH_NOARGS,
   "Return state information for pickling."},
  {"__getstate__", (PyCFunction)pydual_quaternion_getstate, METH_VARARGS,
   "Return state information for pickling."},
  {"__setstate__", (PyCFunction)pydual_quaternion_setstate, METH_VARARGS,
   "Reconstruct state information from pickle."},

  {NULL, NULL, 0, NULL}
};

static PyObject* pydual_quaternion_num_power(PyObject* a, PyObject* b, PyObject *c) { (void) c; return pydual_quaternion_power(a,b); }
static PyObject* pydual_quaternion_num_inplace_power(PyObject* a, PyObject* b, PyObject *c) { (void) c; return pydual_quaternion_inplace_power(a,b); }
static PyObject* pydual_quaternion_num_negative(PyObject* a) { return pydual_quaternion_negative(a,NULL); }
static PyObject* pydual_quaternion_num_positive(PyObject* a) { return pydual_quaternion_positive(a,NULL); }
static PyObject* pydual_quaternion_num_absolute(PyObject* a) { return pydual_quaternion_absolute(a,NULL); }
static PyObject* pydual_quaternion_num_inverse(PyObject* a) { return pydual_quaternion_inverse(a,NULL); }
static int pydual_quaternion_num_nonzero(PyObject* a) {
  dual_quaternion q = ((PyDualQuaternion*)a)->obval;
  return dual_quaternion_nonzero(q);
}

static PyNumberMethods pydual_quaternion_as_number = {
  pydual_quaternion_add,               // nb_add
  pydual_quaternion_subtract,          // nb_subtract
  pydual_quaternion_multiply,          // nb_multiply
  #if PY_MAJOR_VERSION < 3
  pydual_quaternion_divide,            // nb_divide
  #endif
  0,                              // nb_remainder
  0,                              // nb_divmod
  pydual_quaternion_num_power,         // nb_power
  pydual_quaternion_num_negative,      // nb_negative
  pydual_quaternion_num_positive,      // nb_positive
  pydual_quaternion_num_absolute,      // nb_absolute
  pydual_quaternion_num_nonzero,       // nb_nonzero
  pydual_quaternion_num_inverse,       // nb_invert
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
  pydual_quaternion_inplace_add,       // nb_inplace_add
  pydual_quaternion_inplace_subtract,  // nb_inplace_subtract
  pydual_quaternion_inplace_multiply,  // nb_inplace_multiply
  #if PY_MAJOR_VERSION < 3
  pydual_quaternion_inplace_divide,    // nb_inplace_divide
  #endif
  0,                              // nb_inplace_remainder
  pydual_quaternion_num_inplace_power, // nb_inplace_power
  0,                              // nb_inplace_lshift
  0,                              // nb_inplace_rshift
  0,                              // nb_inplace_and
  0,                              // nb_inplace_xor
  0,                              // nb_inplace_or
  pydual_quaternion_divide,            // nb_floor_divide
  pydual_quaternion_divide,            // nb_true_divide
  pydual_quaternion_inplace_divide,    // nb_inplace_floor_divide
  pydual_quaternion_inplace_divide,    // nb_inplace_true_divide
  0,                              // nb_index
  #if PY_MAJOR_VERSION >= 3
  #if PY_MINOR_VERSION >= 5
  0,                              // nb_matrix_multiply
  0,                              //  nb_inplace_matrix_multiply
  #endif
  #endif
};


// This is an array of members (member data) that will be available to
// use on the dual_quaternion objects in python.  This is packaged up here,
// and will be used in the `tp_members` field when definining the
// PyDualQuaternion_Type below.
PyMemberDef pydual_quaternion_members[] = {
  {"real", T_DOUBLE, offsetof(PyDualQuaternion, obval.w), 0,
   "The real component of the dual_quaternion"},
  {"w", T_DOUBLE, offsetof(PyDualQuaternion, obval.w), 0,
   "The real component of the dual_quaternion"},
  {"x", T_DOUBLE, offsetof(PyDualQuaternion, obval.x), 0,
   "The first imaginary component of the dual_quaternion"},
  {"y", T_DOUBLE, offsetof(PyDualQuaternion, obval.y), 0,
   "The second imaginary component of the dual_quaternion"},
  {"z", T_DOUBLE, offsetof(PyDualQuaternion, obval.z), 0,
   "The third imaginary component of the dual_quaternion"},
  {"er", T_DOUBLE, offsetof(PyDualQuaternion, obval.er), 0,
   "TODO: description of er here"},
  {"ei", T_DOUBLE, offsetof(PyDualQuaternion, obval.ei), 0,
   "TODO: description of ei here"},
  {"ej", T_DOUBLE, offsetof(PyDualQuaternion, obval.ej), 0,
   "TODO: description of ej here"},
  {"ek", T_DOUBLE, offsetof(PyDualQuaternion, obval.ek), 0,
   "TODO: description of ek here"},
  {NULL, 0, 0, 0, NULL}
  //TODO: what to name the new variables?
};

//
// The dual_quaternion can be conveniently separated into two complex
// numbers, which we call 'part a' and 'part b'.  These are useful in
// writing Wigner's D matrices directly in terms of dual_quaternions.  This
// is essentially the column-vector presentation of spinors.
static PyObject *
pydual_quaternion_get_part_a(PyObject *self, void *NPY_UNUSED(closure))
{
  return (PyObject*) PyComplex_FromDoubles(((PyDualQuaternion *)self)->obval.w, ((PyDualQuaternion *)self)->obval.z);
}
static PyObject *
pydual_quaternion_get_part_b(PyObject *self, void *NPY_UNUSED(closure))
{
  return (PyObject*) PyComplex_FromDoubles(((PyDualQuaternion *)self)->obval.y, ((PyDualQuaternion *)self)->obval.x);
}

// This will be defined as a member function on the dual_quaternion
// objects, so that calling "vec" will return a numpy array
// with the last three components of the dual_quaternion.
static PyObject *
pydual_quaternion_get_vec(PyObject *self, void *NPY_UNUSED(closure))
{
  dual_quaternion *q = &((PyDualQuaternion *)self)->obval;
  int nd = 1;
  npy_intp dims[1] = { 7 };
  int typenum = NPY_DOUBLE;
  PyObject* components = PyArray_SimpleNewFromData(nd, dims, typenum, &(q->x));
  Py_INCREF(self);
  PyArray_SetBaseObject((PyArrayObject*)components, self);
  return components;
}

// This will be defined as a member function on the dual_quaternion
// objects, so that calling `q.vec = [1,2,3]`, for example,
// will set the vector components appropriately.
static int
pydual_quaternion_set_vec(PyObject *self, PyObject *value, void *NPY_UNUSED(closure))
{
  PyObject *element;
  dual_quaternion *q = &((PyDualQuaternion *)self)->obval;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot set dual_quaternion to empty value");
    return -1;
  }
  if (! (PySequence_Check(value) && PySequence_Size(value)==7) ) {
    PyErr_SetString(PyExc_TypeError,
                    "A dual_quaternion's vector components must be set to something of length 7");
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
  element = PySequence_GetItem(value, 3);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->er = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 4);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->ei = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 5);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->ej = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 6);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->ek = PyFloat_AsDouble(element);
  Py_DECREF(element);
  return 0;
}

// This will be defined as a member function on the dual_quaternion
// objects, so that calling "components" will return a numpy array
// with the components of the dual_quaternion.
static PyObject *
pydual_quaternion_get_components(PyObject *self, void *NPY_UNUSED(closure))
{
  dual_quaternion *q = &((PyDualQuaternion *)self)->obval;
  int nd = 1;
  npy_intp dims[1] = { 8 };
  int typenum = NPY_DOUBLE;
  PyObject* components = PyArray_SimpleNewFromData(nd, dims, typenum, &(q->w));
  Py_INCREF(self);
  PyArray_SetBaseObject((PyArrayObject*)components, self);
  return components;
}

// This will be defined as a member function on the dual_quaternion
// objects, so that calling `q.components = [1,2,3,4]`, for example,
// will set the components appropriately.
static int
pydual_quaternion_set_components(PyObject *self, PyObject *value, void *NPY_UNUSED(closure)){
  PyObject *element;
  dual_quaternion *q = &((PyDualQuaternion *)self)->obval;
  if (value == NULL) {
    PyErr_SetString(PyExc_ValueError, "Cannot set dual_quaternion to empty value");
    return -1;
  }
  if (! (PySequence_Check(value) && PySequence_Size(value)==8) ) {
    PyErr_SetString(PyExc_TypeError,
                    "A dual_quaternion's components must be set to something of length 8");
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
  element = PySequence_GetItem(value, 4);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->er = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 5);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->ei = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 6);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->ej = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 7);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->ek = PyFloat_AsDouble(element);
  Py_DECREF(element);
  return 0;
}

// This collects the methods for getting and setting elements of the
// dual_quaternion.  This is packaged up here, and will be used in the
// `tp_getset` field when defining the PyDualQuaternion_Type
// below.
PyGetSetDef pydual_quaternion_getset[] = {
  {"a", pydual_quaternion_get_part_a, NULL,
   "The complex number (w+i*z)", NULL},
  {"b", pydual_quaternion_get_part_b, NULL,
   "The complex number (y+i*x)", NULL},
  {"imag", pydual_quaternion_get_vec, pydual_quaternion_set_vec,
   "The vector part (x,y,z,er,ei,ej,ek) of the dual_quaternion as a numpy array", NULL},
  {"vec", pydual_quaternion_get_vec, pydual_quaternion_set_vec,
   "The vector part (x,y,z,er,ei,ej,ek) of the dual_quaternion as a numpy array", NULL},
  {"components", pydual_quaternion_get_components, pydual_quaternion_set_components,
   "The components (w,x,y,z,er,ei,ej,ek) of the dual_quaternion as a numpy array", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};



static PyObject*
pydual_quaternion_richcompare(PyObject* a, PyObject* b, int op)
{
  dual_quaternion x = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  dual_quaternion y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int result = 0;
  PyDualQuaternion_AsDualQuaternion(x,a);
  PyDualQuaternion_AsDualQuaternion(y,b);
  #define COMPARISONOP(py,op) case py: result = dual_quaternion_##op(x,y); break;
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
pydual_quaternion_hash(PyObject *o)
{
  dual_quaternion q = ((PyDualQuaternion *)o)->obval;
  long value = 0x456789;
  value = (10000004 * value) ^ _Py_HashDouble(q.w);
  value = (10000004 * value) ^ _Py_HashDouble(q.x);
  value = (10000004 * value) ^ _Py_HashDouble(q.y);
  value = (10000004 * value) ^ _Py_HashDouble(q.z);
  value = (10000004 * value) ^ _Py_HashDouble(q.er);
  value = (10000004 * value) ^ _Py_HashDouble(q.ei);
  value = (10000004 * value) ^ _Py_HashDouble(q.ej);
  value = (10000004 * value) ^ _Py_HashDouble(q.ek);
  if (value == -1)
    value = -2;
  return value;
}

static PyObject *
pydual_quaternion_repr(PyObject *o)
{
  char str[128];
  dual_quaternion q = ((PyDualQuaternion *)o)->obval;
  sprintf(str, "dual_quaternion(%.15g, %.15g, %.15g, %.15g, %.15g, %.15g, %.15g, %.15g)",
          q.w, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek);
  return PyUString_FromString(str);
}

static PyObject *
pydual_quaternion_str(PyObject *o)
{
  char str[128];
  dual_quaternion q = ((PyDualQuaternion *)o)->obval;
  sprintf(str, "dual_quaternion(%.15g, %.15g, %.15g, %.15g, %.15g, %.15g, %.15g, %.15g)",
          q.w, q.x, q.y, q.z, q.er, q.ei, q.ej, q.ek);
  return PyUString_FromString(str);
}


// This establishes the dual_quaternion as a python object (not yet a numpy
// scalar type).  The name may be a little counterintuitive; the idea
// is that this will be a type that can be used as an array dtype.
// Note that many of the slots below will be filled later, after the
// corresponding functions are defined.
static PyTypeObject PyDualQuaternion_Type = {
#if PY_MAJOR_VERSION >= 3
  PyVarObject_HEAD_INIT(NULL, 0)
#else
  PyObject_HEAD_INIT(NULL)
  0,                                          // ob_size
#endif
  "dual_quaternion",                               // tp_name
  sizeof(PyDualQuaternion),                       // tp_basicsize
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
  pydual_quaternion_repr,                          // tp_repr
  &pydual_quaternion_as_number,                    // tp_as_number
  0,                                          // tp_as_sequence
  0,                                          // tp_as_mapping
  pydual_quaternion_hash,                          // tp_hash
  0,                                          // tp_call
  pydual_quaternion_str,                           // tp_str
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
  pydual_quaternion_richcompare,                   // tp_richcompare
  0,                                          // tp_weaklistoffset
  0,                                          // tp_iter
  0,                                          // tp_iternext
  pydual_quaternion_methods,                       // tp_methods
  pydual_quaternion_members,                       // tp_members
  pydual_quaternion_getset,                        // tp_getset
  0,                                          // tp_base; will be reset to &PyGenericArrType_Type after numpy import
  0,                                          // tp_dict
  0,                                          // tp_descr_get
  0,                                          // tp_descr_set
  0,                                          // tp_dictoffset
  pydual_quaternion_init,                          // tp_init
  0,                                          // tp_alloc
  pydual_quaternion_new,                           // tp_new
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
#if PY_VERSION_HEX >= 0x030400a1
  0,                                          // tp_finalize
#endif
};

// Functions implementing internal features. Not all of these function
// pointers must be defined for a given type. The required members are
// nonzero, copyswap, copyswapn, setitem, getitem, and cast.
static PyArray_ArrFuncs _PyDualQuaternion_ArrFuncs;

static npy_bool
DUAL_QUATERNION_nonzero (char *ip, PyArrayObject *ap)
{
  dual_quaternion q;
  dual_quaternion zero = {0,0,0,0,0,0,0,0};
  if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
    q = *(dual_quaternion *)ip;
  }
  else {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswap(&q.w, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.x, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.y, ip+16, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.z, ip+24, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.er, ip+32, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.ei, ip+40, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.ej, ip+48, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.ek, ip+56, !PyArray_ISNOTSWAPPED(ap), NULL);
    Py_DECREF(descr);
  }
  return (npy_bool) !dual_quaternion_equal(q, zero);
}

static void
DUAL_QUATERNION_copyswap(dual_quaternion *dst, dual_quaternion *src,
                    int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(dst, sizeof(double), src, sizeof(double), 8, swap, NULL);
  Py_DECREF(descr);
}

static void
DUAL_QUATERNION_copyswapn(dual_quaternion *dst, npy_intp dstride,
                     dual_quaternion *src, npy_intp sstride,
                     npy_intp n, int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(&dst->w, dstride, &src->w, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->x, dstride, &src->x, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->y, dstride, &src->y, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->z, dstride, &src->z, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->er, dstride, &src->er, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->ei, dstride, &src->ei, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->ej, dstride, &src->ej, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->ek, dstride, &src->ek, sstride, n, swap, NULL);
  Py_DECREF(descr);
}

static int DUAL_QUATERNION_setitem(PyObject* item, dual_quaternion* qp, void* NPY_UNUSED(ap))
{
  PyObject *element;
  if(PyDualQuaternion_Check(item)) {
    memcpy(qp,&(((PyDualQuaternion *)item)->obval),sizeof(dual_quaternion));
  } else if(PySequence_Check(item) && PySequence_Length(item)==8) {
    element = PySequence_GetItem(item, 0);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->w = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 1);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->x = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 2);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->y = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 3);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->z = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 4);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->er = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 5);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->ei = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 6);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->ej = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 7);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->ek = PyFloat_AsDouble(element);
    Py_DECREF(element);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown input to DUAL_QUATERNION_setitem");
    return -1;
  }
  return 0;
}

// When a numpy array of dtype=dual_quaternion is indexed, this function is
// called, returning a new dual_quaternion object with a copy of the
// data... sometimes...
static PyObject *
DUAL_QUATERNION_getitem(void* data, void* NPY_UNUSED(arr))
{
  dual_quaternion q;
  memcpy(&q,data,sizeof(dual_quaternion));
  return PyDualQuaternion_FromDualQuaternion(q);
}

static int
DUAL_QUATERNION_compare(dual_quaternion *pa, dual_quaternion *pb, PyArrayObject *NPY_UNUSED(ap))
{
  dual_quaternion a = *pa, b = *pb;
  npy_bool anan, bnan;
  int ret;

  anan = dual_quaternion_isnan(a);
  bnan = dual_quaternion_isnan(b);

  if (anan) {
    ret = bnan ? 0 : -1;
  } else if (bnan) {
    ret = 1;
  } else if(dual_quaternion_less(a, b)) {
    ret = -1;
  } else if(dual_quaternion_less(b, a)) {
    ret = 1;
  } else {
    ret = 0;
  }

  return ret;
}

static int
DUAL_QUATERNION_argmax(dual_quaternion *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
  npy_intp i;
  dual_quaternion mp = *ip;

  *max_ind = 0;

  if (dual_quaternion_isnan(mp)) {
    // nan encountered; it's maximal
    return 0;
  }

  for (i = 1; i < n; i++) {
    ip++;
    //Propagate nans, similarly as max() and min()
    if (!(dual_quaternion_less_equal(*ip, mp))) {  // negated, for correct nan handling
      mp = *ip;
      *max_ind = i;
      if (dual_quaternion_isnan(mp)) {
        // nan encountered, it's maximal
        break;
      }
    }
  }
  return 0;
}

static void
DUAL_QUATERNION_fillwithscalar(dual_quaternion *buffer, npy_intp length, dual_quaternion *value, void *NPY_UNUSED(ignored))
{
  npy_intp i;
  dual_quaternion val = *value;

  for (i = 0; i < length; ++i) {
    buffer[i] = val;
  }
}

// This is a macro (followed by applications of the macro) that cast
// the input types to standard dual_quaternions with only a nonzero scalar
// part.
#define MAKE_T_TO_DUAL_QUATERNION(TYPE, type)                                \
  static void                                                           \
  TYPE ## _to_dual_quaternion(type *ip, dual_quaternion *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      op->w = (double)(*ip++);                                          \
      op->x = 0;                                                        \
      op->y = 0;                                                        \
      op->z = 0;                                                        \
      op->er = 0;                                                        \
      op->ei = 0;                                                        \
      op->ej = 0;                                                        \
      op->ek = 0;                                                       \
      op++;                                                             \
    }                                                                   \
  }
MAKE_T_TO_DUAL_QUATERNION(FLOAT, npy_float);
MAKE_T_TO_DUAL_QUATERNION(DOUBLE, npy_double);
MAKE_T_TO_DUAL_QUATERNION(LONGDOUBLE, npy_longdouble);
MAKE_T_TO_DUAL_QUATERNION(BOOL, npy_bool);
MAKE_T_TO_DUAL_QUATERNION(BYTE, npy_byte);
MAKE_T_TO_DUAL_QUATERNION(UBYTE, npy_ubyte);
MAKE_T_TO_DUAL_QUATERNION(SHORT, npy_short);
MAKE_T_TO_DUAL_QUATERNION(USHORT, npy_ushort);
MAKE_T_TO_DUAL_QUATERNION(INT, npy_int);
MAKE_T_TO_DUAL_QUATERNION(UINT, npy_uint);
MAKE_T_TO_DUAL_QUATERNION(LONG, npy_long);
MAKE_T_TO_DUAL_QUATERNION(ULONG, npy_ulong);
MAKE_T_TO_DUAL_QUATERNION(LONGLONG, npy_longlong);
MAKE_T_TO_DUAL_QUATERNION(ULONGLONG, npy_ulonglong);

// This is a macro (followed by applications of the macro) that cast
// the input complex types to standard dual_quaternions with only the first
// two components nonzero.  This doesn't make a whole lot of sense to
// me, and may be removed in the future.
#define MAKE_CT_TO_DUAL_QUATERNION(TYPE, type)                               \
  static void                                                           \
  TYPE ## _to_dual_quaternion(type *ip, dual_quaternion *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      op->w = (double)(*ip++);                                          \
      op->x = (double)(*ip++);                                          \
      op->y = 0;                                                        \
      op->z = 0;                                                        \
      op->er = 0;                                                       \
      op->ei = 0;                                                       \
      op->ej = 0;                                                       \
      op->ek = 0;                                                       \
    }                                                                   \
  }
MAKE_CT_TO_DUAL_QUATERNION(CFLOAT, npy_float);
MAKE_CT_TO_DUAL_QUATERNION(CDOUBLE, npy_double);
MAKE_CT_TO_DUAL_QUATERNION(CLONGDOUBLE, npy_longdouble);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
  PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
  PyArray_RegisterCastFunc(descr, destType, castfunc);
  PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
  Py_DECREF(descr);
}


// This is a macro that will be used to define the various basic unary
// dual_quaternion functions, so that they can be applied quickly to a
// numpy array of dual_quaternions.
#define UNARY_GEN_UFUNC(ufunc_name, func_name, ret_type)        \
  static void                                                           \
  dual_quaternion_##ufunc_name##_ufunc(char** args, npy_intp* dimensions,    \
                                  npy_intp* steps, void* NPY_UNUSED(data)) { \
    /* fprintf (stderr, "file %s, line %d, dual_quaternion_%s_ufunc.\n", __FILE__, __LINE__, #ufunc_name); */ \
    char *ip1 = args[0], *op1 = args[1];                                \
    npy_intp is1 = steps[0], os1 = steps[1];                            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){                     \
      const dual_quaternion in1 = *(dual_quaternion *)ip1;                        \
      *((ret_type *)op1) = dual_quaternion_##func_name(in1);};}
#define UNARY_UFUNC(name, ret_type) \
  UNARY_GEN_UFUNC(name, name, ret_type)
// And these all do the work mentioned above, using the macro
UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
//UNARY_UFUNC(norm, npy_double)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(angle, npy_double)
UNARY_UFUNC(sqrt, dual_quaternion)
UNARY_UFUNC(log, dual_quaternion)
UNARY_UFUNC(exp, dual_quaternion)
UNARY_UFUNC(negative, dual_quaternion)
UNARY_UFUNC(conjugate, dual_quaternion)
UNARY_GEN_UFUNC(invert, inverse, dual_quaternion)
UNARY_UFUNC(normalized, dual_quaternion)


// This is a macro that will be used to define the various basic binary
// dual_quaternion functions, so that they can be applied quickly to a
// numpy array of dual_quaternions.
#define BINARY_GEN_UFUNC(ufunc_name, func_name, arg_type1, arg_type2, ret_type) \
  static void                                                           \
  dual_quaternion_##ufunc_name##_ufunc(char** args, npy_intp* dimensions,    \
                                  npy_intp* steps, void* NPY_UNUSED(data)) { \
    /* fprintf (stderr, "file %s, line %d, dual_quaternion_%s_ufunc.\n", __FILE__, __LINE__, #ufunc_name); */ \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];                \
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {        \
      const arg_type1 in1 = *(arg_type1 *)ip1;                          \
      const arg_type2 in2 = *(arg_type2 *)ip2;                          \
      *((ret_type *)op1) = dual_quaternion_##func_name(in1, in2);            \
    };                                                                  \
  };
// A couple special-case versions of the above
#define BINARY_UFUNC(name, ret_type)                    \
  BINARY_GEN_UFUNC(name, name, dual_quaternion, dual_quaternion, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)                             \
  BINARY_GEN_UFUNC(name##_scalar, name##_scalar, dual_quaternion, npy_double, ret_type) \
  BINARY_GEN_UFUNC(scalar_##name, scalar_##name, npy_double, dual_quaternion, ret_type)
// And these all do the work mentioned above, using the macros
BINARY_UFUNC(add, dual_quaternion)
BINARY_UFUNC(subtract, dual_quaternion)
BINARY_UFUNC(multiply, dual_quaternion)
BINARY_UFUNC(divide, dual_quaternion)
BINARY_GEN_UFUNC(true_divide, divide, dual_quaternion, dual_quaternion, dual_quaternion)
BINARY_GEN_UFUNC(floor_divide, divide, dual_quaternion, dual_quaternion, dual_quaternion)
BINARY_UFUNC(power, dual_quaternion)
BINARY_UFUNC(copysign, dual_quaternion)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)
BINARY_SCALAR_UFUNC(add, dual_quaternion)
BINARY_SCALAR_UFUNC(subtract, dual_quaternion)
BINARY_SCALAR_UFUNC(multiply, dual_quaternion)
BINARY_SCALAR_UFUNC(divide, dual_quaternion)
BINARY_GEN_UFUNC(true_divide_scalar, divide_scalar, dual_quaternion, npy_double, dual_quaternion)
BINARY_GEN_UFUNC(floor_divide_scalar, divide_scalar, dual_quaternion, npy_double, dual_quaternion)
BINARY_GEN_UFUNC(scalar_true_divide, scalar_divide, npy_double, dual_quaternion, dual_quaternion)
BINARY_GEN_UFUNC(scalar_floor_divide, scalar_divide, npy_double, dual_quaternion, dual_quaternion)
BINARY_SCALAR_UFUNC(power, dual_quaternion)


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_dual_quaternion",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

#define INITERROR return NULL

// This is the initialization function that does the setup
PyMODINIT_FUNC PyInit_numpy_dual_quaternion(void) {

#else

#define INITERROR return

// This is the initialization function that does the setup
PyMODINIT_FUNC initnumpy_dual_quaternion(void) {

#endif

  PyObject *module;
  PyObject *tmp_ufunc;
  int dual_quaternionNum;
  int arg_types[3];
  PyObject* numpy;
  PyObject* numpy_dict;

  // Initialize a (for now, empty) module
#if PY_MAJOR_VERSION >= 3
  module = PyModule_Create(&moduledef);
#else
  module = Py_InitModule("numpy_dual_quaternion", DualQuaternionMethods);
#endif

  if(module==NULL) {
    INITERROR;
  }

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

  // Register the dual_quaternion array base type.  Couldn't do this until
  // after we imported numpy (above)
  PyDualQuaternion_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&PyDualQuaternion_Type) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "Could not initialize PyDualQuaternion_Type.");
    INITERROR;
  }

  // The array functions, to be used below.  This InitArrFuncs
  // function is a convenient way to set all the fields to zero
  // initially, so we don't get undefined behavior.
  PyArray_InitArrFuncs(&_PyDualQuaternion_ArrFuncs);
  _PyDualQuaternion_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)DUAL_QUATERNION_nonzero;
  _PyDualQuaternion_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)DUAL_QUATERNION_copyswap;
  _PyDualQuaternion_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)DUAL_QUATERNION_copyswapn;
  _PyDualQuaternion_ArrFuncs.setitem = (PyArray_SetItemFunc*)DUAL_QUATERNION_setitem;
  _PyDualQuaternion_ArrFuncs.getitem = (PyArray_GetItemFunc*)DUAL_QUATERNION_getitem;
  _PyDualQuaternion_ArrFuncs.compare = (PyArray_CompareFunc*)DUAL_QUATERNION_compare;
  _PyDualQuaternion_ArrFuncs.argmax = (PyArray_ArgFunc*)DUAL_QUATERNION_argmax;
  _PyDualQuaternion_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)DUAL_QUATERNION_fillwithscalar;

  // The dual_quaternion array descr
  dual_quaternion_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  dual_quaternion_descr->typeobj = &PyDualQuaternion_Type;
  dual_quaternion_descr->kind = 'V';
  dual_quaternion_descr->type = 'q';
  dual_quaternion_descr->byteorder = '=';
  dual_quaternion_descr->flags = 0;
  dual_quaternion_descr->type_num = 0; // assigned at registration
  dual_quaternion_descr->elsize = 8*8;
  dual_quaternion_descr->alignment = 8;
  dual_quaternion_descr->subarray = NULL;
  dual_quaternion_descr->fields = NULL;
  dual_quaternion_descr->names = NULL;
  dual_quaternion_descr->f = &_PyDualQuaternion_ArrFuncs;
  dual_quaternion_descr->metadata = NULL;
  dual_quaternion_descr->c_metadata = NULL;

  Py_INCREF(&PyDualQuaternion_Type);
  dual_quaternionNum = PyArray_RegisterDataType(dual_quaternion_descr);

  if (dual_quaternionNum < 0) {
    INITERROR;
  }

  register_cast_function(NPY_BOOL, dual_quaternionNum, (PyArray_VectorUnaryFunc*)BOOL_to_dual_quaternion);
  register_cast_function(NPY_BYTE, dual_quaternionNum, (PyArray_VectorUnaryFunc*)BYTE_to_dual_quaternion);
  register_cast_function(NPY_UBYTE, dual_quaternionNum, (PyArray_VectorUnaryFunc*)UBYTE_to_dual_quaternion);
  register_cast_function(NPY_SHORT, dual_quaternionNum, (PyArray_VectorUnaryFunc*)SHORT_to_dual_quaternion);
  register_cast_function(NPY_USHORT, dual_quaternionNum, (PyArray_VectorUnaryFunc*)USHORT_to_dual_quaternion);
  register_cast_function(NPY_INT, dual_quaternionNum, (PyArray_VectorUnaryFunc*)INT_to_dual_quaternion);
  register_cast_function(NPY_UINT, dual_quaternionNum, (PyArray_VectorUnaryFunc*)UINT_to_dual_quaternion);
  register_cast_function(NPY_LONG, dual_quaternionNum, (PyArray_VectorUnaryFunc*)LONG_to_dual_quaternion);
  register_cast_function(NPY_ULONG, dual_quaternionNum, (PyArray_VectorUnaryFunc*)ULONG_to_dual_quaternion);
  register_cast_function(NPY_LONGLONG, dual_quaternionNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_dual_quaternion);
  register_cast_function(NPY_ULONGLONG, dual_quaternionNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_dual_quaternion);
  register_cast_function(NPY_FLOAT, dual_quaternionNum, (PyArray_VectorUnaryFunc*)FLOAT_to_dual_quaternion);
  register_cast_function(NPY_DOUBLE, dual_quaternionNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_dual_quaternion);
  register_cast_function(NPY_LONGDOUBLE, dual_quaternionNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_dual_quaternion);
  register_cast_function(NPY_CFLOAT, dual_quaternionNum, (PyArray_VectorUnaryFunc*)CFLOAT_to_dual_quaternion);
  register_cast_function(NPY_CDOUBLE, dual_quaternionNum, (PyArray_VectorUnaryFunc*)CDOUBLE_to_dual_quaternion);
  register_cast_function(NPY_CLONGDOUBLE, dual_quaternionNum, (PyArray_VectorUnaryFunc*)CLONGDOUBLE_to_dual_quaternion);

  // These macros will be used below
  #define REGISTER_UFUNC(name)                                          \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                dual_quaternion_descr->type_num, dual_quaternion_##name##_ufunc, arg_types, NULL)
  #define REGISTER_SCALAR_UFUNC(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                dual_quaternion_descr->type_num, dual_quaternion_scalar_##name##_ufunc, arg_types, NULL)
  #define REGISTER_UFUNC_SCALAR(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                dual_quaternion_descr->type_num, dual_quaternion_##name##_scalar_ufunc, arg_types, NULL)
  #define REGISTER_NEW_UFUNC_GENERAL(pyname, cname, nargin, nargout, doc) \
    tmp_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, nargin, nargout, \
                                        PyUFunc_None, #pyname, doc, 0); \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)tmp_ufunc,             \
                                dual_quaternion_descr->type_num, dual_quaternion_##cname##_ufunc, arg_types, NULL); \
    PyDict_SetItemString(numpy_dict, #pyname, tmp_ufunc);               \
    Py_DECREF(tmp_ufunc)
  #define REGISTER_NEW_UFUNC(name, nargin, nargout, doc)                \
    REGISTER_NEW_UFUNC_GENERAL(name, name, nargin, nargout, doc)

  // quat -> bool
  arg_types[0] = dual_quaternion_descr->type_num;
  arg_types[1] = NPY_BOOL;
  REGISTER_UFUNC(isnan);
  /* // Already works: REGISTER_UFUNC(nonzero); */
  REGISTER_UFUNC(isinf);
  REGISTER_UFUNC(isfinite);

  // quat -> double
  arg_types[0] = dual_quaternion_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  //REGISTER_NEW_UFUNC(norm, 1, 1,
  //                   "Return Cayley norm (square of the absolute value) of each dual_quaternion.\n");
  REGISTER_UFUNC(absolute);
  REGISTER_NEW_UFUNC_GENERAL(angle_of_rotor, angle, 1, 1,
                             "Return angle of rotation, assuming input is a unit rotor\n");

  // quat -> quat
  arg_types[0] = dual_quaternion_descr->type_num;
  arg_types[1] = dual_quaternion_descr->type_num;
  REGISTER_NEW_UFUNC_GENERAL(sqrt_of_rotor, sqrt, 1, 1,
                             "Return square-root of rotor.  Assumes input has unit norm.\n");
  REGISTER_UFUNC(log);
  REGISTER_UFUNC(exp);
  REGISTER_NEW_UFUNC(normalized, 1, 1,
                     "Normalize all dual_quaternions in this array\n");
  REGISTER_UFUNC(negative);
  REGISTER_UFUNC(conjugate);
  REGISTER_UFUNC(invert);

  // quat, quat -> bool
  arg_types[0] = dual_quaternion_descr->type_num;
  arg_types[1] = dual_quaternion_descr->type_num;
  arg_types[2] = NPY_BOOL;
  REGISTER_UFUNC(equal);
  REGISTER_UFUNC(not_equal);
  REGISTER_UFUNC(less);
  REGISTER_UFUNC(less_equal);

  // quat, quat -> quat
  arg_types[0] = dual_quaternion_descr->type_num;
  arg_types[1] = dual_quaternion_descr->type_num;
  arg_types[2] = dual_quaternion_descr->type_num;
  REGISTER_UFUNC(add);
  REGISTER_UFUNC(subtract);
  REGISTER_UFUNC(multiply);
  REGISTER_UFUNC(divide);
  REGISTER_UFUNC(true_divide);
  REGISTER_UFUNC(floor_divide);
  REGISTER_UFUNC(power);
  REGISTER_UFUNC(copysign);

  // double, quat -> quat
  arg_types[0] = NPY_DOUBLE;
  arg_types[1] = dual_quaternion_descr->type_num;
  arg_types[2] = dual_quaternion_descr->type_num;
  REGISTER_SCALAR_UFUNC(add);
  REGISTER_SCALAR_UFUNC(subtract);
  REGISTER_SCALAR_UFUNC(multiply);
  REGISTER_SCALAR_UFUNC(divide);
  REGISTER_SCALAR_UFUNC(true_divide);
  REGISTER_SCALAR_UFUNC(floor_divide);
  REGISTER_SCALAR_UFUNC(power);

  // quat, double -> quat
  arg_types[0] = dual_quaternion_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  arg_types[2] = dual_quaternion_descr->type_num;
  REGISTER_UFUNC_SCALAR(add);
  REGISTER_UFUNC_SCALAR(subtract);
  REGISTER_UFUNC_SCALAR(multiply);
  REGISTER_UFUNC_SCALAR(divide);
  REGISTER_UFUNC_SCALAR(true_divide);
  REGISTER_UFUNC_SCALAR(floor_divide);
  REGISTER_UFUNC_SCALAR(power);


  // Add the constant `_DUAL_QUATERNION_EPS` to the module as `dual_quaternion._eps`
  PyModule_AddObject(module, "_eps", PyFloat_FromDouble(_DUAL_QUATERNION_EPS));
 
  // Finally, add this dual_quaternion object to the dual_quaternion module itself
  PyModule_AddObject(module, "dual_quaternion", (PyObject *)&PyDualQuaternion_Type);


#if PY_MAJOR_VERSION >= 3
    return module;
#else
    return;
#endif
}
