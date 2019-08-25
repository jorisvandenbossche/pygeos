// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdio.h>
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
pyquaternion_new(PyTypeObject *type, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
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

  Py_ssize_t size = PyTuple_Size(args);
  quaternion* q;
  PyObject* Q = {0};
  q = &(((PyQuaternion*)self)->obval);

  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError,
                    "quaternion constructor takes no keyword arguments");
    return -1;
  }

  q->w = 0.0;
  q->x = 0.0;
  q->y = 0.0;
  q->z = 0.0;

  if(size == 0) {
    return 0;
  } else if(size == 1) {
    if(PyArg_ParseTuple(args, "O", &Q) && PyQuaternion_Check(Q)) {
      q->w = ((PyQuaternion*)Q)->obval.w;
      q->x = ((PyQuaternion*)Q)->obval.x;
      q->y = ((PyQuaternion*)Q)->obval.y;
      q->z = ((PyQuaternion*)Q)->obval.z;
      return 0;
    } else if(PyArg_ParseTuple(args, "d", &q->w)) {
      return 0;
    }
  } else if(size == 3 && PyArg_ParseTuple(args, "ddd", &q->x, &q->y, &q->z)) {
    return 0;
  } else if(size == 4 && PyArg_ParseTuple(args, "dddd", &q->w, &q->x, &q->y, &q->z)) {
    return 0;
  }

  PyErr_SetString(PyExc_TypeError,
                  "quaternion constructor takes zero, one, three, or four float arguments, or a single quaternion");
  return -1;
}

// This is an array of methods (member functions) that will be
// available to use on the quaternion objects in python.  This is
// packaged up here, and will be used in the `tp_methods` field when
// definining the PyQuaternion_Type below.
PyMethodDef pyquaternion_methods[] = {
  {NULL, NULL, 0, NULL}
};

// This is an array of members (member data) that will be available to
// use on the quaternion objects in python.  This is packaged up here,
// and will be used in the `tp_members` field when definining the
// PyQuaternion_Type below.
PyMemberDef pyquaternion_members[] = {
  {"real", T_DOUBLE, offsetof(PyQuaternion, obval.w), 0,
   "The real component of the quaternion"},
  {"w", T_DOUBLE, offsetof(PyQuaternion, obval.w), 0,
   "The real component of the quaternion"},
  {"x", T_DOUBLE, offsetof(PyQuaternion, obval.x), 0,
   "The first imaginary component of the quaternion"},
  {"y", T_DOUBLE, offsetof(PyQuaternion, obval.y), 0,
   "The second imaginary component of the quaternion"},
  {"z", T_DOUBLE, offsetof(PyQuaternion, obval.z), 0,
   "The third imaginary component of the quaternion"},
  {NULL, 0, 0, 0, NULL}
};


static PyObject*
pyquaternion_richcompare(PyObject* a, PyObject* b, int op)
{
  quaternion x = {0.0, 0.0, 0.0, 0.0};
  quaternion y = {0.0, 0.0, 0.0, 0.0};
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
  "quaternion.quaternion",                    // tp_name
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
  0,                    // tp_as_number
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
  "Floating-point quaternion numbers",        // tp_doc
  0,                                          // tp_traverse
  0,                                          // tp_clear
  pyquaternion_richcompare,                   // tp_richcompare
  0,                                          // tp_weaklistoffset
  0,                                          // tp_iter
  0,                                          // tp_iternext
  pyquaternion_methods,                       // tp_methods
  pyquaternion_members,                       // tp_members
  0,                        // tp_getset
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
#if PY_VERSION_HEX >= 0x030400a1
  0,                                          // tp_finalize
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

static int QUATERNION_setitem(PyObject* item, quaternion* qp, void* NPY_UNUSED(ap))
{
  printf("In QUATERNION_setitem\n");
  PyObject *element;
  if(PyQuaternion_Check(item)) {
    memcpy(qp,&(((PyQuaternion *)item)->obval),sizeof(quaternion));
  } else if(PySequence_Check(item) && PySequence_Length(item)==4) {
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
QUATERNION_getitem(void* data, void* NPY_UNUSED(arr))
{
  printf("In QUATERNION_getitem\n");
  quaternion q;
  memcpy(&q,data,sizeof(quaternion));
  return PyQuaternion_FromQuaternion(q);
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


// This contains assorted other top-level methods for the module
static PyMethodDef QuaternionMethods[] = {
  {NULL, NULL, 0, NULL}
};

int quaternion_elsize = sizeof(quaternion);

typedef struct { char c; quaternion q; } align_test;
int quaternion_alignment = offsetof(align_test, q);


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//                                                             //
//  Everything above was preparation for the following set up  //
//                                                             //
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////


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
PyMODINIT_FUNC PyInit_dtype(void) {

  PyObject *module;
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
  module = Py_InitModule("dtype", QuaternionMethods);
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
  // _PyQuaternion_ArrFuncs.compare = (PyArray_CompareFunc*)QUATERNION_compare;
  // _PyQuaternion_ArrFuncs.argmax = (PyArray_ArgFunc*)QUATERNION_argmax;
  _PyQuaternion_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)QUATERNION_fillwithscalar;

  // The quaternion array descr
  quaternion_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  quaternion_descr->typeobj = &PyQuaternion_Type;
  quaternion_descr->kind = 'V';
  quaternion_descr->type = 'q';
  quaternion_descr->byteorder = '=';
  quaternion_descr->flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
  quaternion_descr->type_num = 0; // assigned at registration
  quaternion_descr->elsize = quaternion_elsize;
  quaternion_descr->alignment = quaternion_alignment;
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

  // Finally, add this quaternion object to the quaternion module itself
  PyModule_AddObject(module, "quaternion", (PyObject *)&PyQuaternion_Type);
  
  return module;
}
