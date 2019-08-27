// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdio.h>
#include <Python.h>
#include <geos_c.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"

#include "quaternion.h"

#define RAISE_ILLEGAL_GEOS if (!PyErr_Occurred()) {PyErr_Format(PyExc_RuntimeError, "Uncaught GEOS exception");}

#define NPY_COPY_PYOBJECT_PTR(dst, src) memcpy(dst, src, sizeof(PyObject *))

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


/* This initializes a global GEOS Context */
static void *geos_context[1] = {NULL};

static void HandleGEOSError(const char *message, void *userdata) {
    PyErr_SetString(userdata, message);
}

static void HandleGEOSNotice(const char *message, void *userdata) {
    PyErr_WarnEx(PyExc_Warning, message, 1);
}




typedef GEOSGeometry* pGEOSGeom; 

// The basic python object holding a quaternion
typedef struct {
  PyObject_HEAD
  pGEOSGeom ptr;
} GeometryObject;

// typedef struct {
//   PyObject_HEAD
//   quaternion obval;
// } PyQuaternion;


static PyTypeObject GeometryType;

// This is the crucial feature that will make a quaternion into a
// built-in numpy data type.  We will describe its features below.
PyArray_Descr* geometry_descr;


static NPY_INLINE int
PyQuaternion_Check(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&GeometryType);
}

// static PyObject*
// PyQuaternion_FromQuaternion(quaternion q) {
//   PyQuaternion* p = (PyQuaternion*)GeometryType.tp_alloc(&GeometryType,0);
//   if (p) { p->obval = q; }
//   return (PyObject*)p;
// }

/* Initializes a new geometry object, without Empty check */
static PyObject*
GeometryObject_FromGEOS(GEOSGeometry* ptr)
{
    //GEOSGeometry *pgeom = (GEOSGeometry *) ptr;
    GeometryObject *self = (GeometryObject *) GeometryType.tp_alloc(&GeometryType, 0);
    if (self == NULL) {
        return NULL;
    } else {
        self->ptr = ptr;
        return (PyObject *) self;
    }
}


static void GeometryObject_dealloc(GeometryObject *self)
{
    printf("In GeometryObject_dealloc\n");
    void *context_handle = geos_context[0];
    if (self->ptr != NULL) {
        GEOSGeom_destroy_r(context_handle, self->ptr);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef GeometryObject_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(GeometryObject, ptr), READONLY, "pointer to GEOSGeometry"},
    {NULL}  /* Sentinel */
};


static PyObject *to_wkt(GeometryObject *obj, char *format, char trim,
                        int precision, int dimension, int use_old_3d)
{
    void *context_handle = geos_context[0];
    char *wkt;
    PyObject *result;
    if (obj->ptr == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    GEOSWKTWriter *writer = GEOSWKTWriter_create_r(context_handle);
    if (writer == NULL) {
        return NULL;
    }
    GEOSWKTWriter_setRoundingPrecision_r(context_handle, writer, precision);
    GEOSWKTWriter_setTrim_r(context_handle, writer, trim);
    GEOSWKTWriter_setOutputDimension_r(context_handle, writer, dimension);
    GEOSWKTWriter_setOld3D_r(context_handle, writer, use_old_3d);
    wkt = GEOSWKTWriter_write_r(context_handle, writer, obj->ptr);
    result = PyUnicode_FromFormat(format, wkt);
    GEOSFree_r(context_handle, wkt);
    GEOSWKTWriter_destroy_r(context_handle, writer);
    return result;
}


static PyObject *GeometryObject_ToWKT(GeometryObject *self, PyObject *args, PyObject *kw)
{
    printf("In GeometryObject_ToWKT\n");
    char trim = 1;
    int precision = 6;
    int dimension = 3;
    int use_old_3d = 0;
    static char *kwlist[] = {"precision", "trim", "dimension", "use_old_3d", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|ibib", kwlist,
                                     &precision, &trim, &dimension, &use_old_3d))
    {
        return NULL;
    }
    return to_wkt(self, "%s", trim, precision, dimension, use_old_3d);
}

static PyObject *GeometryObject_repr(GeometryObject *self)
{
    printf("In GeometryObject_repr\n");
    if (self->ptr == NULL) {
        printf("-- Pointer is null\n");
        return PyUnicode_FromString("<pygeos.Geometry NULL>");
    // } else if (self == Geom_Empty) {
    //     return PyUnicode_FromString("<pygeos.Empty>");
    } else {
        return to_wkt(self, "<pygeos.dtype.Geometry %s>", 1, 3, 3, 0);
    }
}

static PyObject *GeometryObject_ToWKB(GeometryObject *self, PyObject *args, PyObject *kw)
{
    void *context_handle = geos_context[0];
    unsigned char *wkb;
    size_t size;
    PyObject *result;
    int dimension = 3;
    int byte_order = 1;
    char include_srid = 0;
    char hex = 0;
    static char *kwlist[] = {"dimension", "byte_order", "include_srid", "hex", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|ibbb", kwlist,
                                     &dimension, &byte_order, &include_srid, &hex))
    {
        return NULL;
    }
    if (self->ptr == NULL) {
         Py_INCREF(Py_None);
         return Py_None;
    }
    GEOSWKBWriter *writer = GEOSWKBWriter_create_r(context_handle);
    if (writer == NULL) {
        return NULL;
    }
    GEOSWKBWriter_setOutputDimension_r(context_handle, writer, dimension);
    GEOSWKBWriter_setByteOrder_r(context_handle, writer, byte_order);
    GEOSWKBWriter_setIncludeSRID_r(context_handle, writer, include_srid);
    if (hex) {
        wkb = GEOSWKBWriter_writeHEX_r(context_handle, writer, self->ptr, &size);
    } else {
        wkb = GEOSWKBWriter_write_r(context_handle, writer, self->ptr, &size);
    }
    result = PyBytes_FromStringAndSize((char *) wkb, size);
    GEOSFree_r(context_handle, wkb);
    GEOSWKBWriter_destroy_r(context_handle, writer);
    return result;
}

static PyObject *GeometryObject_FromWKT(PyTypeObject *type, PyObject *value)
{
    printf("In GeometryObject_FromWKT \n");
    void *context_handle = geos_context[0];
    PyObject *result = NULL;
    char *wkt;
    GEOSGeometry *geom;
    GEOSWKTReader *reader;
    int pgeom;

    /* Cast the PyObject (bytes or str) to char* */
    if (PyBytes_Check(value)) {
        wkt = PyBytes_AsString(value);
        if (wkt == NULL) { return NULL; }
    }
    else if (PyUnicode_Check(value)) {
        wkt = PyUnicode_AsUTF8(value);
        if (wkt == NULL) { 
            printf("-- wkt is null \n");
            return NULL;
        }
    } else {
        PyErr_Format(PyExc_TypeError, "Expected bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }

    printf("-- Creating GEOSWKTReader_create_r\n");
    reader = GEOSWKTReader_create_r(context_handle);
    if (reader == NULL) {
        printf("-- no reader\n");
        return NULL;
    }

    printf("-- reading geom\n");
    geom = GEOSWKTReader_read_r(context_handle, reader, wkt);
    GEOSWKTReader_destroy_r(context_handle, reader);
    if (geom == NULL) {
        printf("-- geom is NULL\n");
        return NULL;
    }
    printf("-- Created geom GEOSGeometry pointer\n");

    printf("-- Going to call GeometryObject_FromGEOS\n");
    result = GeometryObject_FromGEOS(geom);
    if (result == NULL) {
        GEOSGeom_destroy_r(context_handle, geom);
        PyErr_Format(PyExc_RuntimeError, "Could not instantiate a new Geometry object");
    }
    printf("-- returning\n");
    return result;
}

static PyObject *GeometryObject_FromWKB(PyTypeObject *type, PyObject *value)
{
    void *context_handle = geos_context[0];
    PyObject *result = NULL;
    GEOSGeometry *geom;
    GEOSWKBReader *reader;
    char *wkb;
    Py_ssize_t size;
    char is_hex;

    /* Cast the PyObject (only bytes) to char* */
    if (!PyBytes_Check(value)) {
        PyErr_Format(PyExc_TypeError, "Expected bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }
    size = PyBytes_Size(value);
    wkb = PyBytes_AsString(value);
    if (wkb == NULL) {
        return NULL;
    }

    /* Check if this is a HEX WKB */
    if (size != 0) {
        is_hex = ((wkb[0] == 48) | (wkb[0] == 49));
    } else {
        is_hex = 0;
    }

    /* Create the reader and read the WKB */
    reader = GEOSWKBReader_create_r(context_handle);
    if (reader == NULL) {
        return NULL;
    }
    if (is_hex) {
        geom = GEOSWKBReader_readHEX_r(context_handle, reader, (unsigned char *) wkb, size);
    } else {
        geom = GEOSWKBReader_read_r(context_handle, reader, (unsigned char *) wkb, size);
    }
    GEOSWKBReader_destroy_r(context_handle, reader);
    if (geom == NULL) {
        return NULL;
    }
    result = GeometryObject_FromGEOS(geom);
    if (result == NULL) {
        GEOSGeom_destroy_r(context_handle, geom);
        PyErr_Format(PyExc_RuntimeError, "Could not instantiate a new Geometry object");
    }
    return result;
}


static PyObject *GeometryObject_new(PyTypeObject *type, PyObject *args,
                                    PyObject *kwds)
{
    printf("In GeometryObject_new \n");
    void *context_handle = geos_context[0];
    GEOSGeometry *arg;
    GEOSGeometry *ptr;
    PyObject *self;
    PyObject *value;

    if (!PyArg_ParseTuple(args, "O", &value)) {
        return NULL;
    }

    if (PyBytes_Check(value)) {
        return GeometryObject_FromWKB(type, value);
    }
    else if (PyUnicode_Check(value)) {
        return GeometryObject_FromWKT(type, value);
    }
    else {
        PyErr_Format(PyExc_TypeError, "Expected string or bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }
    ptr = GEOSGeom_clone_r(context_handle, arg);
    if (ptr == NULL) {
        RAISE_ILLEGAL_GEOS;
        return NULL;
    }
    self = GeometryObject_FromGEOS(ptr);
    return (PyObject *) self;
}

static PyMethodDef GeometryObject_methods[] = {
    {"to_wkt", (PyCFunction) GeometryObject_ToWKT, METH_VARARGS | METH_KEYWORDS,
     "Write the geometry to Well-Known Text (WKT) format"
    },
    {"to_wkb", (PyCFunction) GeometryObject_ToWKB, METH_VARARGS | METH_KEYWORDS,
     "Write the geometry to Well-Known Binary (WKB) format"
    },
    {"from_wkt", (PyCFunction) GeometryObject_FromWKT, METH_CLASS | METH_O,
     "Read the geometry from Well-Known Text (WKT) format"
    },
    {"from_wkb", (PyCFunction) GeometryObject_FromWKB, METH_CLASS | METH_O,
     "Read the geometry from Well-Known Binary (WKB) format"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject GeometryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "geometry",
    .tp_doc = "Geometry type",
    .tp_basicsize = sizeof(GeometryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeometryObject_new,
    .tp_dealloc = (destructor) GeometryObject_dealloc,
    .tp_members = GeometryObject_members,
    .tp_methods = GeometryObject_methods,
    .tp_repr = (reprfunc) GeometryObject_repr,
};


// #define PyQuaternion_AsQuaternion(q, o)                                 \
//   /* fprintf (stderr, "file %s, line %d., PyQuaternion_AsQuaternion\n", __FILE__, __LINE__); */ \
//   if(PyQuaternion_Check(o)) {                                           \
//     q = ((PyQuaternion*)o)->obval;                                      \
//   } else {                                                              \
//     PyErr_SetString(PyExc_TypeError,                                    \
//                     "Input object is not a quaternion.");               \
//     return NULL;                                                        \
//   }

// #define PyQuaternion_AsQuaternionPointer(q, o)                          \
//   /* fprintf (stderr, "file %s, line %d, PyQuaternion_AsQuaternionPointer.\n", __FILE__, __LINE__); */ \
//   if(PyQuaternion_Check(o)) {                                           \
//     q = &((PyQuaternion*)o)->obval;                                     \
//   } else {                                                              \
//     PyErr_SetString(PyExc_TypeError,                                    \
//                     "Input object is not a quaternion.");               \
//     return NULL;                                                        \
//   }

// static PyObject *
// pyquaternion_new(PyTypeObject *type, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
// {
//   PyQuaternion* self;
//   self = (PyQuaternion *)type->tp_alloc(type, 0);
//   return (PyObject *)self;
// }

// static int
// pyquaternion_init(PyObject *self, PyObject *args, PyObject *kwds)
// {
//   // "A good rule of thumb is that for immutable types, all
//   // initialization should take place in `tp_new`, while for mutable
//   // types, most initialization should be deferred to `tp_init`."
//   // ---Python 2.7.8 docs

//   Py_ssize_t size = PyTuple_Size(args);
//   quaternion* q;
//   PyObject* Q = {0};
//   q = &(((PyQuaternion*)self)->obval);

//   if (kwds && PyDict_Size(kwds)) {
//     PyErr_SetString(PyExc_TypeError,
//                     "quaternion constructor takes no keyword arguments");
//     return -1;
//   }

//   q->w = 0.0;
//   q->x = 0.0;
//   q->y = 0.0;
//   q->z = 0.0;

//   if(size == 0) {
//     return 0;
//   } else if(size == 1) {
//     if(PyArg_ParseTuple(args, "O", &Q) && PyQuaternion_Check(Q)) {
//       q->w = ((PyQuaternion*)Q)->obval.w;
//       q->x = ((PyQuaternion*)Q)->obval.x;
//       q->y = ((PyQuaternion*)Q)->obval.y;
//       q->z = ((PyQuaternion*)Q)->obval.z;
//       return 0;
//     } else if(PyArg_ParseTuple(args, "d", &q->w)) {
//       return 0;
//     }
//   } else if(size == 3 && PyArg_ParseTuple(args, "ddd", &q->x, &q->y, &q->z)) {
//     return 0;
//   } else if(size == 4 && PyArg_ParseTuple(args, "dddd", &q->w, &q->x, &q->y, &q->z)) {
//     return 0;
//   }

//   PyErr_SetString(PyExc_TypeError,
//                   "quaternion constructor takes zero, one, three, or four float arguments, or a single quaternion");
//   return -1;
// }

// // This is an array of methods (member functions) that will be
// // available to use on the quaternion objects in python.  This is
// // packaged up here, and will be used in the `tp_methods` field when
// // definining the GeometryType below.
// PyMethodDef pyquaternion_methods[] = {
//   {NULL, NULL, 0, NULL}
// };

// // This is an array of members (member data) that will be available to
// // use on the quaternion objects in python.  This is packaged up here,
// // and will be used in the `tp_members` field when definining the
// // GeometryType below.
// PyMemberDef pyquaternion_members[] = {
//   {"real", T_DOUBLE, offsetof(PyQuaternion, obval.w), 0,
//    "The real component of the quaternion"},
//   {"w", T_DOUBLE, offsetof(PyQuaternion, obval.w), 0,
//    "The real component of the quaternion"},
//   {"x", T_DOUBLE, offsetof(PyQuaternion, obval.x), 0,
//    "The first imaginary component of the quaternion"},
//   {"y", T_DOUBLE, offsetof(PyQuaternion, obval.y), 0,
//    "The second imaginary component of the quaternion"},
//   {"z", T_DOUBLE, offsetof(PyQuaternion, obval.z), 0,
//    "The third imaginary component of the quaternion"},
//   {NULL, 0, 0, 0, NULL}
// };


// static PyObject*
// pyquaternion_richcompare(PyObject* a, PyObject* b, int op)
// {
//   quaternion x = {0.0, 0.0, 0.0, 0.0};
//   quaternion y = {0.0, 0.0, 0.0, 0.0};
//   int result = 0;
//   PyQuaternion_AsQuaternion(x,a);
//   PyQuaternion_AsQuaternion(y,b);
//   #define COMPARISONOP(py,op) case py: result = quaternion_##op(x,y); break;
//   switch (op) {
//     COMPARISONOP(Py_LT,less)
//     COMPARISONOP(Py_LE,less_equal)
//     COMPARISONOP(Py_EQ,equal)
//     COMPARISONOP(Py_NE,not_equal)
//     COMPARISONOP(Py_GT,greater)
//     COMPARISONOP(Py_GE,greater_equal)
//   };
//   #undef COMPARISONOP
//   return PyBool_FromLong(result);
// }


// static long
// pyquaternion_hash(PyObject *o)
// {
//   quaternion q = ((PyQuaternion *)o)->obval;
//   long value = 0x456789;
//   value = (10000004 * value) ^ _Py_HashDouble(q.w);
//   value = (10000004 * value) ^ _Py_HashDouble(q.x);
//   value = (10000004 * value) ^ _Py_HashDouble(q.y);
//   value = (10000004 * value) ^ _Py_HashDouble(q.z);
//   if (value == -1)
//     value = -2;
//   return value;
// }

// static PyObject *
// pyquaternion_repr(PyObject *o)
// {
//   char str[128];
//   quaternion q = ((PyQuaternion *)o)->obval;
//   sprintf(str, "quaternion(%.15g, %.15g, %.15g, %.15g)", q.w, q.x, q.y, q.z);
//   return PyUString_FromString(str);
// }

// static PyObject *
// pyquaternion_str(PyObject *o)
// {
//   char str[128];
//   quaternion q = ((PyQuaternion *)o)->obval;
//   sprintf(str, "quaternion(%.15g, %.15g, %.15g, %.15g)", q.w, q.x, q.y, q.z);
//   return PyUString_FromString(str);
// }


// // This establishes the quaternion as a python object (not yet a numpy
// // scalar type).  The name may be a little counterintuitive; the idea
// // is that this will be a type that can be used as an array dtype.
// // Note that many of the slots below will be filled later, after the
// // corresponding functions are defined.
// static PyTypeObject GeometryType = {
// #if PY_MAJOR_VERSION >= 3
//   PyVarObject_HEAD_INIT(NULL, 0)
// #else
//   PyObject_HEAD_INIT(NULL)
//   0,                                          // ob_size
// #endif
//   "quaternion.quaternion",                    // tp_name
//   sizeof(GeometryObject),                       // tp_basicsize
//   0,                                          // tp_itemsize
//   0,                                          // tp_dealloc
//   0,                                          // tp_print
//   0,                                          // tp_getattr
//   0,                                          // tp_setattr
// #if PY_MAJOR_VERSION >= 3
//   0,                                          // tp_reserved
// #else
//   0,                                          // tp_compare
// #endif
//   pyquaternion_repr,                          // tp_repr
//   0,                    // tp_as_number
//   0,                                          // tp_as_sequence
//   0,                                          // tp_as_mapping
//   pyquaternion_hash,                          // tp_hash
//   0,                                          // tp_call
//   pyquaternion_str,                           // tp_str
//   0,                                          // tp_getattro
//   0,                                          // tp_setattro
//   0,                                          // tp_as_buffer
// #if PY_MAJOR_VERSION >= 3
//   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   // tp_flags
// #else
//   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, // tp_flags
// #endif
//   "Floating-point quaternion numbers",        // tp_doc
//   0,                                          // tp_traverse
//   0,                                          // tp_clear
//   pyquaternion_richcompare,                   // tp_richcompare
//   0,                                          // tp_weaklistoffset
//   0,                                          // tp_iter
//   0,                                          // tp_iternext
//   pyquaternion_methods,                       // tp_methods
//   pyquaternion_members,                       // tp_members
//   0,                        // tp_getset
//   0,                                          // tp_base; will be reset to &PyGenericArrType_Type after numpy import
//   0,                                          // tp_dict
//   0,                                          // tp_descr_get
//   0,                                          // tp_descr_set
//   0,                                          // tp_dictoffset
//   pyquaternion_init,                          // tp_init
//   0,                                          // tp_alloc
//   pyquaternion_new,                           // tp_new
//   0,                                          // tp_free
//   0,                                          // tp_is_gc
//   0,                                          // tp_bases
//   0,                                          // tp_mro
//   0,                                          // tp_cache
//   0,                                          // tp_subclasses
//   0,                                          // tp_weaklist
//   0,                                          // tp_del
// #if PY_VERSION_HEX >= 0x02060000
//   0,                                          // tp_version_tag
// #endif
// #if PY_VERSION_HEX >= 0x030400a1
//   0,                                          // tp_finalize
// #endif
// };

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


static int
QUATERNION_setitem(PyObject *op, void *ov, void *NPY_UNUSED(ap))
{
    printf("In QUATERNION_setitem\n");
    PyObject *obj;

    NPY_COPY_PYOBJECT_PTR(&obj, ov);

    Py_INCREF(op);
    Py_XDECREF(obj);

    NPY_COPY_PYOBJECT_PTR(ov, &op);

    return PyErr_Occurred() ? -1 : 0;
}




// static int QUATERNION_setitem(PyObject* item, GeometryObject* qp, void* NPY_UNUSED(ap))
// {
//   printf("In QUATERNION_setitem\n");
//   void *context_handle = geos_context[0];
//   PyObject *element;
//   GEOSGeometry *ptr, *ptr_copy;
//   GeometryObject *geom_copy;

//   if(PyQuaternion_Check(item)) {
    
//     NPY_COPY_PYOBJECT_PTR(&element, qp);
//     Py_INCREF(item);
//     Py_XDECREF(element);
//     NPY_COPY_PYOBJECT_PTR(qp, &item);

//     // // ptr = ((GeometryObject *)item)->ptr;
//     // // ptr_copy = GEOSGeom_clone_r(context_handle, ptr);
//     // // geom_copy = (GeometryObject *) GeometryObject_FromGEOS(ptr_copy);
//     // Py_INCREF(item);

//     // memcpy(qp,&item,sizeof(*GeometryObject));

//   } else {
//     PyErr_SetString(PyExc_TypeError,
//                     "Unknown input to QUATERNION_setitem");
//     return -1;
//   }
//   return 0;
// }

// When a numpy array of dtype=quaternion is indexed, this function is
// called, returning a new quaternion object with a copy of the
// data... sometimes...
static PyObject *
QUATERNION_getitem(void* data, void* NPY_UNUSED(arr))
{
  printf("In QUATERNION_getitem\n");
//   quaternion q;
//   memcpy(&q,data,sizeof(quaternion));
//   return PyQuaternion_FromQuaternion(q);
  PyObject *obj;
  NPY_COPY_PYOBJECT_PTR(&obj, data);
  if (obj == NULL) {
    Py_RETURN_NONE;
  }
  else {
    Py_INCREF(obj);
    return obj;
  }

  // void *context_handle = geos_context[0];
  // pGEOSGeom ptr, ptr_copy;
  // GeometryObject *geom, *geom_copy;
  
  // printf("-- going to copy memory\n");
  // memcpy(&geom,data,sizeof(GeometryObject));
  // //ptr = (GEOSGeometry *) &data
  // Py_INCREF(geom);

  // // printf("-- getting actual geos pointer\n");
  // // ptr = ((GeometryObject *)geom)->ptr;
  // // printf("-- taking copy\n");
  // // ptr_copy = GEOSGeom_clone_r(context_handle, ptr);
  // // printf("-- creating new GeometryObject\n");
  // // geom_copy = (GeometryObject *) GeometryObject_FromGEOS(ptr_copy);
  // // printf("-- returning the object as PyObject*\n");
  // return (PyObject *) geom;

//   ptr_copy = GEOSGeom_clone_r(context_handle, ptr);
//   return GeometryObject_FromGEOS(ptr_copy);

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

// int quaternion_elsize = sizeof(GeometryObject);
int quaternion_elsize = sizeof(PyObject *);

typedef struct { char c; GeometryObject q; } align_test;
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

  void *context_handle = GEOS_init_r();
  PyObject* GEOSException = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
  PyModule_AddObject(module, "GEOSException", GEOSException);
  GEOSContext_setErrorMessageHandler_r(context_handle, HandleGEOSError, GEOSException);
  GEOSContext_setNoticeMessageHandler_r(context_handle, HandleGEOSNotice, NULL);
  geos_context[0] = context_handle;  /* for global access */

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
  GeometryType.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&GeometryType) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "Could not initialize GeometryType.");
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
  geometry_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  geometry_descr->typeobj = &GeometryType;
  //geometry_descr->kind = 'V';
  geometry_descr->kind = 'O';
  geometry_descr->type = 'q';
  geometry_descr->byteorder = '|';
  //geometry_descr->flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
  geometry_descr->flags = NPY_OBJECT_DTYPE_FLAGS;
  geometry_descr->type_num = 0; // assigned at registration
  geometry_descr->elsize = quaternion_elsize;
  geometry_descr->alignment = quaternion_alignment;
  geometry_descr->subarray = NULL;
  geometry_descr->fields = NULL;
  geometry_descr->names = NULL;
  geometry_descr->f = &_PyQuaternion_ArrFuncs;
  geometry_descr->metadata = NULL;
  geometry_descr->c_metadata = NULL;

  Py_INCREF(&GeometryType);
  quaternionNum = PyArray_RegisterDataType(geometry_descr);

  if (quaternionNum < 0) {
    INITERROR;
  }

  // Finally, add this quaternion object to the quaternion module itself
  PyModule_AddObject(module, "geometry", (PyObject *)&GeometryType);
  
  return module;
}
