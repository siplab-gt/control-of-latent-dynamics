#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
#cython: cpow=True
#cython: infer_types=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport fabs, sin, cos, tan, sinh, cosh, tanh, exp, log, log10, expm1, log1p, sqrt, asin, acos, atan, fmod, floor, ceil, isinf
cdef extern from "math.h":
    double M_PI
# Import the two versions of std::abs
from libc.stdlib cimport abs  # For integers
from libc.math cimport abs  # For floating point values
from libc.limits cimport INT_MIN, INT_MAX
from libcpp cimport bool
from libcpp.set cimport set
from cython.operator cimport dereference as _deref, preincrement as _preinc
cimport cython as _cython

_numpy.import_array()
cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(_numpy.PyArrayObject *arr, int flags)
from libc.stdlib cimport free

cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double x)
    double NPY_INFINITY

cdef extern from "stdint_compat.h":
    # Longness only used for type promotion
    # Actual compile time size used for conversion
    ctypedef signed int int32_t
    ctypedef signed long int64_t
    ctypedef unsigned long uint64_t
    # It seems we cannot used a fused type here
    cdef int int_(bool)
    cdef int int_(char)
    cdef int int_(short)
    cdef int int_(int)
    cdef int int_(long)
    cdef int int_(float)
    cdef int int_(double)
    cdef int int_(long double)

# PyCapsule support for direct C++ pointer access
from cpython.pycapsule cimport PyCapsule_GetPointer

# Dynamic array C++ interface declarations
cdef extern from "dynamic_array.h":
    cdef cppclass DynamicArray1DCpp "DynamicArray1D"[T]:
        void resize(size_t) except +
        void shrink_to_fit()
        T& operator[](size_t)
        T* get_data_ptr()
        size_t size()
        size_t capacity()

    cdef cppclass DynamicArray2DCpp "DynamicArray2D"[T]:
        void resize(size_t, size_t) except +
        void resize_along_first(size_t) except +
        void shrink_to_fit()
        T& operator()(size_t, size_t)
        T* get_data_ptr()
        size_t rows()
        size_t cols()
        size_t stride()


# support code



# template-specific support code

def main(_namespace):
    cdef size_t _idx
    cdef size_t _vectorisation_idx
        
    _var__spikespace = _namespace["_var__spikespace"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_ng_s__spikespace = _namespace['_array_ng_s__spikespace']
    cdef int32_t * _array_ng_s__spikespace = <int32_t *> _buf__array_ng_s__spikespace.data
    cdef size_t _num_array_ng_s__spikespace = len(_namespace['_array_ng_s__spikespace'])
    cdef int32_t _spikespace
    _var_kdm1_per_neuron = _namespace["_var_kdm1_per_neuron"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_s_kdm1_per_neuron = _namespace['_array_ng_s_kdm1_per_neuron']
    cdef float * _array_ng_s_kdm1_per_neuron = <float *> _buf__array_ng_s_kdm1_per_neuron.data
    cdef size_t _num_array_ng_s_kdm1_per_neuron = len(_namespace['_array_ng_s_kdm1_per_neuron'])
    cdef float kdm1_per_neuron
    _var_sf = _namespace["_var_sf"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_s_sf = _namespace['_array_ng_s_sf']
    cdef float * _array_ng_s_sf = <float *> _buf__array_ng_s_sf.data
    cdef size_t _num_array_ng_s_sf = len(_namespace['_array_ng_s_sf'])
    cdef float sf
    _var_ss = _namespace["_var_ss"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_s_ss = _namespace['_array_ng_s_ss']
    cdef float * _array_ng_s_ss = <float *> _buf__array_ng_s_ss.data
    cdef size_t _num_array_ng_s_ss = len(_namespace['_array_ng_s_ss'])
    cdef float ss
    _var_xPDown = _namespace["_var_xPDown"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_s_xPDown = _namespace['_array_ng_s_xPDown']
    cdef float * _array_ng_s_xPDown = <float *> _buf__array_ng_s_xPDown.data
    cdef size_t _num_array_ng_s_xPDown = len(_namespace['_array_ng_s_xPDown'])
    cdef float xPDown
    _var_xPUp = _namespace["_var_xPUp"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_s_xPUp = _namespace['_array_ng_s_xPUp']
    cdef float * _array_ng_s_xPUp = <float *> _buf__array_ng_s_xPUp.data
    cdef size_t _num_array_ng_s_xPUp = len(_namespace['_array_ng_s_xPUp'])
    cdef float xPUp

    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    # scalar code
    _vectorisation_idx = 1
        


    cdef size_t _num_events = _array_ng_s__spikespace[_num_array_ng_s__spikespace-1]
    for _index_events in range(_num_events):
        # vector code
        _idx = _array_ng_s__spikespace[_index_events]
        _vectorisation_idx = _idx
                
        sf = _array_ng_s_sf[_idx]
        ss = _array_ng_s_ss[_idx]
        xPDown = _array_ng_s_xPDown[_idx]
        xPUp = _array_ng_s_xPUp[_idx]
        kdm1_per_neuron = (ss * xPUp) + (sf * xPDown)
        _array_ng_s_kdm1_per_neuron[_idx] = kdm1_per_neuron