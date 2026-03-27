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
        
    _var_N = _namespace["_var_N"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_main_mon_N = _namespace['_array_main_mon_N']
    cdef int32_t * _array_main_mon_N = <int32_t *> _buf__array_main_mon_N.data
    cdef int32_t N
    _var__clock_t = _namespace["_var__clock_t"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_defaultclock_t = _namespace['_array_defaultclock_t']
    cdef double * _array_defaultclock_t = <double *> _buf__array_defaultclock_t.data
    cdef double _clock_t
    _var__indices = _namespace["_var__indices"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_main_mon__indices = _namespace['_array_main_mon__indices']
    cdef int32_t * _array_main_mon__indices = <int32_t *> _buf__array_main_mon__indices.data
    cdef size_t _num_array_main_mon__indices = len(_namespace['_array_main_mon__indices'])
    cdef int32_t _indices
    _var__source_v = _namespace["_var__source_v"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_v = _namespace['_array_ng_main_v']
    cdef float * _array_ng_main_v = <float *> _buf__array_ng_main_v.data
    cdef size_t _num_array_ng_main_v = len(_namespace['_array_ng_main_v'])
    cdef float _source_v
    _var__source_vJsbar = _namespace["_var__source_vJsbar"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_vJsbar = _namespace['_array_ng_main_vJsbar']
    cdef float * _array_ng_main_vJsbar = <float *> _buf__array_ng_main_vJsbar.data
    cdef size_t _num_array_ng_main_vJsbar = len(_namespace['_array_ng_main_vJsbar'])
    cdef float _source_vJsbar
    cdef float _to_record_v
    cdef float _to_record_vJsbar
    _var_t = _namespace["_var_t"]
    cdef object _dynamic_array_main_mon_t_capsule = _namespace["_dynamic_array_main_mon_t_capsule"]
    cdef DynamicArray1DCpp[double]* _dynamic_array_main_mon_t_ptr = <DynamicArray1DCpp[double]*>PyCapsule_GetPointer(_dynamic_array_main_mon_t_capsule, "DynamicArray1D")
    cdef double* _array_main_mon_t = <double*> _dynamic_array_main_mon_t_ptr.get_data_ptr()
    cdef size_t _num_array_main_mon_t = len(_namespace['_array_main_mon_t'])
    cdef double t
    _var_v = _namespace["_var_v"]
    cdef object _dynamic_array_main_mon_v_capsule = _namespace["_dynamic_array_main_mon_v_capsule"]
    cdef DynamicArray2DCpp[float]* _dynamic_array_main_mon_v_ptr = <DynamicArray2DCpp[float]*>PyCapsule_GetPointer(_dynamic_array_main_mon_v_capsule, "DynamicArray2D")
    cdef float* _array_main_mon_v = <float*> _dynamic_array_main_mon_v_ptr.get_data_ptr()
    cdef size_t _num_array_main_mon_v = len(_namespace['_array_main_mon_v'])
    cdef float v
    _var_vJsbar = _namespace["_var_vJsbar"]
    cdef object _dynamic_array_main_mon_vJsbar_capsule = _namespace["_dynamic_array_main_mon_vJsbar_capsule"]
    cdef DynamicArray2DCpp[float]* _dynamic_array_main_mon_vJsbar_ptr = <DynamicArray2DCpp[float]*>PyCapsule_GetPointer(_dynamic_array_main_mon_vJsbar_capsule, "DynamicArray2D")
    cdef float* _array_main_mon_vJsbar = <float*> _dynamic_array_main_mon_vJsbar_ptr.get_data_ptr()
    cdef size_t _num_array_main_mon_vJsbar = len(_namespace['_array_main_mon_vJsbar'])
    cdef float vJsbar

    if '_owner' in _namespace:
        _owner = _namespace['_owner']

    cdef size_t _new_len = _array_main_mon_N[0] + 1

    # Resize the recorded times
    _dynamic_array_main_mon_t_ptr.resize(_new_len)
    cdef double* _t_data = _dynamic_array_main_mon_t_ptr.get_data_ptr()
    _t_data[_new_len-1] = _array_defaultclock_t[0]

    # scalar code
    _vectorisation_idx = 1
        


    cdef size_t _i


    # Resize the recorded variable "v" and get the (potentially
    # changed) reference to the underlying data
    _dynamic_array_main_mon_v_ptr.resize(_new_len, _num_array_main_mon__indices)
    cdef float* _record_data_v = <float*> _dynamic_array_main_mon_v_ptr.get_data_ptr()
    for _i in range(_num_array_main_mon__indices):
        # vector code
        _idx = _array_main_mon__indices[_i]
        _vectorisation_idx = _idx

                
        _source_v = _array_ng_main_v[_idx]
        _source_vJsbar = _array_ng_main_vJsbar[_idx]
        _to_record_v = _source_v
        _to_record_vJsbar = _source_vJsbar


        _record_data_v[(_new_len-1)*_num_array_main_mon__indices + _i] = _to_record_v

    # Resize the recorded variable "vJsbar" and get the (potentially
    # changed) reference to the underlying data
    _dynamic_array_main_mon_vJsbar_ptr.resize(_new_len, _num_array_main_mon__indices)
    cdef float* _record_data_vJsbar = <float*> _dynamic_array_main_mon_vJsbar_ptr.get_data_ptr()
    for _i in range(_num_array_main_mon__indices):
        # vector code
        _idx = _array_main_mon__indices[_i]
        _vectorisation_idx = _idx

                
        _source_v = _array_ng_main_v[_idx]
        _source_vJsbar = _array_ng_main_vJsbar[_idx]
        _to_record_v = _source_v
        _to_record_vJsbar = _source_vJsbar


        _record_data_vJsbar[(_new_len-1)*_num_array_main_mon__indices + _i] = _to_record_vJsbar

    # set the N variable explicitly (since we do not call `StateMonitor.resize`)
    _array_main_mon_N[0] = _new_len