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

cdef double *_namespace_timedarray_3_values

cdef double _timedarray_3(const double t, const int i):
    global _namespace_timedarray_3_values
    cdef double epsilon = 0.001000000000000000 / 8
    if i < 0 or i >= 3:
        return _numpy.nan
    cdef int timestep = (int)((t/epsilon + 0.5)/8)
    if timestep < 0:
       timestep = 0
    elif timestep >= 20260:
        timestep = 20260-1
    return _namespace_timedarray_3_values[timestep*3 + i]


# template-specific support code

def main(_namespace):
    cdef size_t _idx
    cdef size_t _vectorisation_idx
        
    _var_N = _namespace["_var_N"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_in_mon_N = _namespace['_array_in_mon_N']
    cdef int32_t * _array_in_mon_N = <int32_t *> _buf__array_in_mon_N.data
    cdef int32_t N
    _var___source_fin_ng_in_i = _namespace["_var___source_fin_ng_in_i"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_ng_in_i = _namespace['_array_ng_in_i']
    cdef int32_t * _array_ng_in_i = <int32_t *> _buf__array_ng_in_i.data
    cdef size_t _num_array_ng_in_i = len(_namespace['_array_ng_in_i'])
    cdef int32_t __source_fin_ng_in_i
    _var___source_fin_ng_in_t = _namespace["_var___source_fin_ng_in_t"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_defaultclock_t = _namespace['_array_defaultclock_t']
    cdef double * _array_defaultclock_t = <double *> _buf__array_defaultclock_t.data
    cdef double __source_fin_ng_in_t
    _var__clock_t = _namespace["_var__clock_t"]
    _var__indices = _namespace["_var__indices"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_in_mon__indices = _namespace['_array_in_mon__indices']
    cdef int32_t * _array_in_mon__indices = <int32_t *> _buf__array_in_mon__indices.data
    cdef size_t _num_array_in_mon__indices = len(_namespace['_array_in_mon__indices'])
    cdef int32_t _indices
    cdef float _source_fin
    cdef float _to_record_fin
    _var_fin = _namespace["_var_fin"]
    cdef object _dynamic_array_in_mon_fin_capsule = _namespace["_dynamic_array_in_mon_fin_capsule"]
    cdef DynamicArray2DCpp[float]* _dynamic_array_in_mon_fin_ptr = <DynamicArray2DCpp[float]*>PyCapsule_GetPointer(_dynamic_array_in_mon_fin_capsule, "DynamicArray2D")
    cdef float* _array_in_mon_fin = <float*> _dynamic_array_in_mon_fin_ptr.get_data_ptr()
    cdef size_t _num_array_in_mon_fin = len(_namespace['_array_in_mon_fin'])
    cdef float fin
    # namespace for function fin_ta
    global _namespace_timedarray_3_values
    global _namespace_num_timedarray_3_values
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__timedarray_3_values = _namespace['_timedarray_3_values']
    _namespace_timedarray_3_values = <double *> _buf__timedarray_3_values.data
    _namespace_num_timedarray_3_values = len(_namespace['_timedarray_3_values'])
    _var_t = _namespace["_var_t"]
    cdef object _dynamic_array_in_mon_t_capsule = _namespace["_dynamic_array_in_mon_t_capsule"]
    cdef DynamicArray1DCpp[double]* _dynamic_array_in_mon_t_ptr = <DynamicArray1DCpp[double]*>PyCapsule_GetPointer(_dynamic_array_in_mon_t_capsule, "DynamicArray1D")
    cdef double* _array_in_mon_t = <double*> _dynamic_array_in_mon_t_ptr.get_data_ptr()
    cdef size_t _num_array_in_mon_t = len(_namespace['_array_in_mon_t'])
    cdef double t
    _var_t_ta_start = _namespace["_var_t_ta_start"]
    cdef double t_ta_start = _namespace["t_ta_start"]
    cdef float _lio_1

    if '_owner' in _namespace:
        _owner = _namespace['_owner']

    cdef size_t _new_len = _array_in_mon_N[0] + 1

    # Resize the recorded times
    _dynamic_array_in_mon_t_ptr.resize(_new_len)
    cdef double* _t_data = _dynamic_array_in_mon_t_ptr.get_data_ptr()
    _t_data[_new_len-1] = _array_defaultclock_t[0]

    # scalar code
    _vectorisation_idx = 1
        
    __source_fin_ng_in_t = _array_defaultclock_t[0]
    _lio_1 = __source_fin_ng_in_t - t_ta_start


    cdef size_t _i


    # Resize the recorded variable "fin" and get the (potentially
    # changed) reference to the underlying data
    _dynamic_array_in_mon_fin_ptr.resize(_new_len, _num_array_in_mon__indices)
    cdef float* _record_data_fin = <float*> _dynamic_array_in_mon_fin_ptr.get_data_ptr()
    for _i in range(_num_array_in_mon__indices):
        # vector code
        _idx = _array_in_mon__indices[_i]
        _vectorisation_idx = _idx

                
        __source_fin_ng_in_i = _array_ng_in_i[_idx]
        _source_fin = _timedarray_3(_lio_1, __source_fin_ng_in_i)
        _to_record_fin = _source_fin


        _record_data_fin[(_new_len-1)*_num_array_in_mon__indices + _i] = _to_record_fin

    # set the N variable explicitly (since we do not call `StateMonitor.resize`)
    _array_in_mon_N[0] = _new_len