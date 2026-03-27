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
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_spk_mon_N = _namespace['_array_spk_mon_N']
    cdef int32_t * _array_spk_mon_N = <int32_t *> _buf__array_spk_mon_N.data
    cdef int32_t N
    _var__clock_t = _namespace["_var__clock_t"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_defaultclock_t = _namespace['_array_defaultclock_t']
    cdef double * _array_defaultclock_t = <double *> _buf__array_defaultclock_t.data
    cdef double _clock_t
    _var__source_i = _namespace["_var__source_i"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_ng_main_i = _namespace['_array_ng_main_i']
    cdef int32_t * _array_ng_main_i = <int32_t *> _buf__array_ng_main_i.data
    cdef size_t _num_array_ng_main_i = len(_namespace['_array_ng_main_i'])
    cdef int32_t _source_i
    _var__source_idx = _namespace["_var__source_idx"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_spk_mon__source_idx = _namespace['_array_spk_mon__source_idx']
    cdef int32_t * _array_spk_mon__source_idx = <int32_t *> _buf__array_spk_mon__source_idx.data
    cdef size_t _num_array_spk_mon__source_idx = len(_namespace['_array_spk_mon__source_idx'])
    cdef int32_t _source_idx
    _var__source_start = _namespace["_var__source_start"]
    cdef int64_t _source_start = _namespace["_source_start"]
    _var__source_stop = _namespace["_var__source_stop"]
    cdef int64_t _source_stop = _namespace["_source_stop"]
    _var__source_t = _namespace["_var__source_t"]
    _var__spikespace = _namespace["_var__spikespace"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_ng_main__spikespace = _namespace['_array_ng_main__spikespace']
    cdef int32_t * _array_ng_main__spikespace = <int32_t *> _buf__array_ng_main__spikespace.data
    cdef size_t _num_array_ng_main__spikespace = len(_namespace['_array_ng_main__spikespace'])
    cdef int32_t _spikespace
    cdef int32_t _to_record_i
    cdef double _to_record_t
    _var_count = _namespace["_var_count"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_spk_mon_count = _namespace['_array_spk_mon_count']
    cdef int32_t * _array_spk_mon_count = <int32_t *> _buf__array_spk_mon_count.data
    cdef size_t _num_array_spk_mon_count = len(_namespace['_array_spk_mon_count'])
    cdef int32_t count
    _var_i = _namespace["_var_i"]
    cdef object _dynamic_array_spk_mon_i_capsule = _namespace["_dynamic_array_spk_mon_i_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_spk_mon_i_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_spk_mon_i_capsule, "DynamicArray1D")
    cdef int32_t* _array_spk_mon_i = <int32_t*> _dynamic_array_spk_mon_i_ptr.get_data_ptr()
    cdef size_t _num_array_spk_mon_i = len(_namespace['_array_spk_mon_i'])
    cdef int32_t i
    _var_t = _namespace["_var_t"]
    cdef object _dynamic_array_spk_mon_t_capsule = _namespace["_dynamic_array_spk_mon_t_capsule"]
    cdef DynamicArray1DCpp[double]* _dynamic_array_spk_mon_t_ptr = <DynamicArray1DCpp[double]*>PyCapsule_GetPointer(_dynamic_array_spk_mon_t_capsule, "DynamicArray1D")
    cdef double* _array_spk_mon_t = <double*> _dynamic_array_spk_mon_t_ptr.get_data_ptr()
    cdef size_t _num_array_spk_mon_t = len(_namespace['_array_spk_mon_t'])
    cdef double t

    if '_owner' in _namespace:
        _owner = _namespace['_owner']

    cdef size_t _num_events = _array_ng_main__spikespace[_num_array_ng_main__spikespace-1]
    cdef size_t _start_idx, _end_idx, _curlen, _newlen, _j
    cdef int32_t* _i_ptr
    cdef double* _t_ptr
    if _num_events > 0:
        # For subgroups, we do not want to record all spikes
        # We assume that spikes are ordered
        _start_idx = _num_events
        _end_idx = _num_events
        for _j in range(_num_events):
            _idx = _array_ng_main__spikespace[_j]
            if _idx >= _source_start:
                _start_idx = _j
                break
        for _j in range(_num_events-1, _start_idx-1, -1):
            _idx = _array_ng_main__spikespace[_j]
            if _idx < _source_stop:
                break
            _end_idx = _j
        _num_events = _end_idx - _start_idx
        if _num_events > 0:
            # scalar code
            _vectorisation_idx = 1
                        
            _source_t = _array_defaultclock_t[0]

            _curlen = _array_spk_mon_N[0]
            _newlen = _curlen + _num_events
            # Resize the C++ arrays directly, avoiding Python indirection for efficiency
            _dynamic_array_spk_mon_i_ptr.resize(_newlen)
            _dynamic_array_spk_mon_t_ptr.resize(_newlen)
            # Update N after resize
            _array_spk_mon_N[0] = _newlen

            # No we get new fresh pointers after resize
            _i_ptr = _dynamic_array_spk_mon_i_ptr.get_data_ptr()
            _t_ptr = _dynamic_array_spk_mon_t_ptr.get_data_ptr()

            # Copy the values across
            for _j in range(_start_idx, _end_idx):
                _idx = _array_ng_main__spikespace[_j]
                _vectorisation_idx = _idx
                                
                _source_i = _array_ng_main_i[_idx]
                _to_record_i = _source_i
                _to_record_t = _source_t

                _i_ptr[_curlen + _j - _start_idx] = _to_record_i
                _t_ptr[_curlen + _j - _start_idx] = _to_record_t
                _array_spk_mon_count[_idx - _source_start] += 1