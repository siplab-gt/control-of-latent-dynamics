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
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_syn_ProportionalCurrentOpsin_ng_main_N = _namespace['_array_syn_ProportionalCurrentOpsin_ng_main_N']
    cdef int32_t * _array_syn_ProportionalCurrentOpsin_ng_main_N = <int32_t *> _buf__array_syn_ProportionalCurrentOpsin_ng_main_N.data
    cdef int32_t N = _namespace["N"]
    _var_N_post = _namespace["_var_N_post"]
    cdef int64_t N_post = _namespace["N_post"]
    _var_N_pre = _namespace["_var_N_pre"]
    cdef int64_t N_pre = _namespace["N_pre"]
    cdef int32_t _real_sources
    cdef int32_t _real_targets
    _var__source_offset = _namespace["_var__source_offset"]
    cdef int64_t _source_offset = _namespace["_source_offset"]
    _var__synaptic_post = _namespace["_var__synaptic_post"]
    cdef object _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_capsule = _namespace["_dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_capsule, "DynamicArray1D")
    cdef int32_t* _array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post = <int32_t*> _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_ptr.get_data_ptr()
    cdef size_t _num_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post = len(_namespace['_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post'])
    cdef int32_t _synaptic_post
    _var__synaptic_pre = _namespace["_var__synaptic_pre"]
    cdef object _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_capsule = _namespace["_dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_capsule, "DynamicArray1D")
    cdef int32_t* _array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre = <int32_t*> _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_ptr.get_data_ptr()
    cdef size_t _num_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre = len(_namespace['_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre'])
    cdef int32_t _synaptic_pre
    _var__target_offset = _namespace["_var__target_offset"]
    cdef int64_t _target_offset = _namespace["_target_offset"]
    _var_sources = _namespace["_var_sources"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_syn_ProportionalCurrentOpsin_ng_main_sources = _namespace['_array_syn_ProportionalCurrentOpsin_ng_main_sources']
    cdef int32_t * _array_syn_ProportionalCurrentOpsin_ng_main_sources = <int32_t *> _buf__array_syn_ProportionalCurrentOpsin_ng_main_sources.data
    cdef size_t _num_array_syn_ProportionalCurrentOpsin_ng_main_sources = len(_namespace['_array_syn_ProportionalCurrentOpsin_ng_main_sources'])
    cdef int32_t sources
    _var_targets = _namespace["_var_targets"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_syn_ProportionalCurrentOpsin_ng_main_targets = _namespace['_array_syn_ProportionalCurrentOpsin_ng_main_targets']
    cdef int32_t * _array_syn_ProportionalCurrentOpsin_ng_main_targets = <int32_t *> _buf__array_syn_ProportionalCurrentOpsin_ng_main_targets.data
    cdef size_t _num_array_syn_ProportionalCurrentOpsin_ng_main_targets = len(_namespace['_array_syn_ProportionalCurrentOpsin_ng_main_targets'])
    cdef int32_t targets

    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    cdef size_t _old_num_synapses = _array_syn_ProportionalCurrentOpsin_ng_main_N[0]
    cdef size_t _new_num_synapses = _old_num_synapses + _num_array_syn_ProportionalCurrentOpsin_ng_main_sources

    _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_ptr.resize(_new_num_synapses)
    _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_ptr.resize(_new_num_synapses)
    # Get the potentially newly created underlying data arrays
    cdef int32_t* _synaptic_pre_data = _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_pre_ptr.get_data_ptr()
    cdef int32_t* _synaptic_post_data = _dynamic_array_syn_ProportionalCurrentOpsin_ng_main__synaptic_post_ptr.get_data_ptr()

    for _idx in range(_num_array_syn_ProportionalCurrentOpsin_ng_main_sources):
        # After this code has been executed, the arrays _real_sources and
        # _real_variables contain the final indices. Having any code here it all is
        # only necessary for supporting subgroups
                
        sources = _array_syn_ProportionalCurrentOpsin_ng_main_sources[_idx]
        targets = _array_syn_ProportionalCurrentOpsin_ng_main_targets[_idx]
        _real_sources = sources
        _real_targets = targets

        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets

    # now we need to resize all registered variables and set the total number
    # of synapses (via Python)
    _owner._resize(_new_num_synapses)

    # And update N_incoming, N_outgoing and synapse_number
    _owner._update_synapse_numbers(_old_num_synapses)