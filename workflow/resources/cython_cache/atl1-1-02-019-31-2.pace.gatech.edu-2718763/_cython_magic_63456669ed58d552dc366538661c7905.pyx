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
        
    _var_Irr0_pre = _namespace["_var_Irr0_pre"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_light_source_Irr0 = _namespace['_array_light_source_Irr0']
    cdef float * _array_light_source_Irr0 = <float *> _buf__array_light_source_Irr0.data
    cdef size_t _num_array_light_source_Irr0 = len(_namespace['_array_light_source_Irr0'])
    cdef float Irr0_pre
    _var_Irr_post = _namespace["_var_Irr_post"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_light_agg_ProportionalCurrentOpsin_ng_main_Irr = _namespace['_array_light_agg_ProportionalCurrentOpsin_ng_main_Irr']
    cdef float * _array_light_agg_ProportionalCurrentOpsin_ng_main_Irr = <float *> _buf__array_light_agg_ProportionalCurrentOpsin_ng_main_Irr.data
    cdef size_t _num_array_light_agg_ProportionalCurrentOpsin_ng_main_Irr = len(_namespace['_array_light_agg_ProportionalCurrentOpsin_ng_main_Irr'])
    cdef float Irr_post
    _var_N = _namespace["_var_N"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_light_prop_ProportionalCurrentOpsin_ng_main_N = _namespace['_array_light_prop_ProportionalCurrentOpsin_ng_main_N']
    cdef int32_t * _array_light_prop_ProportionalCurrentOpsin_ng_main_N = <int32_t *> _buf__array_light_prop_ProportionalCurrentOpsin_ng_main_N.data
    cdef int32_t N = _namespace["N"]
    _var_N_post = _namespace["_var_N_post"]
    cdef int64_t N_post = _namespace["N_post"]
    _var_T = _namespace["_var_T"]
    cdef object _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_T_capsule = _namespace["_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_T_capsule"]
    cdef DynamicArray1DCpp[float]* _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_T_ptr = <DynamicArray1DCpp[float]*>PyCapsule_GetPointer(_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_T_capsule, "DynamicArray1D")
    cdef float* _array_light_prop_ProportionalCurrentOpsin_ng_main_T = <float*> _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_T_ptr.get_data_ptr()
    cdef size_t _num_array_light_prop_ProportionalCurrentOpsin_ng_main_T = len(_namespace['_array_light_prop_ProportionalCurrentOpsin_ng_main_T'])
    cdef float T
    _var__postsynaptic_idx = _namespace["_var__postsynaptic_idx"]
    cdef object _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post_capsule = _namespace["_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post_capsule, "DynamicArray1D")
    cdef int32_t* _array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post = <int32_t*> _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post_ptr.get_data_ptr()
    cdef size_t _num_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post = len(_namespace['_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post'])
    cdef int32_t _postsynaptic_idx
    _var__presynaptic_idx = _namespace["_var__presynaptic_idx"]
    cdef object _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre_capsule = _namespace["_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre_capsule, "DynamicArray1D")
    cdef int32_t* _array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre = <int32_t*> _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre_ptr.get_data_ptr()
    cdef size_t _num_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre = len(_namespace['_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre'])
    cdef int32_t _presynaptic_idx
    _var__synaptic_post = _namespace["_var__synaptic_post"]
    _var_epsilon = _namespace["_var_epsilon"]
    cdef object _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon_capsule = _namespace["_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon_capsule"]
    cdef DynamicArray1DCpp[float]* _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon_ptr = <DynamicArray1DCpp[float]*>PyCapsule_GetPointer(_dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon_capsule, "DynamicArray1D")
    cdef float* _array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon = <float*> _dynamic_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon_ptr.get_data_ptr()
    cdef size_t _num_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon = len(_namespace['_array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon'])
    cdef float epsilon
    cdef float _synaptic_var

    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    cdef int _target_idx

    # Set all the target variable values to zero
    for _target_idx in range(N_post):
        _array_light_agg_ProportionalCurrentOpsin_ng_main_Irr[_target_idx + 0] = 0

    # scalar code
    _vectorisation_idx = 1
        


    for _idx in range(_array_light_prop_ProportionalCurrentOpsin_ng_main_N[0]):
        # vector_code
        vectorisation_idx = _idx
                
        _presynaptic_idx = _array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_pre[_idx]
        Irr0_pre = _array_light_source_Irr0[_presynaptic_idx]
        T = _array_light_prop_ProportionalCurrentOpsin_ng_main_T[_idx]
        epsilon = _array_light_prop_ProportionalCurrentOpsin_ng_main_epsilon[_idx]
        _synaptic_var = (epsilon * T) * Irr0_pre

        _array_light_agg_ProportionalCurrentOpsin_ng_main_Irr[_array_light_prop_ProportionalCurrentOpsin_ng_main__synaptic_post[_idx]] += _synaptic_var