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
from cpython.pycapsule cimport PyCapsule_GetPointer
# We declare minimal C++ interface here - only methods we actually want to use
# This avoids importing the full SpikeQueue wrapper and its dependencies
cdef extern from "spikequeue.h":
    cdef cppclass CSpikeQueue:
        void push(int32_t *, int)

def main(_namespace):
    cdef size_t _idx
    cdef size_t _vectorisation_idx
        
    _queue_capsule = _namespace["_queue_capsule"]
    _var__spikespace = _namespace["_var__spikespace"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_ng_main__spikespace = _namespace['_array_ng_main__spikespace']
    cdef int32_t * _array_ng_main__spikespace = <int32_t *> _buf__array_ng_main__spikespace.data
    cdef size_t _num_array_ng_main__spikespace = len(_namespace['_array_ng_main__spikespace'])
    cdef int32_t _spikespace

    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    cdef int spike_count
    # Get the spike count from the last entry in the spike buffer
    spike_count = _array_ng_main__spikespace[_num_array_ng_main__spikespace-1]

    # Recover the C++ spike queue object from the PyCapsule passed at runtime
    cdef object capsule = _queue_capsule
    cdef CSpikeQueue* cpp_queue = <CSpikeQueue*>PyCapsule_GetPointer(capsule, "CSpikeQueue")

    if spike_count > 0:
    # Optimized: avoid the push_spikes method call overhead
      cpp_queue.push(_array_ng_main__spikespace,spike_count)