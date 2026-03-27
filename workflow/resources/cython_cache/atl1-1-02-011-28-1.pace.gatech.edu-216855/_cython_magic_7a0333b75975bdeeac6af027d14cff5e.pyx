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

cdef int64_t *_namespace_rand_buffer
cdef int32_t *_namespace_rand_buffer_index

cdef double _rand(int _idx):
    cdef double **buffer_pointer = <double**>_namespace_rand_buffer
    cdef double *buffer = buffer_pointer[0]
    cdef _numpy.ndarray _new_rand

    if(_namespace_rand_buffer_index[0] == 0):
        if buffer != NULL:
            free(buffer)
        _new_rand = _numpy.random.rand(20000)
        buffer = <double *>_numpy.PyArray_DATA(_new_rand)
        PyArray_CLEARFLAGS(<_numpy.PyArrayObject*>_new_rand, _numpy.NPY_ARRAY_OWNDATA)
        buffer_pointer[0] = buffer

    cdef double val = buffer[_namespace_rand_buffer_index[0]]
    _namespace_rand_buffer_index[0] += 1
    if _namespace_rand_buffer_index[0] == 20000:
        _namespace_rand_buffer_index[0] = 0
    return val


# template-specific support code
from libc.string cimport memcpy

cdef int _buffer_size = 1024
cdef int[:] _prebuf = _numpy.zeros(_buffer_size, dtype=_numpy.int32)
cdef int[:] _postbuf = _numpy.zeros(_buffer_size, dtype=_numpy.int32)
cdef int _curbuf = 0
cdef int _raw_pre_idx
cdef int _raw_post_idx

# We now update this function to be a use direct dynamic array pointers
cdef void _flush_buffer(int[:] buf,DynamicArray1DCpp[int32_t]* dynarr, int buf_len):
    cdef size_t _curlen = dynarr.size()
    cdef size_t _newlen = _curlen+buf_len
    # Resize the array
    dynarr.resize(_newlen)
    # Get raw data pointer from C++ array
    cdef int32_t* data_ptr = dynarr.get_data_ptr()

    # Use memcpy for fast bulk copy
    memcpy(&data_ptr[_curlen], &buf[0], buf_len * sizeof(int32_t))


def main(_namespace):
    cdef size_t _idx
    cdef size_t _vectorisation_idx
        
    _var_N = _namespace["_var_N"]
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__array_syn_in_N = _namespace['_array_syn_in_N']
    cdef int32_t * _array_syn_in_N = <int32_t *> _buf__array_syn_in_N.data
    cdef int32_t N = _namespace["N"]
    _var_N_post = _namespace["_var_N_post"]
    cdef int64_t N_post = _namespace["N_post"]
    _var_N_pre = _namespace["_var_N_pre"]
    cdef int64_t N_pre = _namespace["N_pre"]
    cdef bool _cond
    cdef int32_t _iter_high
    cdef int32_t _iter_low
    cdef int32_t _iter_step
    cdef int32_t _j
    cdef int32_t _k
    cdef int32_t _n
    cdef int32_t _post_idx
    cdef int32_t _pre_idx
    cdef int32_t _raw_post_idx
    cdef int32_t _raw_pre_idx
    _var__source_offset = _namespace["_var__source_offset"]
    cdef int64_t _source_offset = _namespace["_source_offset"]
    _var__synaptic_post = _namespace["_var__synaptic_post"]
    cdef object _dynamic_array_syn_in__synaptic_post_capsule = _namespace["_dynamic_array_syn_in__synaptic_post_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_syn_in__synaptic_post_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_syn_in__synaptic_post_capsule, "DynamicArray1D")
    cdef int32_t* _array_syn_in__synaptic_post = <int32_t*> _dynamic_array_syn_in__synaptic_post_ptr.get_data_ptr()
    cdef size_t _num_array_syn_in__synaptic_post = len(_namespace['_array_syn_in__synaptic_post'])
    cdef int32_t _synaptic_post
    _var__synaptic_pre = _namespace["_var__synaptic_pre"]
    cdef object _dynamic_array_syn_in__synaptic_pre_capsule = _namespace["_dynamic_array_syn_in__synaptic_pre_capsule"]
    cdef DynamicArray1DCpp[int32_t]* _dynamic_array_syn_in__synaptic_pre_ptr = <DynamicArray1DCpp[int32_t]*>PyCapsule_GetPointer(_dynamic_array_syn_in__synaptic_pre_capsule, "DynamicArray1D")
    cdef int32_t* _array_syn_in__synaptic_pre = <int32_t*> _dynamic_array_syn_in__synaptic_pre_ptr.get_data_ptr()
    cdef size_t _num_array_syn_in__synaptic_pre = len(_namespace['_array_syn_in__synaptic_pre'])
    cdef int32_t _synaptic_pre
    _var__target_offset = _namespace["_var__target_offset"]
    cdef int64_t _target_offset = _namespace["_target_offset"]
    # namespace for function rand
    global _namespace_rand_buffer
    global _namespace_num_rand_buffer
    cdef _numpy.ndarray[int64_t, ndim=1, mode='c'] _buf__rand_buffer = _namespace['_rand_buffer']
    _namespace_rand_buffer = <int64_t *> _buf__rand_buffer.data
    _namespace_num_rand_buffer = len(_namespace['_rand_buffer'])
    # namespace for function rand
    global _namespace_rand_buffer_index
    global _namespace_num_rand_buffer_index
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _buf__rand_buffer_index = _namespace['_rand_buffer_index']
    _namespace_rand_buffer_index = <int32_t *> _buf__rand_buffer_index.data
    _namespace_num_rand_buffer_index = len(_namespace['_rand_buffer_index'])

    if '_owner' in _namespace:
        _owner = _namespace['_owner']

    cdef int* _prebuf_ptr = &(_prebuf[0])
    cdef int* _postbuf_ptr = &(_postbuf[0])

    global _curbuf

    cdef size_t oldsize = _dynamic_array_syn_in__synaptic_pre_ptr.size()
    cdef size_t newsize

    # The following variables are only used for probabilistic connections


    # scalar code
    _vectorisation_idx = 1
        

        

        

        


    for _i in range(N_pre):
        _raw_pre_idx = _i + _source_offset

                
        _cond = True

        if not _cond:
            continue
                
        _iter_low = 0
        _iter_high = N_post
        _iter_step = 1

        for _k in range(_iter_low, _iter_high, _iter_step):

                        
            _pre_idx = _raw_pre_idx
            _j = _k

            _raw_post_idx = _j + _target_offset

            if _j<0 or _j>=N_post:
                raise IndexError("index j=%d outside allowed range from 0 to %d" % (_j, N_post-1))
                        
            _post_idx = _raw_post_idx
            _n = 1


            for _repetition in range(_n):
                _prebuf_ptr[_curbuf] = _pre_idx
                _postbuf_ptr[_curbuf] = _post_idx
                _curbuf += 1
                # Flush buffer
                if _curbuf==_buffer_size:
                    _flush_buffer(_prebuf, _dynamic_array_syn_in__synaptic_pre_ptr, _curbuf)
                    _flush_buffer(_postbuf, _dynamic_array_syn_in__synaptic_post_ptr, _curbuf)
                    _curbuf = 0

    # Final buffer flush
    _flush_buffer(_prebuf, _dynamic_array_syn_in__synaptic_pre_ptr, _curbuf)
    _flush_buffer(_postbuf, _dynamic_array_syn_in__synaptic_post_ptr, _curbuf)
    _curbuf = 0  # reset the buffer for the next run

    newsize = _dynamic_array_syn_in__synaptic_pre_ptr.size()
    # now we need to resize all registered variables and set the total number
    # of synapse (via Python)
    _owner._resize(newsize)

    # And update N_incoming, N_outgoing and synapse_number
    _owner._update_synapse_numbers(oldsize)