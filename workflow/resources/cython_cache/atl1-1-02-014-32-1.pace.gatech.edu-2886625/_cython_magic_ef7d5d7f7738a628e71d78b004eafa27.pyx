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
        
    _var_I_in = _namespace["_var_I_in"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_I_in = _namespace['_array_ng_main_I_in']
    cdef float * _array_ng_main_I_in = <float *> _buf__array_ng_main_I_in.data
    cdef size_t _num_array_ng_main_I_in = len(_namespace['_array_ng_main_I_in'])
    cdef float I_in
    _var_Iopto = _namespace["_var_Iopto"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_Iopto = _namespace['_array_ng_main_Iopto']
    cdef float * _array_ng_main_Iopto = <float *> _buf__array_ng_main_Iopto.data
    cdef size_t _num_array_ng_main_Iopto = len(_namespace['_array_ng_main_Iopto'])
    cdef float Iopto
    _var_J0fbars = _namespace["_var_J0fbars"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_J0fbars = _namespace['_array_ng_main_J0fbars']
    cdef float * _array_ng_main_J0fbars = <float *> _buf__array_ng_main_J0fbars.data
    cdef size_t _num_array_ng_main_J0fbars = len(_namespace['_array_ng_main_J0fbars'])
    cdef float J0fbars
    _var_J0fs = _namespace["_var_J0fs"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_J0fs = _namespace['_array_ng_main_J0fs']
    cdef float * _array_ng_main_J0fs = <float *> _buf__array_ng_main_J0fs.data
    cdef size_t _num_array_ng_main_J0fs = len(_namespace['_array_ng_main_J0fs'])
    cdef float J0fs
    _var_J0ss = _namespace["_var_J0ss"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_J0ss = _namespace['_array_ng_main_J0ss']
    cdef float * _array_ng_main_J0ss = <float *> _buf__array_ng_main_J0ss.data
    cdef size_t _num_array_ng_main_J0ss = len(_namespace['_array_ng_main_J0ss'])
    cdef float J0ss
    _var_N = _namespace["_var_N"]
    cdef int64_t N = _namespace["N"]
    _var_dt = _namespace["_var_dt"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_defaultclock_dt = _namespace['_array_defaultclock_dt']
    cdef double * _array_defaultclock_dt = <double *> _buf__array_defaultclock_dt.data
    cdef double dt = _namespace["dt"]
    _var_sf = _namespace["_var_sf"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_sf = _namespace['_array_ng_main_sf']
    cdef float * _array_ng_main_sf = <float *> _buf__array_ng_main_sf.data
    cdef size_t _num_array_ng_main_sf = len(_namespace['_array_ng_main_sf'])
    cdef float sf
    _var_ss = _namespace["_var_ss"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_ss = _namespace['_array_ng_main_ss']
    cdef float * _array_ng_main_ss = <float *> _buf__array_ng_main_ss.data
    cdef size_t _num_array_ng_main_ss = len(_namespace['_array_ng_main_ss'])
    cdef float ss
    _var_tau_f = _namespace["_var_tau_f"]
    cdef double tau_f = _namespace["tau_f"]
    _var_tau_s = _namespace["_var_tau_s"]
    cdef double tau_s = _namespace["tau_s"]
    _var_tau_v = _namespace["_var_tau_v"]
    cdef double tau_v = _namespace["tau_v"]
    _var_uy = _namespace["_var_uy"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_uy = _namespace['_array_ng_main_uy']
    cdef float * _array_ng_main_uy = <float *> _buf__array_ng_main_uy.data
    cdef size_t _num_array_ng_main_uy = len(_namespace['_array_ng_main_uy'])
    cdef float uy
    _var_v = _namespace["_var_v"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_v = _namespace['_array_ng_main_v']
    cdef float * _array_ng_main_v = <float *> _buf__array_ng_main_v.data
    cdef size_t _num_array_ng_main_v = len(_namespace['_array_ng_main_v'])
    cdef float v
    _var_vJsbar = _namespace["_var_vJsbar"]
    cdef _numpy.ndarray[float, ndim=1, mode='c'] _buf__array_ng_main_vJsbar = _namespace['_array_ng_main_vJsbar']
    cdef float * _array_ng_main_vJsbar = <float *> _buf__array_ng_main_vJsbar.data
    cdef size_t _num_array_ng_main_vJsbar = len(_namespace['_array_ng_main_vJsbar'])
    cdef float vJsbar
    cdef float _J0fbars
    cdef float _J0fs
    cdef float _J0ss
    cdef float _lio_1
    cdef float _lio_2
    cdef float _lio_3
    cdef float _lio_4
    cdef float _lio_5
    cdef float _sf
    cdef float _ss
    cdef float _v

    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    # scalar code
    _vectorisation_idx = 1
        
    dt = _array_defaultclock_dt[0]
    _lio_1 = dt / tau_f
    _lio_2 = dt / tau_s
    _lio_3 = (- dt) / tau_f
    _lio_4 = (- dt) / tau_s
    _lio_5 = dt / tau_v


    # vector code
    for _idx in range(N):
        _vectorisation_idx = _idx
                
        I_in = _array_ng_main_I_in[_idx]
        Iopto = _array_ng_main_Iopto[_idx]
        J0fbars = _array_ng_main_J0fbars[_idx]
        J0fs = _array_ng_main_J0fs[_idx]
        J0ss = _array_ng_main_J0ss[_idx]
        sf = _array_ng_main_sf[_idx]
        ss = _array_ng_main_ss[_idx]
        uy = _array_ng_main_uy[_idx]
        v = _array_ng_main_v[_idx]
        vJsbar = _array_ng_main_vJsbar[_idx]
        _J0fbars = (_lio_1 * (- J0fbars)) + J0fbars
        _J0fs = (_lio_1 * (- J0fs)) + J0fs
        _J0ss = (_lio_2 * (- J0ss)) + J0ss
        _sf = (_lio_3 * sf) + sf
        _ss = (_lio_4 * ss) + ss
        _v = (_lio_5 * ((((((I_in + Iopto) + J0fbars) + J0fs) + J0ss) + uy) - (v + vJsbar))) + v
        J0fbars = _J0fbars
        J0fs = _J0fs
        J0ss = _J0ss
        sf = _sf
        ss = _ss
        v = _v
        _array_ng_main_J0fbars[_idx] = J0fbars
        _array_ng_main_J0fs[_idx] = J0fs
        _array_ng_main_J0ss[_idx] = J0ss
        _array_ng_main_sf[_idx] = sf
        _array_ng_main_ss[_idx] = ss
        _array_ng_main_v[_idx] = v