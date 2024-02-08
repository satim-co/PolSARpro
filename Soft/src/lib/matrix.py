'''
*-------------------------------------------------------------------------------

Description :  MATRICES Routines

*-------------------------------------------------------------------------------
'''

import numpy
import numba


@numba.njit(parallel=False)
def vector_char(nrh):
    return numpy.zeros(nrh + 1, dtype=numpy.byte)


@numba.njit(parallel=False)
def vector_float(nrh):
    return numpy.zeros(nrh + 1, dtype=numpy.float32)


@numba.njit(parallel=False)
def vector_double(nrh):
    return numpy.zeros(nrh + 1, dtype=numpy.double)


@numba.njit(parallel=False)
def vector_int(nrh):
    return numpy.zeros(nrh + 1, dtype=numpy.int32)


@numba.njit(parallel=False)
def matrix_float(nrh, nch):
    return numpy.zeros((nrh, nch), dtype=numpy.float32)


@numba.njit(parallel=False)
def matrix3d_float(nz, nrh, nch):
    return numpy.zeros((nz + 1, nrh + 1, nch + 1), dtype=numpy.float32)
