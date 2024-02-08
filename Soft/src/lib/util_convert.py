"""
Polsarpro
===
util_convert
"""

# %% [codecell] import
from concurrent.futures import ThreadPoolExecutor
import math
import os
import sys
import struct
import numpy as np
import numba
from . import util

def S2_to_T6(S_in1, S_in2, M_in, Nlig, Ncol, NwinLig, NwinCol):
    hh = 0
    hv = 1
    vh = 2
    vv = 3

    with ThreadPoolExecutor() as executor:
        executor.map(process_s2_to_t6, range(Nlig + NwinLig),
                     [Ncol] * Nlig, [NwinCol] * Nlig,
                     [hh] * Nlig, [hv] * Nlig, [vh] * Nlig, [vv] * Nlig,
                     [S_in1] * Nlig, [S_in2] * Nlig, [M_in] * Nlig)

    return 1


def process_s2_to_t6(lig, Ncol, NwinCol, hh, hv, vh, vv, S_in1, S_in2, M_in):
    for col in range(Ncol + NwinCol):
        k1r = (S_in1[hh][lig][2 * col] + S_in1[vv][lig][2 * col]) / math.sqrt(2.)
        k1i = (S_in1[hh][lig][2 * col + 1] + S_in1[vv][lig][2 * col + 1]) / math.sqrt(2.)
        k2r = (S_in1[hh][lig][2 * col] - S_in1[vv][lig][2 * col]) / math.sqrt(2.)
        k2i = (S_in1[hh][lig][2 * col + 1] - S_in1[vv][lig][2 * col + 1]) / math.sqrt(2.)
        k3r = (S_in1[hv][lig][2 * col] + S_in1[vh][lig][2 * col]) / math.sqrt(2.)
        k3i = (S_in1[hv][lig][2 * col + 1] + S_in1[vh][lig][2 * col + 1]) / math.sqrt(2.)

        k4r = (S_in2[hh][lig][2 * col] + S_in2[vv][lig][2 * col]) / math.sqrt(2.)
        k4i = (S_in2[hh][lig][2 * col + 1] + S_in2[vv][lig][2 * col + 1]) / math.sqrt(2.)
        k5r = (S_in2[hh][lig][2 * col] - S_in2[vv][lig][2 * col]) / math.sqrt(2.)
        k5i = (S_in2[hh][lig][2 * col + 1] - S_in2[vv][lig][2 * col + 1]) / math.sqrt(2.)
        k6r = (S_in2[hv][lig][2 * col] + S_in2[vh][lig][2 * col]) / math.sqrt(2.)
        k6i = (S_in2[hv][lig][2 * col + 1] + S_in2[vh][lig][2 * col + 1]) / math.sqrt(2.)

        M_in[0][lig][col] = k1r * k1r + k1i * k1i


@numba.njit()
def t3_to_c3(M_in, Nlig, Ncol, NwinLig, NwinCol):
    """Description : create an array of the C3 matrix from T3 matrix"""
    #int lig, col;
    #float T11, T12_re, T12_im, T13_re, T13_im;
    #float T22, T23_re, T23_im;
    #float T33;

    #T11 = T12_re = T12_im = T13_re = T13_im = 0.
    #T22 = T23_re = T23_im = T33 = 0.
    #pragma omp parallel for private(col) firstprivate(T11, T12_re, T12_im, T13_re, T13_im, T22, T23_re, T23_im, T33)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            T11 = M_in[util.T311][lig][col]
            T12_re = M_in[util.T312_RE][lig][col]
            T12_im = M_in[util.T312_IM][lig][col]
            T13_re = M_in[util.T313_RE][lig][col]
            T13_im = M_in[util.T313_IM][lig][col]
            T22 = M_in[util.T322][lig][col]
            T23_re = M_in[util.T323_RE][lig][col]
            T23_im = M_in[util.T323_IM][lig][col]
            T33 = M_in[util.T333][lig][col]

            M_in[util.C311][lig][col] = (T11 + 2 * T12_re + T22) / 2
            M_in[util.C312_RE][lig][col] = (T13_re + T23_re) / math.sqrt(2)
            M_in[util.C312_IM][lig][col] = (T13_im + T23_im) / math.sqrt(2)
            M_in[util.C313_RE][lig][col] = (T11 - T22) / 2
            M_in[util.C313_IM][lig][col] = -T12_im
            M_in[util.C322][lig][col] = T33
            M_in[util.C323_RE][lig][col] = (T13_re - T23_re) / math.sqrt(2)
            M_in[util.C323_IM][lig][col] = (-T13_im + T23_im) / math.sqrt(2)
            M_in[util.C333][lig][col] = (T11 - 2 * T12_re + T22) / 2


@numba.njit()
def s2_to_t3(S_in, M_in, Nlig, Ncol, NwinLig, NwinCol):
    '''
    Routine  : s2_to_t3
    Authors  : Eric POTTIER
    Creation : 08/2009
    Update  :
    *--------------------------------------------------------------------
    Description : create an array of the T3 matrix from S2 matrix
    ********************************************************************/
    '''
    hh = 0
    hv = 1
    vh = 2
    vv = 3

    k1r = k1i = k2r = k2i = k3r = k3i = 0.
    # pragma omp parallel for private(col) firstprivate(k1r, k1i, k2r, k2i, k3r, k3i)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            k1r = (S_in[hh][lig][2 * col] + S_in[vv][lig][2 * col]) / math.sqrt(2.)
            k1i = (S_in[hh][lig][2 * col + 1] + S_in[vv][lig][2 * col + 1]) / math.sqrt(2.)
            k2r = (S_in[hh][lig][2 * col] - S_in[vv][lig][2 * col]) / math.sqrt(2.)
            k2i = (S_in[hh][lig][2 * col + 1] - S_in[vv][lig][2 * col + 1]) / math.sqrt(2.)
            k3r = (S_in[hv][lig][2 * col] + S_in[vh][lig][2 * col]) / math.sqrt(2.)
            k3i = (S_in[hv][lig][2 * col + 1] + S_in[vh][lig][2 * col + 1]) / math.sqrt(2.)

            M_in[0][lig][col] = k1r * k1r + k1i * k1i
            M_in[1][lig][col] = k1r * k2r + k1i * k2i
            M_in[2][lig][col] = k1i * k2r - k1r * k2i
            M_in[3][lig][col] = k1r * k3r + k1i * k3i
            M_in[4][lig][col] = k1i * k3r - k1r * k3i
            M_in[5][lig][col] = k2r * k2r + k2i * k2i
            M_in[6][lig][col] = k2r * k3r + k2i * k3i
            M_in[7][lig][col] = k2i * k3r - k2r * k3i
            M_in[8][lig][col] = k3r * k3r + k3i * k3i
    return 1


@numba.njit()
def c3_to_t3(M_in, Nlig, Ncol, NwinLig, NwinCol):
    '''
    Routine  : c3_to_t3
    Authors  : Eric POTTIER
    Creation : 08/2009
    Update  :
    *--------------------------------------------------------------------
    Description : create an array of the T3 matrix from C3 matrix
    ********************************************************************
    '''
    C11 = C12_re = C12_im = C13_re = C13_im = 0.
    C22 = C23_re = C23_im = C33 = 0.
    # pragma omp parallel for private(col) firstprivate(C11, C12_re, C12_im, C13_re, C13_im, C22, C23_re, C23_im, C33)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            C11 = M_in[util.C311][lig][col]
            C12_re = M_in[util.C312_RE][lig][col]
            C12_im = M_in[util.C312_IM][lig][col]
            C13_re = M_in[util.C313_RE][lig][col]
            C13_im = M_in[util.C313_IM][lig][col]
            C22 = M_in[util.C322][lig][col]
            C23_re = M_in[util.C323_RE][lig][col]
            C23_im = M_in[util.C323_IM][lig][col]
            C33 = M_in[util.C333][lig][col]

            M_in[util.T311][lig][col] = (C11 + 2 * C13_re + C33) / 2
            M_in[util.T312_RE][lig][col] = (C11 - C33) / 2
            M_in[util.T312_IM][lig][col] = -C13_im
            M_in[util.T313_RE][lig][col] = (C12_re + C23_re) / math.sqrt(2)
            M_in[util.T313_IM][lig][col] = (C12_im - C23_im) / math.sqrt(2)
            M_in[util.T322][lig][col] = (C11 - 2 * C13_re + C33) / 2
            M_in[util.T323_RE][lig][col] = (C12_re - C23_re) / math.sqrt(2)
            M_in[util.T323_IM][lig][col] = (C12_im + C23_im) / math.sqrt(2)
            M_in[util.T333][lig][col] = C22
    return 1


@numba.njit()
def c4_to_t3(M_in, Nlig, Ncol, NwinLig, NwinCol):
    '''
    Routine  : C4_to_T3
    Authors  : Eric POTTIER
    Creation : 08/2009
    Update  :
    *--------------------------------------------------------------------
    Description : create an array of the T3 matrix from C4 matrix
    ********************************************************************
    '''
    C11 = C12_re = C12_im = C13_re = C13_im = 0.
    C22 = C23_re = C23_im = C33 = 0.
    # pragma omp parallel for private(col) firstprivate(C11, C12_re, C12_im, C13_re, C13_im, C22, C23_re, C23_im, C33)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            C11 = M_in[util.C411][lig][col]
            C12_re = (M_in[util.C412_RE][lig][col] + M_in[util.C413_RE][lig][col]) / math.sqrt(2)
            C12_im = (M_in[util.C412_IM][lig][col] + M_in[util.C413_IM][lig][col]) / math.sqrt(2)
            C13_re = M_in[util.C414_RE][lig][col]
            C13_im = M_in[util.C414_IM][lig][col]
            C22 = (M_in[util.C422][lig][col] + M_in[util.C433][lig][col] + 2 * M_in[util.C423_re][lig][col]) / 2
            C23_re = (M_in[util.C424_RE][lig][col] + M_in[util.C434_RE][lig][col]) / math.sqrt(2)
            C23_im = (M_in[util.C424_IM][lig][col] + M_in[util.C434_IM][lig][col]) / math.sqrt(2)
            C33 = M_in[util.C444][lig][col]

            M_in[util.T311][lig][col] = (C11 + 2 * C13_re + C33) / 2
            M_in[util.T312_RE][lig][col] = (C11 - C33) / 2
            M_in[util.T312_IM][lig][col] = -C13_im
            M_in[util.T313_RE][lig][col] = (C12_re + C23_re) / math.sqrt(2)
            M_in[util.T313_IM][lig][col] = (C12_im - C23_im) / math.sqrt(2)
            M_in[util.T322][lig][col] = (C11 - 2 * C13_re + C33) / 2
            M_in[util.T323_RE][lig][col] = (C12_re - C23_re) / math.sqrt(2)
            M_in[util.T323_IM][lig][col] = (C12_im + C23_im) / math.sqrt(2)
            M_in[util.T333][lig][col] = C22
    return 1


@numba.njit()
def t4_to_t3(M_in, Nlig, Ncol, NwinLig, NwinCol):
    '''
    Routine  : t4_to_t3
    Authors  : Eric POTTIER
    Creation : 08/2009
    Update  :
    *--------------------------------------------------------------------
    Description : create an array of the T3 matrix from T4 matrix
    ********************************************************************
    '''
    T11 = T12_re = T12_im = T13_re = T13_im = 0.
    T22 = T23_re = T23_im = T33 = 0.
    # pragma omp parallel for private(col) firstprivate(T11, T12_re, T12_im, T13_re, T13_im, T22, T23_re, T23_im, T33)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            T11 = M_in[util.T411][lig][col]
            T12_re = M_in[util.T412_RE][lig][col]
            T12_im = M_in[util.T412_IM][lig][col]
            T13_re = M_in[util.T413_RE][lig][col]
            T13_im = M_in[util.T413_IM][lig][col]
            T22 = M_in[util.T422][lig][col]
            T23_re = M_in[util.T423_RE][lig][col]
            T23_im = M_in[util.T423_IM][lig][col]
            T33 = M_in[util.T433][lig][col]

            M_in[util.T311][lig][col] = T11
            M_in[util.T312_RE][lig][col] = T12_re
            M_in[util.T312_IM][lig][col] = T12_im
            M_in[util.T313_RE][lig][col] = T13_re
            M_in[util.T313_IM][lig][col] = T13_im
            M_in[util.T322][lig][col] = T22
            M_in[util.T323_RE][lig][col] = T23_re
            M_in[util.T323_IM][lig][col] = T23_im
            M_in[util.T333][lig][col] = T33
    return 1


