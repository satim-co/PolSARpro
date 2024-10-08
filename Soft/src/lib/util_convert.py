'''
********************************************************************
PolSARpro v5.0 is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 (1991) of
the License, or any later version. This program is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See the GNU General Public License (Version 2, 1991) for more details

*********************************************************************

File   : util_convert.c
Project  : ESA_POLSARPRO
Authors  : Eric POTTIER
Version  : 1.0
Creation : 08/2010
Update  :

--------------------------------------------------------------------
INSTITUT D'ELECTRONIQUE et de TELECOMMUNICATIONS de RENNES (I.E.T.R)
UMR CNRS 6164

Waves and Signal department
SHINE Team


UNIVERSITY OF RENNES I
Bât. 11D - Campus de Beaulieu
263 Avenue Général Leclerc
35042 RENNES Cedex
Tel :(+33) 2 23 23 57 63
Fax :(+33) 2 23 23 69 63
e-mail: eric.pottier@univ-rennes1.fr
    laurent.ferro-famil@univ-rennes1.fr
--------------------------------------------------------------------
'''

import math
import numba
from . import util


@numba.njit(parallel=False, fastmath=True)
def t3_to_c3(M_in, Nlig, Ncol, NwinLig, NwinCol):
    """Description : create an array of the C3 matrix from T3 matrix"""
    # pragma omp parallel for private(col) firstprivate(T11, T12_re, T12_im, T13_re, T13_im, T22, T23_re, T23_im, T33)
    sqrt_2 = math.sqrt(2)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            T11, T12_re, T12_im, T13_re, T13_im, T22, T23_re, T23_im, T33 = [
                M_in[util.T311][lig][col],
                M_in[util.T312_RE][lig][col],
                M_in[util.T312_IM][lig][col],
                M_in[util.T313_RE][lig][col],
                M_in[util.T313_IM][lig][col],
                M_in[util.T322][lig][col],
                M_in[util.T323_RE][lig][col],
                M_in[util.T323_IM][lig][col],
                M_in[util.T333][lig][col]
            ]
            M_in[util.C311][lig][col] = (T11 + 2 * T12_re + T22) / 2
            M_in[util.C312_RE][lig][col] = (T13_re + T23_re) / sqrt_2
            M_in[util.C312_IM][lig][col] = (T13_im + T23_im) / sqrt_2
            M_in[util.C313_RE][lig][col] = (T11 - T22) / 2
            M_in[util.C313_IM][lig][col] = -T12_im
            M_in[util.C322][lig][col] = T33
            M_in[util.C323_RE][lig][col] = (T13_re - T23_re) / sqrt_2
            M_in[util.C323_IM][lig][col] = (-T13_im + T23_im) / sqrt_2
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


@numba.njit(parallel=False, fastmath=True)
def c3_to_t3(m_in, Nlig, Ncol, NwinLig, NwinCol):
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
    sqrt_2 = math.sqrt(2)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            C11 = m_in[util.C311][lig][col]
            C12_re = m_in[util.C312_RE][lig][col]
            C12_im = m_in[util.C312_IM][lig][col]
            C13_re = m_in[util.C313_RE][lig][col]
            C13_im = m_in[util.C313_IM][lig][col]
            C22 = m_in[util.C322][lig][col]
            C23_re = m_in[util.C323_RE][lig][col]
            C23_im = m_in[util.C323_IM][lig][col]
            C33 = m_in[util.C333][lig][col]

            m_in[util.T311][lig][col] = (C11 + 2 * C13_re + C33) / 2
            m_in[util.T312_RE][lig][col] = (C11 - C33) / 2
            m_in[util.T312_IM][lig][col] = -C13_im
            m_in[util.T313_RE][lig][col] = (C12_re + C23_re) / sqrt_2
            m_in[util.T313_IM][lig][col] = (C12_im - C23_im) / sqrt_2
            m_in[util.T322][lig][col] = (C11 - 2 * C13_re + C33) / 2
            m_in[util.T323_RE][lig][col] = (C12_re - C23_re) / sqrt_2
            m_in[util.T323_IM][lig][col] = (C12_im + C23_im) / sqrt_2
            m_in[util.T333][lig][col] = C22
    return 1


@numba.njit(parallel=False, fastmath=True)
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
    sqrt_2 = math.sqrt(2)
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            C11 = M_in[util.C411][lig][col]
            C12_re = (M_in[util.C412_RE][lig][col] + M_in[util.C413_RE][lig][col]) / sqrt_2
            C12_im = (M_in[util.C412_IM][lig][col] + M_in[util.C413_IM][lig][col]) / sqrt_2
            C13_re = M_in[util.C414_RE][lig][col]
            C13_im = M_in[util.C414_IM][lig][col]
            C22 = (M_in[util.C422][lig][col] + M_in[util.C433][lig][col] + 2 * M_in[util.C423_re][lig][col]) / 2
            C23_re = (M_in[util.C424_RE][lig][col] + M_in[util.C434_RE][lig][col]) / sqrt_2
            C23_im = (M_in[util.C424_IM][lig][col] + M_in[util.C434_IM][lig][col]) / sqrt_2
            C33 = M_in[util.C444][lig][col]

            M_in[util.T311][lig][col] = (C11 + 2 * C13_re + C33) / 2
            M_in[util.T312_RE][lig][col] = (C11 - C33) / 2
            M_in[util.T312_IM][lig][col] = -C13_im
            M_in[util.T313_RE][lig][col] = (C12_re + C23_re) / sqrt_2
            M_in[util.T313_IM][lig][col] = (C12_im - C23_im) / sqrt_2
            M_in[util.T322][lig][col] = (C11 - 2 * C13_re + C33) / 2
            M_in[util.T323_RE][lig][col] = (C12_re - C23_re) / sqrt_2
            M_in[util.T323_IM][lig][col] = (C12_im + C23_im) / sqrt_2
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
