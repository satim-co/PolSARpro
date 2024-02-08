'''
processing.py
====================================================================
*******************************************************************************
PolSARpro v5.0 is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 (1991) of the License, or any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. 

See the GNU General Public License (Version 2, 1991) for more details.

********************************************************************************
File   : processing.c
Project  : ESA_POLSARPRO
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Translate to python: Krzysztof Smaza
Version  : 1.0
Creation : 09/2003
Update  :

-------------------------------------------------------------------------------
INSTITUT D'ELECTRONIQUE et de TELECOMMUNICATIONS de RENNES (I.E.T.R)
UMR CNRS 6164
Groupe Image et Teledetection
Equipe SAPHIR (SAr Polarimetrie Holographie Interferometrie Radargrammetrie)
UNIVERSITE DE RENNES I
Pôle Micro-Ondes Radar
Bât. 11D - Campus de Beaulieu
263 Avenue Général Leclerc
35042 RENNES Cedex
Tel :(+33) 2 23 23 57 63
Fax :(+33) 2 23 23 69 63
e-mail : eric.pottier@univ-rennes1.fr, laurent.ferro-famil@univ-rennes1.fr
-------------------------------------------------------------------------------
'''

import math
import numpy
import numba
import itertools
from . import util
from . import matrix


@numba.njit(parallel=False)
def median_array(array, n):
    '''
    Routine  : median_array
    Authors  :
      This Quickselect routine is based on the algorithm described in
      "Numerical recipes in C", Second Edition,
      Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
      This code by Nicolas Devillard - 1998. Public domain.
    Creation : 11/2005
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the median value of an array of size n
    *-------------------------------------------------------------------------------
    Inputs arguments :
    arr : float array
    n : size of the array
    Returned values  :
    median : median value
    '''
    medianval = 0.
    arr = matrix.vector_float(n)
    npts = 0
    # Check NaN and Inf values
    for ll in range(n):
        if util.my_isfinite(array[ll]) is True:
            arr[npts] = array[ll]
            npts += 1

    low = 0
    high = npts - 1
    median = (low + high) / 2
    for i in itertools.count(0):
        if high <= low:  # One element only
            if npts & 1:
                medianval = arr[median]
            else:
                medianval = (arr[median] + arr[median + 1]) / 2
            return medianval

        if high == low + 1:  # Two elements only
            if arr[low] > arr[high]:
                arr[low], arr[high] = arr[high], arr[low]
            if npts & 1:
                medianval = arr[median]
            else:
                medianval = (arr[median] + arr[median + 1]) / 2
            return medianval

        # Find median of low, middle and high items; swap into position low
        middle = (low + high) / 2
        if arr[middle] > arr[high]:
            arr[middle], arr[high] = arr[high], arr[middle]
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[middle] > arr[low]:
            arr[middle], arr[low] = arr[low], arr[middle]

        # Swap low item (now in position middle) into position (low+1)
        arr[middle], arr[low + 1] = arr[low + 1], arr[middle]

        # Nibble from each end towards middle, swapping items when stuck
        ll = low + 1
        hh = high
        for j in itertools.count(0):
            for k in itertools.count(0):
                ll += 1
                if arr[low] > arr[ll]:
                    continue
                break
            for k in itertools.count(0):
                hh -= 1
                if arr[hh] > arr[low]:
                    continue
                break

            if hh < ll:
                break
            arr[ll], arr[hh] = arr[hh], arr[ll]

        # Swap middle item (in position low) back into correct position
        arr[low], arr[hh] = arr[hh], arr[low]

        # Re-set active partition
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1


@numba.njit(parallel=False, fastmath=True)
def diagonalisation(matrix_dim, hm, eigen_vect, eigen_val):
    '''
    Routine  : diagonalisation
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Translate to python: Krzysztof Smaza
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the eigenvectors and eigenvalues of a N*N hermitian
    matrix (with N < 10)
    *-------------------------------------------------------------------------------
    Inputs arguments :
    MatrixDim : Dimension of the Hermitian Matrix (N)
    HermitianMatrix : N*N*2 complex hermitian matrix
    Returned values  :
    EigenVect : N*N*2 complex eigenvector matrix
    EigenVal  : N elements eigenvalue real vector
    '''

    a = matrix.matrix3d_float(10, 10, 2)
    v = matrix.matrix3d_float(10, 10, 2)
    d = matrix.vector_float(10)
    z = matrix.vector_float(10)
    # double b[10];
    w = matrix.vector_float(2)
    s = matrix.vector_float(2)
    c = matrix.vector_float(2)
    titi = matrix.vector_float(2)
    gc = matrix.vector_float(2)
    hc = matrix.vector_float(2)
    # double sm, tresh, x, toto, e, f, g, h, r, d1, d2
    # int n, pp, qq
    # int ii, i, j, k

    n = matrix_dim
    pp = 0
    qq = 0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            a[i][j][0] = hm[i - 1][j - 1][0]
            a[i][j][1] = hm[i - 1][j - 1][1]
            v[i][j][0] = 0.
            v[i][j][1] = 0.
        v[i][i][0] = 1.
        v[i][i][1] = 0.

    for pp in range(1, n + 1):
        d[pp] = a[pp][pp][0]
        # b[pp] = d[pp]
        z[pp] = 0.

    for ii in range(1, 1000 * n * n):
        sm = 0.
        for pp in range(1, n):
            for qq in range(pp + 1, n + 1):
                sm = sm + 2. * math.sqrt(a[pp][qq][0] * a[pp][qq][0] + a[pp][qq][1] * a[pp][qq][1])
        sm = sm / (n * (n - 1))
        if sm < 1.E-16:
            break  # goto Sortie;
        tresh = 1.E-17
        if ii < 4:
            tresh = 0.2 * sm / (n * n)
        x = -1.E-15
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                toto = math.sqrt(a[i][j][0] * a[i][j][0] + a[i][j][1] * a[i][j][1])
                if x < toto:
                    x = toto
                    pp = i
                    qq = j

        toto = math.sqrt(a[pp][qq][0] * a[pp][qq][0] + a[pp][qq][1] * a[pp][qq][1])
        if toto > tresh:
            e = d[pp] - d[qq]
            w[0] = a[pp][qq][0]
            w[1] = a[pp][qq][1]
            g = math.sqrt(w[0] * w[0] + w[1] * w[1])
            g = g * g
            f = math.sqrt(e * e + 4. * g)
            d1 = e + f
            d2 = e - f
            if math.fabs(d2) > math.fabs(d1):
                d1 = d2
            r = math.fabs(d1) / math.sqrt(d1 * d1 + 4. * g)
            s[0] = r
            s[1] = 0.
            titi[0] = 2. * r / d1
            titi[1] = 0.
            c[0] = titi[0] * w[0] - titi[1] * w[1]
            c[1] = titi[0] * w[1] + titi[1] * w[0]
            r = math.sqrt(s[0] * s[0] + s[1] * s[1])
            r = r * r
            h = (d1 / 2. + 2. * g / d1) * r
            d[pp] = d[pp] - h
            z[pp] = z[pp] - h
            d[qq] = d[qq] + h
            z[qq] = z[qq] + h
            a[pp][qq][0] = 0.
            a[pp][qq][1] = 0.

            for j in range(1, pp):
                gc[0] = a[j][pp][0]
                gc[1] = a[j][pp][1]
                hc[0] = a[j][qq][0]
                hc[1] = a[j][qq][1]
                a[j][pp][0] = c[0] * gc[0] - c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1]
                a[j][pp][1] = c[0] * gc[1] + c[1] * gc[0] - s[0] * hc[1] + s[1] * hc[0]
                a[j][qq][0] = s[0] * gc[0] - s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1]
                a[j][qq][1] = s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0]

            for j in range(pp + 1, qq):
                gc[0] = a[pp][j][0]
                gc[1] = a[pp][j][1]
                hc[0] = a[j][qq][0]
                hc[1] = a[j][qq][1]
                a[pp][j][0] = c[0] * gc[0] + c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1]
                a[pp][j][1] = c[0] * gc[1] - c[1] * gc[0] + s[0] * hc[1] - s[1] * hc[0]
                a[j][qq][0] = s[0] * gc[0] + s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1]
                a[j][qq][1] = -s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0]

            for j in range(qq + 1, n + 1):
                gc[0] = a[pp][j][0]
                gc[1] = a[pp][j][1]
                hc[0] = a[qq][j][0]
                hc[1] = a[qq][j][1]
                a[pp][j][0] = c[0] * gc[0] + c[1] * gc[1] - s[0] * hc[0] + s[1] * hc[1]
                a[pp][j][1] = c[0] * gc[1] - c[1] * gc[0] - s[0] * hc[1] - s[1] * hc[0]
                a[qq][j][0] = s[0] * gc[0] + s[1] * gc[1] + c[0] * hc[0] - c[1] * hc[1]
                a[qq][j][1] = s[0] * gc[1] - s[1] * gc[0] + c[0] * hc[1] + c[1] * hc[0]

            for j in range(1, n + 1):
                gc[0] = v[j][pp][0]
                gc[1] = v[j][pp][1]
                hc[0] = v[j][qq][0]
                hc[1] = v[j][qq][1]
                v[j][pp][0] = c[0] * gc[0] - c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1]
                v[j][pp][1] = c[0] * gc[1] + c[1] * gc[0] - s[0] * hc[1] + s[1] * hc[0]
                v[j][qq][0] = s[0] * gc[0] - s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1]
                v[j][qq][1] = s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0]

    #  Sortie
    for k in range(1, n + 1):
        d[k] = 0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                d[k] = d[k] + v[i][k][0] * (hm[i - 1][j - 1][0] * v[j][k][0] - hm[i - 1][j - 1][1] * v[j][k][1])
                d[k] = d[k] + v[i][k][1] * (hm[i - 1][j - 1][0] * v[j][k][1] + hm[i - 1][j - 1][1] * v[j][k][0])

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if d[j] > d[i]:
                x = d[i]
                d[i] = d[j]
                d[j] = x
                for k in range(1, n + 1):
                    c[0] = v[k][i][0]
                    c[1] = v[k][i][1]
                    v[k][i][0] = v[k][j][0]
                    v[k][i][1] = v[k][j][1]
                    v[k][j][0] = c[0]
                    v[k][j][1] = c[1]

    for i in range(n):
        eigen_val[i] = d[i + 1]
        for j in range(n):
            eigen_vect[i][j][0] = v[i + 1][j + 1][0]
            eigen_vect[i][j][1] = v[i + 1][j + 1][1]


@numba.njit(parallel=False)
def inverse_hermitian_matrix2(HM, IHM, eps):
    '''
    Routine  : InverseHermitianMatrix2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the Inverse of a 2x2 Hermitian Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM      : 2*2*2 Hermitian Matrix
    Returned values  :
    IHM     : 2*2*2 Inverse Hermitian Matrix
    '''

    det = matrix.vector_double(2)

    IHM[0][0][0] = HM[1][1][0]
    IHM[0][0][1] = HM[1][1][1]

    IHM[0][1][0] = -HM[0][1][0]
    IHM[0][1][1] = -HM[0][1][1]

    IHM[1][0][0] = -HM[1][0][0]
    IHM[1][0][1] = -HM[1][0][1]

    IHM[1][1][0] = HM[0][0][0]
    IHM[1][1][1] = HM[0][0][1]

    det[0] = math.fabs(HM[0][0][0] * HM[1][1][0] - (HM[0][1][0] * HM[0][1][0] + HM[0][1][1] * HM[0][1][1])) + eps
    det[1] = 0.

    for k in range(2):
        for l in range(2):
            IHM[k][l][0] /= det[0]
            IHM[k][l][1] /= det[0]


@numba.njit(parallel=False)
def trace2_hm1xhm2(HM1, HM2):
    '''
    Routine  : Trace2_HM1xHM2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  computes the trace of the product of 2 2x2 Hermitian Matrices
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM1      : 2*2*2 Hermitian Matrix n°1
    HM2      : 2*2*2 Hermitian Matrix n°2
    Returned values  :
    trace     : trace of the product
    '''
    trace = HM1[0][0][0] * HM2[0][0][0] + HM1[1][1][0] * HM2[1][1][0]
    trace =  trace + 2 * (HM1[0][1][0] * HM2[0][1][0] + HM1[0][1][1] * HM2[0][1][1])
    return trace


@numba.njit(parallel=False)
def determinant_hermitian_matrix2(M, det, eps):
    '''
    Routine  : DeterminantCmplxMatrix2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2007
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the determinant of a 2x2 Complex Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    M      : 2*2*2 Complex Matrix
    Returned values  :
    det     : Complex Determinant of the Complex Matrix
    '''
    det[0] = M[0][0][0] * M[1][1][0] - M[0][0][1] * M[1][1][1]
    det[0] = det[0] - (M[0][1][0] * M[1][0][0] - M[0][1][1] * M[1][0][1]) + eps
    det[1] = M[0][0][0] * M[1][1][1] + M[0][0][1] * M[1][1][0]
    det[1] = det[1] - (M[0][1][0] * M[1][0][1] + M[0][1][1] * M[1][0][0]) + eps


@numba.njit(parallel=False)
def inverse_hermitian_matrix3(HM, IHM):
    '''
    Routine  : InverseHermitianMatrix3
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the Inverse of a 3x3 Hermitian Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM      : 3*3*2 Hermitian Matrix
    Returned values  :
    IHM     : 3*3*2 Inverse Hermitian Matrix
    '''

    det = matrix.vector_float(2)

    IHM[0][0][0] = (HM[1][1][0] * HM[2][2][0] - HM[1][1][1] * HM[2][2][1]) - (HM[1][2][0] * HM[2][1][0] - HM[1][2][1] * HM[2][1][1])
    IHM[0][0][1] = (HM[1][1][0] * HM[2][2][1] + HM[1][1][1] * HM[2][2][0]) - (HM[1][2][0] * HM[2][1][1] + HM[1][2][1] * HM[2][1][0])

    IHM[0][1][0] = -(HM[0][1][0] * HM[2][2][0] - HM[0][1][1] * HM[2][2][1]) + (HM[0][2][0] * HM[2][1][0] - HM[0][2][1] * HM[2][1][1])
    IHM[0][1][1] = -(HM[0][1][0] * HM[2][2][1] + HM[0][1][1] * HM[2][2][0]) + (HM[0][2][0] * HM[2][1][1] + HM[0][2][1] * HM[2][1][0])

    IHM[0][2][0] = (HM[0][1][0] * HM[1][2][0] - HM[0][1][1] * HM[1][2][1]) - (HM[1][1][0] * HM[0][2][0] - HM[1][1][1] * HM[0][2][1])
    IHM[0][2][1] = (HM[0][1][0] * HM[1][2][1] + HM[0][1][1] * HM[1][2][0]) - (HM[1][1][0] * HM[0][2][1] + HM[1][1][1] * HM[0][2][0])

    IHM[1][0][0] = -(HM[1][0][0] * HM[2][2][0] - HM[1][0][1] * HM[2][2][1]) + (HM[2][0][0] * HM[1][2][0] - HM[2][0][1] * HM[1][2][1])
    IHM[1][0][1] = -(HM[1][0][0] * HM[2][2][1] + HM[1][0][1] * HM[2][2][0]) + (HM[2][0][0] * HM[1][2][1] + HM[2][0][1] * HM[1][2][0])

    IHM[1][1][0] = (HM[0][0][0] * HM[2][2][0] - HM[0][0][1] * HM[2][2][1]) - (HM[0][2][0] * HM[2][0][0] - HM[0][2][1] * HM[2][0][1])
    IHM[1][1][1] = (HM[0][0][0] * HM[2][2][1] + HM[0][0][1] * HM[2][2][0]) - (HM[0][2][0] * HM[2][0][1] + HM[0][2][1] * HM[2][0][0])

    IHM[1][2][0] = -(HM[0][0][0] * HM[1][2][0] - HM[0][0][1] * HM[1][2][1]) + (HM[0][2][0] * HM[1][0][0] - HM[0][2][1] * HM[1][0][1])
    IHM[1][2][1] = -(HM[0][0][0] * HM[1][2][1] + HM[0][0][1] * HM[1][2][0]) + (HM[0][2][0] * HM[1][0][1] + HM[0][2][1] * HM[1][0][0])

    IHM[2][0][0] = (HM[1][0][0] * HM[2][1][0] - HM[1][0][1] * HM[2][1][1]) - (HM[1][1][0] * HM[2][0][0] - HM[1][1][1] * HM[2][0][1])
    IHM[2][0][1] = (HM[1][0][0] * HM[2][1][1] + HM[1][0][1] * HM[2][1][0]) - (HM[1][1][0] * HM[2][0][1] + HM[1][1][1] * HM[2][0][0])

    IHM[2][1][0] = -(HM[0][0][0] * HM[2][1][0] - HM[0][0][1] * HM[2][1][1]) + (HM[0][1][0] * HM[2][0][0] - HM[0][1][1] * HM[2][0][1])
    IHM[2][1][1] = -(HM[0][0][0] * HM[2][1][1] + HM[0][0][1] * HM[2][1][0]) + (HM[0][1][0] * HM[2][0][1] + HM[0][1][1] * HM[2][0][0])

    IHM[2][2][0] = (HM[0][0][0] * HM[1][1][0] - HM[0][0][1] * HM[1][1][1]) - (HM[0][1][0] * HM[1][0][0] - HM[0][1][1] * HM[1][0][1])
    IHM[2][2][1] = (HM[0][0][0] * HM[1][1][1] + HM[0][0][1] * HM[1][1][0]) - (HM[0][1][0] * HM[1][0][1] + HM[0][1][1] * HM[1][0][0])

    det[0] = HM[0][0][0] * IHM[0][0][0] - HM[0][0][1] * IHM[0][0][1] + HM[1][0][0] * IHM[0][1][0] - HM[1][0][1] * IHM[0][1][1] + HM[2][0][0] * IHM[0][2][0] - HM[2][0][1] * IHM[0][2][1]
    det[1] = HM[0][0][0] * IHM[0][0][1] + HM[0][0][1] * IHM[0][0][0] + HM[1][0][0] * IHM[0][1][1] + HM[1][0][1] * IHM[0][1][0] + HM[2][0][0] * IHM[0][2][1] + HM[2][0][1] * IHM[0][2][0]

    for k in range(3):
        for l in range(3):
            re = IHM[k][l][0]
            im = IHM[k][l][1]
            IHM[k][l][0] = (re * det[0] + im * det[1]) / (det[0] * det[0] + det[1] * det[1])
            IHM[k][l][1] = (im * det[0] - re * det[1]) / (det[0] * det[0] + det[1] * det[1])


@numba.njit(parallel=False)
def trace3_hm1xhm2(HM1, HM2):
    '''
    Routine  : Trace3_HM1xHM2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  computes the trace of the product of 2 3x3 Hermitian Matrices
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM1      : 3*3*2 Hermitian Matrix n°1
    HM2      : 3*3*2 Hermitian Matrix n°2
    Returned values  :
    trace     : trace of the product
    '''
    trace = HM2[0][0][0] * HM1[0][0][0] - HM2[0][0][1] * HM1[0][0][1]
    trace = trace + HM2[1][1][0] * HM1[1][1][0] - HM2[1][1][1] * HM1[1][1][1]
    trace = trace + HM2[2][2][0] * HM1[2][2][0] - HM2[2][2][1] * HM1[2][2][1]
    trace = trace + 2 * (HM2[0][1][0] * HM1[0][1][0] + HM2[0][1][1] * HM1[0][1][1])
    trace = trace + 2 * (HM2[0][2][0] * HM1[0][2][0] + HM2[0][2][1] * HM1[0][2][1])
    trace = trace + 2 * (HM2[1][2][0] * HM1[1][2][0] + HM2[1][2][1] * HM1[1][2][1])
    return trace


@numba.njit(parallel=False)
def trace4_hm1xhm2(HM1, HM2):
    '''
    Routine  : Trace4_HM1xHM2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  computes the trace of the product of 2 4x4 Hermitian Matrices
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM1      : 4*4*2 Hermitian Matrix n°1
    HM2      : 4*4*2 Hermitian Matrix n°2
    Returned values  :
    trace     : trace of the product
    '''
    trace = HM2[0][0][0] * HM1[0][0][0] - HM2[0][0][1] * HM1[0][0][1]
    trace = trace + HM2[1][1][0] * HM1[1][1][0] - HM2[1][1][1] * HM1[1][1][1]
    trace = trace + HM2[2][2][0] * HM1[2][2][0] - HM2[2][2][1] * HM1[2][2][1]
    trace = trace + HM2[3][3][0] * HM1[3][3][0] - HM2[3][3][1] * HM1[3][3][1]
    trace = trace + 2 * (HM2[0][1][0] * HM1[0][1][0] + HM2[0][1][1] * HM1[0][1][1])
    trace = trace + 2 * (HM2[0][2][0] * HM1[0][2][0] + HM2[0][2][1] * HM1[0][2][1])
    trace = trace + 2 * (HM2[0][3][0] * HM1[0][3][0] + HM2[0][3][1] * HM1[0][3][1])
    trace = trace + 2 * (HM2[1][2][0] * HM1[1][2][0] + HM2[1][2][1] * HM1[1][2][1])
    trace = trace + 2 * (HM2[1][3][0] * HM1[1][3][0] + HM2[1][3][1] * HM1[1][3][1])
    trace = trace + 2 * (HM2[2][3][0] * HM1[2][3][0] + HM2[2][3][1] * HM1[2][3][1])
    return trace


@numba.njit(parallel=False)
def determinant_hermitian_matrix3(HM, det, eps):
    '''
    fRoutine  : DeterminantHermitianMatrix3
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the determinant of a 3x3 Hermitian Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM      : 3*3*2 Hermitian Matrix
    Returned values  :
    det     : Complex Determinant of the Hermitian Matrix
    '''
    IHM = matrix.matrix3d_float(3, 3, 2)

    IHM[0][0][0] = (HM[1][1][0] * HM[2][2][0] - HM[1][1][1] * HM[2][2][1]) - (HM[1][2][0] * HM[2][1][0] - HM[1][2][1] * HM[2][1][1])
    IHM[0][0][1] = (HM[1][1][0] * HM[2][2][1] + HM[1][1][1] * HM[2][2][0]) - (HM[1][2][0] * HM[2][1][1] + HM[1][2][1] * HM[2][1][0])

    IHM[0][1][0] = -(HM[0][1][0] * HM[2][2][0] - HM[0][1][1] * HM[2][2][1]) + (HM[0][2][0] * HM[2][1][0] - HM[0][2][1] * HM[2][1][1])
    IHM[0][1][1] = -(HM[0][1][0] * HM[2][2][1] + HM[0][1][1] * HM[2][2][0]) + (HM[0][2][0] * HM[2][1][1] + HM[0][2][1] * HM[2][1][0])

    IHM[0][2][0] = (HM[0][1][0] * HM[1][2][0] - HM[0][1][1] * HM[1][2][1]) - (HM[1][1][0] * HM[0][2][0] - HM[1][1][1] * HM[0][2][1])
    IHM[0][2][1] = (HM[0][1][0] * HM[1][2][1] + HM[0][1][1] * HM[1][2][0]) - (HM[1][1][0] * HM[0][2][1] + HM[1][1][1] * HM[0][2][0])

    IHM[1][0][0] = -(HM[1][0][0] * HM[2][2][0] - HM[1][0][1] * HM[2][2][1]) + (HM[2][0][0] * HM[1][2][0] - HM[2][0][1] * HM[1][2][1])
    IHM[1][0][1] = -(HM[1][0][0] * HM[2][2][1] + HM[1][0][1] * HM[2][2][0]) + (HM[2][0][0] * HM[1][2][1] + HM[2][0][1] * HM[1][2][0])

    IHM[1][1][0] = (HM[0][0][0] * HM[2][2][0] - HM[0][0][1] * HM[2][2][1]) - (HM[0][2][0] * HM[2][0][0] - HM[0][2][1] * HM[2][0][1])
    IHM[1][1][1] = (HM[0][0][0] * HM[2][2][1] + HM[0][0][1] * HM[2][2][0]) - (HM[0][2][0] * HM[2][0][1] + HM[0][2][1] * HM[2][0][0])

    IHM[1][2][0] = -(HM[0][0][0] * HM[1][2][0] - HM[0][0][1] * HM[1][2][1]) + (HM[0][2][0] * HM[1][0][0] - HM[0][2][1] * HM[1][0][1])
    IHM[1][2][1] = -(HM[0][0][0] * HM[1][2][1] + HM[0][0][1] * HM[1][2][0]) + (HM[0][2][0] * HM[1][0][1] + HM[0][2][1] * HM[1][0][0])

    IHM[2][0][0] = (HM[1][0][0] * HM[2][1][0] - HM[1][0][1] * HM[2][1][1]) - (HM[1][1][0] * HM[2][0][0] - HM[1][1][1] * HM[2][0][1])
    IHM[2][0][1] = (HM[1][0][0] * HM[2][1][1] + HM[1][0][1] * HM[2][1][0]) - (HM[1][1][0] * HM[2][0][1] + HM[1][1][1] * HM[2][0][0])

    IHM[2][1][0] = -(HM[0][0][0] * HM[2][1][0] - HM[0][0][1] * HM[2][1][1]) + (HM[0][1][0] * HM[2][0][0] - HM[0][1][1] * HM[2][0][1])
    IHM[2][1][1] = -(HM[0][0][0] * HM[2][1][1] + HM[0][0][1] * HM[2][1][0]) + (HM[0][1][0] * HM[2][0][1] + HM[0][1][1] * HM[2][0][0])

    IHM[2][2][0] = (HM[0][0][0] * HM[1][1][0] - HM[0][0][1] * HM[1][1][1]) - (HM[0][1][0] * HM[1][0][0] - HM[0][1][1] * HM[1][0][1])
    IHM[2][2][1] = (HM[0][0][0] * HM[1][1][1] + HM[0][0][1] * HM[1][1][0]) - (HM[0][1][0] * HM[1][0][1] + HM[0][1][1] * HM[1][0][0])

    det[0] = HM[0][0][0] * IHM[0][0][0] - HM[0][0][1] * IHM[0][0][1] + HM[1][0][0] * IHM[0][1][0] - HM[1][0][1] * IHM[0][1][1] + HM[2][0][0] * IHM[0][2][0] - HM[2][0][1] * IHM[0][2][1]
    det[1] = HM[0][0][0] * IHM[0][0][1] + HM[0][0][1] * IHM[0][0][0] + HM[1][0][0] * IHM[0][1][1] + HM[1][0][1] * IHM[0][1][0] + HM[2][0][0] * IHM[0][2][1] + HM[2][0][1] * IHM[0][2][0]

    if det[0] < eps:
        det[0] = eps
    if det[1] < eps:
        det[1] = eps


@numba.njit(parallel=False)
def inverse_cmplx_matrix2(M, IM, eps):
    '''
    Routine  : InverseCmplxMatrix2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2007
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the Inverse of a 2x2 Complex Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    M      : 2*2*2 Complex Matrix
    Returned values  :
    IM      : 2*2*2 Inverse Complex Matrix
    '''
    det = matrix.vector_float(2)

    IM[0][0][0] = M[1][1][0]
    IM[0][0][1] = M[1][1][1]

    IM[0][1][0] = -M[0][1][0]
    IM[0][1][1] = -M[0][1][1]

    IM[1][0][0] = -M[1][0][0]
    IM[1][0][1] = -M[1][0][1]

    IM[1][1][0] = M[0][0][0]
    IM[1][1][1] = M[0][0][1]

    det[0] = M[0][0][0] * M[1][1][0] - M[0][0][1] * M[1][1][1]
    det[0] = det[0] - (M[0][1][0] * M[1][0][0] - M[0][1][1] * M[1][0][1]) + eps

    det[1] = M[0][0][0] * M[1][1][1] + M[0][0][1] * M[1][1][0]
    det[1] = det[1] - (M[0][1][0] * M[1][0][1] + M[0][1][1] * M[1][0][0]) + eps

    for k in range(2):
        for l in range(2):
            re = IM[k][l][0]
            im = IM[k][l][1]
            IM[k][l][0] = (re * det[0] + im * det[1]) / (det[0] * det[0] + det[1] * det[1])
            IM[k][l][1] = (im * det[0] - re * det[1]) / (det[0] * det[0] + det[1] * det[1])


@numba.njit(parallel=False)
def product_cmplx_matrix(M1, M2, M3, N):
    '''
    Routine  : ProductCmplxMatrix
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2007
    Update  :
    *-------------------------------------------------------------------------------
    Description :  computes the product of 2 NxN Complex Matrices
    *-------------------------------------------------------------------------------
    Inputs arguments :
    M1      : N*N*2 Cmplx Matrix n°1
    M2      : N*N*2 Cmplx Matrix n°2
    Returned values  :
    M3      : N*N*2 Cmplx Matrix n°3 = M1xM2
    '''
    for i in range(N):
        for j in range(N):
            M3[i][j][0] = 0.
            M3[i][j][1] = 0.
            for k in range(N):
                M3[i][j][0] += M1[i][k][0] * M2[k][j][0] - M1[i][k][1] * M2[k][j][1]
                M3[i][j][1] += M1[i][k][0] * M2[k][j][1] + M1[i][k][1] * M2[k][j][0]


@numba.njit(parallel=False)
def pseudo_inverse_hermitian_matrix4(HM, IHM):
    '''
    Routine  : PseudoInverseHermitianMatrix4
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2007
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the Pseudo-Inverse of a 4x4 Hermitian Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM      : 4*4*2 Hermitian Matrix
    Returned values  :
    IHM     : 4*4*2 Pseudo Inverse Hermitian Matrix
    '''

    V = matrix.matrix3d_float(4, 4, 2)
    Vm1 = matrix.matrix3d_float(4, 4, 2)
    VL = matrix.matrix3d_float(4, 4, 2)
    lmda = matrix.vector_float(4)
    Tmp1 = matrix.matrix3d_float(4, 4, 2)
    diagonalisation(4, HM, V, lmda)
    for k in range(4):
        for l in range(4):
            VL[k][l][0] = 0.
            VL[k][l][1] = 0.

    for k in range(4):
        if lmda[k] > 1.E-10:
            VL[k][k][0] = 1. / lmda[k]

    # Transpose Conjugate Matrix
    for k in range(4):
        for l in range(4):
            Vm1[k][l][0] = V[l][k][0]
            Vm1[k][l][1] = -V[l][k][1]
    product_cmplx_matrix(V, VL, Tmp1, 4)
    product_cmplx_matrix(Tmp1, Vm1, IHM, 4)


@numba.njit(parallel=False)
def inverse_hermitian_matrix4(HM, IHM, eps):
    '''
    Routine  : InverseHermitianMatrix4
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the Inverse of a 4x4 Hermitian Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM      : 4*4*2 Hermitian Matrix
    Returned values  :
    IHM     : 4*4*2 Inverse Hermitian Matrix
    '''

    det = matrix.vector_float(2)
    determinant_hermitian_matrix4(HM, det, eps)
    determinant = math.sqrt(det[0] * det[0] + det[1] * det[1])

    if determinant < 1.E-10:
        pseudo_inverse_hermitian_matrix4(HM, IHM)
    else:
        A = matrix.matrix3d_float(2, 2, 2)
        B = matrix.matrix3d_float(2, 2, 2)
        C = matrix.matrix3d_float(2, 2, 2)
        D = matrix.matrix3d_float(2, 2, 2)
        Am1 = matrix.matrix3d_float(2, 2, 2)
        Dm1 = matrix.matrix3d_float(2, 2, 2)
        Q = matrix.matrix3d_float(2, 2, 2)
        Qm1 = matrix.matrix3d_float(2, 2, 2)
        Tmp1 = matrix.matrix3d_float(2, 2, 2)
        Tmp2 = matrix.matrix3d_float(2, 2, 2)

        A[0][0][0] = HM[0][0][0]
        A[0][0][1] = HM[0][0][1]
        A[0][1][0] = HM[0][1][0]
        A[0][1][1] = HM[0][1][1]
        A[1][0][0] = HM[1][0][0]
        A[1][0][1] = HM[1][0][1]
        A[1][1][0] = HM[1][1][0]
        A[1][1][1] = HM[1][1][1]
        B[0][0][0] = HM[0][2][0]
        B[0][0][1] = HM[0][2][1]
        B[0][1][0] = HM[0][3][0]
        B[0][1][1] = HM[0][3][1]
        B[1][0][0] = HM[1][2][0]
        B[1][0][1] = HM[1][2][1]
        B[1][1][0] = HM[1][3][0]
        B[1][1][1] = HM[1][3][1]
        C[0][0][0] = HM[2][0][0]
        C[0][0][1] = HM[2][0][1]
        C[0][1][0] = HM[2][1][0]
        C[0][1][1] = HM[2][1][1]
        C[1][0][0] = HM[3][0][0]
        C[1][0][1] = HM[3][0][1]
        C[1][1][0] = HM[3][1][0]
        C[1][1][1] = HM[3][1][1]
        D[0][0][0] = HM[2][2][0]
        D[0][0][1] = HM[2][2][1]
        D[0][1][0] = HM[2][3][0]
        D[0][1][1] = HM[2][3][1]
        D[1][0][0] = HM[3][2][0]
        D[1][0][1] = HM[3][2][1]
        D[1][1][0] = HM[3][3][0]
        D[1][1][1] = HM[3][3][1]

        inverse_cmplx_matrix2(A, Am1, eps)
        inverse_cmplx_matrix2(D, Dm1, eps)

        product_cmplx_matrix(B, Dm1, Tmp1, 2)
        product_cmplx_matrix(Tmp1, C, Tmp2, 2)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Q[i][j][k] = A[i][j][k] - Tmp2[i][j][k]

        inverse_cmplx_matrix2(Q, Qm1, eps)

        IHM[0][0][0] = Qm1[0][0][0]
        IHM[0][0][1] = Qm1[0][0][1]
        IHM[0][1][0] = Qm1[0][1][0]
        IHM[0][1][1] = Qm1[0][1][1]
        IHM[1][0][0] = Qm1[1][0][0]
        IHM[1][0][1] = Qm1[1][0][1]
        IHM[1][1][0] = Qm1[1][1][0]
        IHM[1][1][1] = Qm1[1][1][1]

        product_cmplx_matrix(Qm1, B, Tmp1, 2)
        product_cmplx_matrix(Tmp1, Dm1, Tmp2, 2)

        IHM[0][2][0] = -Tmp2[0][0][0]
        IHM[0][2][1] = -Tmp2[0][0][1]
        IHM[0][3][0] = -Tmp2[0][1][0]
        IHM[0][3][1] = -Tmp2[0][1][1]
        IHM[1][2][0] = -Tmp2[1][0][0]
        IHM[1][2][1] = -Tmp2[1][0][1]
        IHM[1][3][0] = -Tmp2[1][1][0]
        IHM[1][3][1] = -Tmp2[1][1][1]

        product_cmplx_matrix(C, Tmp2, Tmp1, 2)
        Tmp1[0][0][0] = Tmp1[0][0][0] + 1.
        Tmp1[1][1][0] = Tmp1[1][1][0] + 1.
        product_cmplx_matrix(Dm1, Tmp1, Tmp2, 2)

        IHM[2][2][0] = Tmp2[0][0][0]
        IHM[2][2][1] = Tmp2[0][0][1]
        IHM[2][3][0] = Tmp2[0][1][0]
        IHM[2][3][1] = Tmp2[0][1][1]
        IHM[3][2][0] = Tmp2[1][0][0]
        IHM[3][2][1] = Tmp2[1][0][1]
        IHM[3][3][0] = Tmp2[1][1][0]
        IHM[3][3][1] = Tmp2[1][1][1]

        product_cmplx_matrix(Dm1, C, Tmp1, 2)
        product_cmplx_matrix(Tmp1, Qm1, Tmp2, 2)

        IHM[2][0][0] = -Tmp2[0][0][0]
        IHM[2][0][1] = -Tmp2[0][0][1]
        IHM[2][1][0] = -Tmp2[0][1][0]
        IHM[2][1][1] = -Tmp2[0][1][1]
        IHM[3][0][0] = -Tmp2[1][0][0]
        IHM[3][0][1] = -Tmp2[1][0][1]
        IHM[3][1][0] = -Tmp2[1][1][0]
        IHM[3][1][1] = -Tmp2[1][1][1]


@numba.njit(parallel=False)
def determinant_cmplx_matrix2(M, det, eps):
    '''
    Routine  : DeterminantCmplxMatrix2
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2007
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the determinant of a 2x2 Complex Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    M      : 2*2*2 Complex Matrix
    Returned values  :
    det     : Complex Determinant of the Complex Matrix
    '''
    det[0] = M[0][0][0] * M[1][1][0] - M[0][0][1] * M[1][1][1]
    det[0] = det[0] - (M[0][1][0] * M[1][0][0] - M[0][1][1] * M[1][0][1]) + eps

    det[1] = M[0][0][0] * M[1][1][1] + M[0][0][1] * M[1][1][0]
    det[1] = det[1] - (M[0][1][0] * M[1][0][1] + M[0][1][1] * M[1][0][0]) + eps


@numba.njit(parallel=False)
def determinant_hermitian_matrix4(HM, det, eps):
    '''
    Routine  : DeterminantHermitianMatrix4
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *-------------------------------------------------------------------------------
    Description :  Computes the determinant of a 4x4 Hermitian Matrix
    *-------------------------------------------------------------------------------
    Inputs arguments :
    HM      : 4*4*4 Hermitian Matrix
    Returned values  :
    det      : Complex Determinant of the Hermitian Matrix
    '''
    A = matrix.matrix3d_float(2, 2, 2)
    B = matrix.matrix3d_float(2, 2, 2)
    C = matrix.matrix3d_float(2, 2, 2)
    D = matrix.matrix3d_float(2, 2, 2)
    Am1 = matrix.matrix3d_float(2, 2, 2)
    P = matrix.matrix3d_float(2, 2, 2)
    Tmp1 = matrix.matrix3d_float(2, 2, 2)
    Tmp2 = matrix.matrix3d_float(2, 2, 2)
    det1 = matrix.vector_float(2)
    det2 = matrix.vector_float(2)

    A[0][0][0] = HM[0][0][0]
    A[0][0][1] = HM[0][0][1]
    A[0][1][0] = HM[0][1][0]
    A[0][1][1] = HM[0][1][1]
    A[1][0][0] = HM[1][0][0]
    A[1][0][1] = HM[1][0][1]
    A[1][1][0] = HM[1][1][0]
    A[1][1][1] = HM[1][1][1]
    B[0][0][0] = HM[0][2][0]
    B[0][0][1] = HM[0][2][1]
    B[0][1][0] = HM[0][3][0]
    B[0][1][1] = HM[0][3][1]
    B[1][0][0] = HM[1][2][0]
    B[1][0][1] = HM[1][2][1]
    B[1][1][0] = HM[1][3][0]
    B[1][1][1] = HM[1][3][1]
    C[0][0][0] = HM[2][0][0]
    C[0][0][1] = HM[2][0][1]
    C[0][1][0] = HM[2][1][0]
    C[0][1][1] = HM[2][1][1]
    C[1][0][0] = HM[3][0][0]
    C[1][0][1] = HM[3][0][1]
    C[1][1][0] = HM[3][1][0]
    C[1][1][1] = HM[3][1][1]
    D[0][0][0] = HM[2][2][0]
    D[0][0][1] = HM[2][2][1]
    D[0][1][0] = HM[2][3][0]
    D[0][1][1] = HM[2][3][1]
    D[1][0][0] = HM[3][2][0]
    D[1][0][1] = HM[3][2][1]
    D[1][1][0] = HM[3][3][0]
    D[1][1][1] = HM[3][3][1]

    inverse_cmplx_matrix2(A, Am1, eps)

    product_cmplx_matrix(C, Am1, Tmp1, 2)
    product_cmplx_matrix(Tmp1, B, Tmp2, 2)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                P[i][j][k] = D[i][j][k] - Tmp2[i][j][k]

    determinant_cmplx_matrix2(A, det1, eps)
    determinant_cmplx_matrix2(P, det2, eps)

    det[0] = det1[0] * det2[0] - det1[1] * det2[1]
    det[1] = det1[0] * det2[1] + det1[1] * det2[0]

    if det[0] < eps:
        det[0] = eps
    if det[1] < eps:
        det[1] = eps


