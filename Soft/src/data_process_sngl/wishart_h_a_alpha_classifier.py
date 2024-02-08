#!/usr/bin/env python3

'''
wishart_h_a_alpha_classifier.py
====================================================================
PolSARpro v5.0 is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 (1991) of
the License, or any later version. This program is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See the GNU General Public License (Version 2, 1991) for more details

*********************************************************************

File  : wishart_h_a_alpha_classifier.c
Project  : ESA_POLSARPRO-SATIM
Authors  : Eric POTTIER, Jacek STRZELCZYK
Translate to python: Krzysztof Smaza
Version  : 2.2
Creation : 07/2015
Update  : 2024-02-05
*--------------------------------------------------------------------
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

*--------------------------------------------------------------------

Description :  Unsupervised maximum likelihood classification of a
polarimetric image from the Wishart PDF of its coherency
matrices

********************************************************************/
'''


import os
import sys
import platform
import errno
import numpy
import math
import logging
import datetime
import numba
from functools import partial
from multiprocessing import Pool
from multiprocessing import current_process

sys.path.append(r'../')
import lib.util
import lib.util_block
import lib.util_convert
from lib.processing import trace3_hm1xhm2
from lib.processing import trace4_hm1xhm2
from lib.processing import inverse_hermitian_matrix3
from lib.processing import determinant_hermitian_matrix3
from lib.processing import inverse_hermitian_matrix4
from lib.processing import determinant_hermitian_matrix4
import lib.matrix
import lib.graphics

numba.config.THREADING_LAYER = 'omp'
if numba.config.NUMBA_NUM_THREADS > 1:
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS - 1)

NUMBA_IS_LINUX = numba.np.ufunc.parallel._IS_LINUX

if NUMBA_IS_LINUX is True:
    get_thread_id_type = numba.typeof(numba.np.ufunc.parallel._get_thread_id())

    @numba.njit
    def numba_get_thread_id():
        with numba.objmode(ret=get_thread_id_type):
            ret = numba.np.ufunc.parallel._get_thread_id()
            return ret
else:
    @numba.njit
    def numba_get_thread_id():
        with numba.objmode(ret=numba.int32):
            ret = numba.int32(-1)
            return ret


@numba.njit(parallel=False, fastmath=True)
def average_complex_coherency_matrix_determination_1(sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, m_prm_al, m_prm_h, n_pp, coh_area, cpt_area, class_im, lig_g):
    LIM_AL1 = numpy.float32(55.0)  # H and alpha decision boundaries
    LIM_AL2 = numpy.float32(50.0)
    LIM_AL3 = numpy.float32(48.0)
    LIM_AL4 = numpy.float32(42.0)
    LIM_AL5 = numpy.float32(40.0)
    LIM_H1 = numpy.float32(0.90)
    LIM_H2 = numpy.float32(0.50)
    a1 = a2 = a3 = a4 = a5 = h1 = h2 = numpy.float32(0.0)
    r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = numpy.float32(0.0)
    area = 0
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type_out in ['C3', 'T3']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.0
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.0
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.0

            if pol_type_out in ['C4', 'T4']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.0
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[0][3][0] = eps + m_avg[5][col]
                m[0][3][1] = eps + m_avg[6][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[7][col]
                m[1][1][1] = 0.0
                m[1][2][0] = eps + m_avg[8][col]
                m[1][2][1] = eps + m_avg[9][col]
                m[1][3][0] = eps + m_avg[10][col]
                m[1][3][1] = eps + m_avg[11][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[12][col]
                m[2][2][1] = 0.0
                m[2][3][0] = eps + m_avg[13][col]
                m[2][3][1] = eps + m_avg[14][col]
                m[3][0][0] = m[0][3][0]
                m[3][0][1] = -m[0][3][1]
                m[3][1][0] = m[1][3][0]
                m[3][1][1] = -m[1][3][1]
                m[3][2][0] = m[2][3][0]
                m[3][2][1] = -m[2][3][1]
                m[3][3][0] = eps + m_avg[15][col]
                m[3][3][1] = 0.0

            a1 = numpy.float32(m_prm_al[lig][col] <= LIM_AL1)
            a2 = numpy.float32(m_prm_al[lig][col] <= LIM_AL2)
            a3 = numpy.float32(m_prm_al[lig][col] <= LIM_AL3)
            a4 = numpy.float32(m_prm_al[lig][col] <= LIM_AL4)
            a5 = numpy.float32(m_prm_al[lig][col] <= LIM_AL5)

            h1 = numpy.float32(m_prm_h[lig][col] <= LIM_H1)
            h2 = numpy.float32(m_prm_h[lig][col] <= LIM_H2)

            # ZONE 1 (top left)
            r1 = (not a3) * h2
            # ZONE 2 (center left)
            r2 = a3 * (not a4) * h2
            # ZONE 3 (bottom left)
            r3 = a4 * h2
            # ZONE 4 (top center)
            r4 = (not a2) * h1 * (not h2)
            # ZONE 5 (center center)
            r5 = a2 * (not a5) * h1 * (not h2)
            # ZONE 6 (bottom center)
            r6 = a5 * h1 * (not h2)
            # ZONE 7 (top right)
            r7 = (not a1) * (not h1)
            # ZONE 8 (center right)
            r8 = a1 * (not a5) * (not h1)
            # ZONE 9 (bottom right)
            r9 = a5 * (not h1)  # Non feasible region

            area = numpy.int32(r1 + 2 * r2 + 3 * r3 + 4 * r4 + 5 * r5 + 6 * r6 + 7 * r7 + 8 * r8 + 9 * r9)

            # Class center coherency matrices are initialized
            # according to the H_alpha classification results
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area[k][l][0][area] = coh_area[k][l][0][area] + m[k][l][0]
                    coh_area[k][l][1][area] = coh_area[k][l][1][area] + m[k][l][1]

            cpt_area[area] = cpt_area[area] + 1.0
            class_im[lig_g][col] = numpy.float32(area)


@numba.njit(parallel=False, fastmath=True)
def prepare_coh_area_for_inverse_center_coherency_matrices_computation(n_area, cpt_area, n_pp, coh_area):
    for area in range(1, n_area + 1):
        if cpt_area[area] != 0.:
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area[k][l][0][area] = coh_area[k][l][0][area] / cpt_area[area]
                    coh_area[k][l][1][area] = coh_area[k][l][1][area] / cpt_area[area]


@numba.njit(parallel=False, fastmath=True)
def inverse_center_coherency_matrices_computation_1(n_area, cpt_area, n_pp, coh, coh_area, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area):
    for area in range(1, n_area + 1):
        if cpt_area[area] != 0.:
            for k in range(n_pp):
                for l in range(n_pp):
                    coh[k][l][0] = coh_area[k][l][0][area]
                    coh[k][l][1] = coh_area[k][l][1][area]
            if pol_type_out in ['C3', 'T3']:
                inverse_hermitian_matrix3(coh, coh_m1)
                determinant_hermitian_matrix3(coh, det, eps)
            if pol_type_out in ['C4', 'T4']:
                inverse_hermitian_matrix4(coh, coh_m1, eps)
                determinant_hermitian_matrix4(coh, det, eps)
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area_m1[k][l][0][area] = coh_m1[k][l][0]
                    coh_area_m1[k][l][1][area] = coh_m1[k][l][1]
            det_area[0][area] = det[0]
            det_area[1][area] = det[1]


@numba.njit(parallel=False, fastmath=True)
def average_complex_coherency_matrix_determination_2(modif, sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_area, n_pp, coh_m1, coh_area_m1, distance, det_area, init_minmax, cpt_area, class_im, lig_g):
    zone = 0
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type_out in ['C3', 'T3']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.

            if pol_type_out in ['C4', 'T4']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[0][3][0] = eps + m_avg[5][col]
                m[0][3][1] = eps + m_avg[6][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[7][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[8][col]
                m[1][2][1] = eps + m_avg[9][col]
                m[1][3][0] = eps + m_avg[10][col]
                m[1][3][1] = eps + m_avg[11][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[12][col]
                m[2][2][1] = 0.
                m[2][3][0] = eps + m_avg[13][col]
                m[2][3][1] = eps + m_avg[14][col]
                m[3][0][0] = m[0][3][0]
                m[3][0][1] = -m[0][3][1]
                m[3][1][0] = m[1][3][0]
                m[3][1][1] = -m[1][3][1]
                m[3][2][0] = m[2][3][0]
                m[3][2][1] = -m[2][3][1]
                m[3][3][0] = eps + m_avg[15][col]
                m[3][3][1] = 0.

            # Seeking for the closest cluster center
            for area in range(1, n_area + 1):
                if cpt_area[area] != 0.:
                    for k in range(n_pp):
                        for l in range(n_pp):
                            coh_m1[k][l][0] = coh_area_m1[k][l][0][area]
                            coh_m1[k][l][1] = coh_area_m1[k][l][1][area]
                    distance[area] = math.log(math.sqrt(det_area[0][area] * det_area[0][area] + det_area[1][area] * det_area[1][area]))
                    if pol_type_out in ['C3', 'T3']:
                        distance[area] = distance[area] + trace3_hm1xhm2(coh_m1, m)
                    if pol_type_out in ['C4', 'T4']:
                        distance[area] = distance[area] + trace4_hm1xhm2(coh_m1, m)
            dist_min = init_minmax
            for area in range(1, n_area + 1):
                if cpt_area[area] != 0.:
                    if dist_min > distance[area]:
                        dist_min = distance[area]
                        zone = area
            if zone != numpy.int32(class_im[lig_g][col]):
                modif = modif + 1.
            class_im[lig_g][col] = numpy.float32(zone)
    return modif


@numba.njit(parallel=False, fastmath=True)
def average_complex_coherency_matrix_determination_3(sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_pp, coh_area, cpt_area, class_im, lig_g):
    zone = 0
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type_out in ['C3', 'T3']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.

            if pol_type_out in ['C4', 'T4']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[0][3][0] = eps + m_avg[5][col]
                m[0][3][1] = eps + m_avg[6][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[7][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[8][col]
                m[1][2][1] = eps + m_avg[9][col]
                m[1][3][0] = eps + m_avg[10][col]
                m[1][3][1] = eps + m_avg[11][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[12][col]
                m[2][2][1] = 0.
                m[2][3][0] = eps + m_avg[13][col]
                m[2][3][1] = eps + m_avg[14][col]
                m[3][0][0] = m[0][3][0]
                m[3][0][1] = -m[0][3][1]
                m[3][1][0] = m[1][3][0]
                m[3][1][1] = -m[1][3][1]
                m[3][2][0] = m[2][3][0]
                m[3][2][1] = -m[2][3][1]
                m[3][3][0] = eps + m_avg[15][col]
                m[3][3][1] = 0.

            area = numpy.int32(class_im[lig_g][col])

            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area[k][l][0][area] = coh_area[k][l][0][area] + m[k][l][0]
                    coh_area[k][l][1][area] = coh_area[k][l][1][area] + m[k][l][1]

            cpt_area[area] = cpt_area[area] + 1.


@numba.njit(parallel=False, fastmath=True)
def inverse_center_coherency_matrices_computation_3a(n_area, cpt_area, n_pp, coh, coh_area, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area):
    for area in range(1, n_area + 1):
        if cpt_area[area] != 0.:
            for k in range(n_pp):
                for l in range(n_pp):
                    coh[k][l][0] = coh_area[k][l][0][area]
                    coh[k][l][1] = coh_area[k][l][1][area]
            if pol_type_out in ['C3', 'T3']:
                inverse_hermitian_matrix3(coh, coh_m1)
                determinant_hermitian_matrix3(coh, det, eps)
            if pol_type_out in ['C4', 'T4']:
                inverse_hermitian_matrix4(coh, coh_m1, eps)
                determinant_hermitian_matrix4(coh, det, eps)
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area_m1[k][l][0][area] = coh_m1[k][l][0]
                    coh_area_m1[k][l][1][area] = coh_m1[k][l][1]
            det_area[0][area] = det[0]
            det_area[1][area] = det[1]



@numba.njit(parallel=False, fastmath=True)
def average_complex_coherency_matrix_determination_4(sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_pp, coh_area, cpt_area, class_im, m_prm_a, lig_g):
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type_out in ['C3', 'T3']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.

            if pol_type_out in ['C4', 'T4']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[0][3][0] = eps + m_avg[5][col]
                m[0][3][1] = eps + m_avg[6][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[7][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[8][col]
                m[1][2][1] = eps + m_avg[9][col]
                m[1][3][0] = eps + m_avg[10][col]
                m[1][3][1] = eps + m_avg[11][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[12][col]
                m[2][2][1] = 0.
                m[2][3][0] = eps + m_avg[13][col]
                m[2][3][1] = eps + m_avg[14][col]
                m[3][0][0] = m[0][3][0]
                m[3][0][1] = -m[0][3][1]
                m[3][1][0] = m[1][3][0]
                m[3][1][1] = -m[1][3][1]
                m[3][2][0] = m[2][3][0]
                m[3][2][1] = -m[2][3][1]
                m[3][3][0] = eps + m_avg[15][col]
                m[3][3][1] = 0.

            area = numpy.int32(class_im[lig_g][col])
            if m_prm_a[lig][col] > 0.5:
                area = area + 8

            # Class center coherency matrices are initialize
            # according to the H_alpha classification results
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area[k][l][0][area] = coh_area[k][l][0][area] + m[k][l][0]
                    coh_area[k][l][1][area] = coh_area[k][l][1][area] + m[k][l][1]
            cpt_area[area] = cpt_area[area] + 1.
            class_im[lig_g][col] = numpy.float32(area)


@numba.njit(parallel=False, fastmath=True)
def inverse_center_coherency_matrices_computation_4a(n_area, cpt_area, n_pp, coh, coh_area, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area):
    for area in range(1, n_area + 1):
        if cpt_area[area] != 0.:
            for k in range(n_pp):
                for l in range(n_pp):
                    coh[k][l][0] = coh_area[k][l][0][area]
                    coh[k][l][1] = coh_area[k][l][1][area]
            if pol_type_out in ['C3', 'T3']:
                inverse_hermitian_matrix3(coh, coh_m1)
                determinant_hermitian_matrix3(coh, det, eps)
            if pol_type_out in ['C4', 'T4']:
                inverse_hermitian_matrix4(coh, coh_m1, eps)
                determinant_hermitian_matrix4(coh, det, eps)
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area_m1[k][l][0][area] = coh_m1[k][l][0]
                    coh_area_m1[k][l][1][area] = coh_m1[k][l][1]
            det_area[0][area] = det[0]
            det_area[1][area] = det[1]


@numba.njit(parallel=False, fastmath=True)
def average_complex_coherency_matrix_determination_5(modif, sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_area, n_pp, coh_m1, coh_area_m1, distance, det_area,  init_minmax, cpt_area, class_im, m_prm_a, lig_g):
    zone = 0
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type_out in ['C3', 'T3']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.

            if pol_type_out in ['C4', 'T4']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[0][3][0] = eps + m_avg[5][col]
                m[0][3][1] = eps + m_avg[6][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[7][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[8][col]
                m[1][2][1] = eps + m_avg[9][col]
                m[1][3][0] = eps + m_avg[10][col]
                m[1][3][1] = eps + m_avg[11][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[12][col]
                m[2][2][1] = 0.
                m[2][3][0] = eps + m_avg[13][col]
                m[2][3][1] = eps + m_avg[14][col]
                m[3][0][0] = m[0][3][0]
                m[3][0][1] = -m[0][3][1]
                m[3][1][0] = m[1][3][0]
                m[3][1][1] = -m[1][3][1]
                m[3][2][0] = m[2][3][0]
                m[3][2][1] = -m[2][3][1]
                m[3][3][0] = eps + m_avg[15][col]
                m[3][3][1] = 0.

            # Seeking for the closest cluster center */
            for area in range(1, n_area + 1):
                if cpt_area[area] != 0.:
                    for k in range(n_pp):
                        for l in range(n_pp):
                            coh_m1[k][l][0] = coh_area_m1[k][l][0][area]
                            coh_m1[k][l][1] = coh_area_m1[k][l][1][area]
                    distance[area] = math.log(math.sqrt(det_area[0][area] * det_area[0][area] + det_area[1][area] * det_area[1][area]))
                    if pol_type_out in ['C3', 'T3']:
                        distance[area] = distance[area] + trace3_hm1xhm2(coh_m1, m)
                    if pol_type_out in ['C4', 'T4']:
                        distance[area] = distance[area] + trace4_hm1xhm2(coh_m1, m)
            dist_min = init_minmax
            for area in range(1, n_area + 1):
                if cpt_area[area] != 0.:
                    if dist_min > distance[area]:
                        dist_min = distance[area]
                        zone = area
            if zone != (int)(class_im[lig_g][col]):
                modif = modif + 1.
            class_im[lig_g][col] = numpy.float32(zone)
    return modif


@numba.njit(parallel=False, fastmath=True)
def average_complex_coherency_matrix_determination_6(sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_pp, coh_area, cpt_area, class_im, lig_g):
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type_out in ['C3', 'T3']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.

            if pol_type_out in ['C4', 'T4']:
                # Average complex coherency matrix determination
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[0][3][0] = eps + m_avg[5][col]
                m[0][3][1] = eps + m_avg[6][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[7][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[8][col]
                m[1][2][1] = eps + m_avg[9][col]
                m[1][3][0] = eps + m_avg[10][col]
                m[1][3][1] = eps + m_avg[11][col]
                m[2][0][0] = m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] = m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[12][col]
                m[2][2][1] = 0.
                m[2][3][0] = eps + m_avg[13][col]
                m[2][3][1] = eps + m_avg[14][col]
                m[3][0][0] = m[0][3][0]
                m[3][0][1] = -m[0][3][1]
                m[3][1][0] = m[1][3][0]
                m[3][1][1] = -m[1][3][1]
                m[3][2][0] = m[2][3][0]
                m[3][2][1] = -m[2][3][1]
                m[3][3][0] = eps + m_avg[15][col]
                m[3][3][1] = 0.

            area = numpy.int32(class_im[lig_g][col])

            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area[k][l][0][area] = coh_area[k][l][0][area] + m[k][l][0]
                    coh_area[k][l][1][area] = coh_area[k][l][1][area] + m[k][l][1]
            cpt_area[area] = cpt_area[area] + 1.


@numba.njit(parallel=False, fastmath=True)
def inverse_center_coherency_matrices_computation_6a(n_area, cpt_area, n_pp, coh_area, coh, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area):
    for area in range(1, n_area + 1):
        if cpt_area[area] != 0.:
            for k in range(n_pp):
                for l in range(n_pp):
                    coh[k][l][0] = coh_area[k][l][0][area]
                    coh[k][l][1] = coh_area[k][l][1][area]
            if pol_type_out in ['C3', 'T3']:
                lib.processing.inverse_hermitian_matrix3(coh, coh_m1)
                lib.processing.determinant_hermitian_matrix3(coh, det, eps)
            if pol_type_out in ['C4', 'T4']:
                lib.processing.inverse_hermitian_matrix4(coh, coh_m1, eps)
                lib.processing.determinant_hermitian_matrix4(coh, det, eps)
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area_m1[k][l][0][area] = coh_m1[k][l][0]
                    coh_area_m1[k][l][1][area] = coh_m1[k][l][1]
            det_area[0][area] = det[0]
            det_area[1][area] = det[1]


class App(lib.util.Application):
    ALPHA = 0
    H = 1
    A = 2
    # CONSTANTS
    NPRM = 3  # nb of parameter files
    LIM_AL1 = 55.  # H and alpha decision boundaries
    LIM_AL2 = 50.
    LIM_AL3 = 48.
    LIM_AL4 = 42.
    LIM_AL5 = 40.
    LIM_H1 = 0.9
    LIM_H2 = 0.5

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col, sub_n_lig):
        """
        Allocate matrices with given dimensions
        """
        logging.info(f'{n_col=}, {n_polar_out=}, {n_win_l=}, {n_win_c=}, {n_lig_block=}, {sub_n_col=}')
        self.vc_in = lib.matrix.vector_float(2 * n_col)
        self.vf_in = lib.matrix.vector_float(n_col)
        self.mc_in = lib.matrix.matrix_float(4, 2 * n_col)
        self.mf_in = lib.matrix.matrix3d_float(n_polar_out, n_win_l, n_col + n_win_c)

        self.valid = lib.matrix.matrix_float(n_lig_block + n_win_l, sub_n_col + n_win_c)

        self.m_in = lib.matrix.matrix3d_float(n_polar_out, n_lig_block + n_win_l, n_col + n_win_c)
        self.class_im = lib.matrix.matrix_float(sub_n_lig, n_col)
        self.m_prm_h = lib.matrix.matrix_float(n_lig_block, n_col)
        self.m_prm_al = lib.matrix.matrix_float(n_lig_block, n_col)
        self.m_prm_a = lib.matrix.matrix_float(n_lig_block, n_col)
        self.cpt_area = lib.matrix.vector_float(100)
        self.distance = lib.matrix.vector_float(100)

    def run(self):
        logging.info('******************** Welcome in wishart_h_a_alpha_classifier ********************')
        logging.info(self.args)
        in_dir = self.args.id
        out_dir = self.args.od
        pol_type = self.args.iodf
        n_win_l = self.args.nwr
        n_win_c = self.args.nwc
        off_lig = self.args.ofr
        off_col = self.args.ofc
        sub_n_lig = self.args.fnr
        sub_n_col = self.args.fnc
        file_entropy = self.args.hf
        file_anisotropy = self.args.af
        file_alpha = self.args.alf
        nit_max = self.args.nit
        pct_switch_min = self.args.pct
        bmp_flag = self.args.bmp
        color_map_wishart8 = None
        color_map_wishart16 = None
        if bmp_flag == 1:
            color_map_wishart8 = self.args.co8
            color_map_wishart16 = self.args.co16
        file_memerr = self.args.errf

        flag_valid = False
        file_valid = ''

        if self.args.mask is not None and self.args.mask:
            file_valid = self.args.mask
            flag_valid = True
        logging.info(f'{flag_valid=}, {file_valid=}')

        in_dir = self.check_dir(in_dir)
        logging.info(f'{in_dir=}')
        out_dir = self.check_dir(out_dir)
        logging.info(f'{out_dir=}')
        if bmp_flag == 1:
            self.check_file(color_map_wishart8)
            self.check_file(color_map_wishart16)
        self.check_file(file_entropy)
        self.check_file(file_anisotropy)
        self.check_file(file_alpha)

        if flag_valid is True:
            self.check_file(file_valid)

        n_win_l_m1s2 = (n_win_l - 1) // 2
        logging.info(f'{n_win_l_m1s2=}')
        n_win_c_m1s2 = (n_win_c - 1) // 2
        logging.info(f'{n_win_c_m1s2=}')

        pct_switch_min = pct_switch_min / 100.

        # INPUT/OUPUT CONFIGURATIONS
        n_lig, n_col, polar_case, polar_type = lib.util.read_config(in_dir)
        logging.info(f'{n_lig=}, {n_col=}, {polar_case=}, {polar_type=}')

        # POLAR TYPE CONFIGURATION
        if pol_type == 'S2':
            if polar_case == 'monostatic':
                pol_type = 'S2T3'
            if polar_case == 'bistatic':
                pol_type = 'S2T4'
        logging.info(f'{pol_type=}')

        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type)
        logging.info(f'{pol_type=}, {n_polar_in=}, {pol_type_in=}, {n_polar_out=}, {pol_type_out=}')

        # INPUT/OUTPUT FILE CONFIGURATION
        file_name_in = lib.util.init_file_name(pol_type_in, in_dir)
        logging.info(f'{file_name_in=}')

        # INPUT FILE OPENING
        in_datafile = []
        in_valid = None
        in_datafile, in_valid, flag_valid = self.open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)
        prm_file_al = self.open_file(file_alpha, "rb")
        prm_file_h = self.open_file(file_entropy, "rb")
        prm_file_a = self.open_file(file_anisotropy, "rb")

        # OUTPUT FILE OPENING
        file_name_out = [
            os.path.join(f'{out_dir}', f'wishart_H_alpha_class_{n_win_l}x{n_win_c}.bin'),
            os.path.join(f'{out_dir}', f'wishart_H_A_alpha_class_{n_win_l}x{n_win_c}.bin'),
        ]
        logging.info(f'{file_name_out=}')

        w_h_alpha_file = self.open_output_file(file_name_out[0])
        w_h_a_alpha_file = self.open_output_file(file_name_out[1])

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)

        # MprmH = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # Mprmal = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # MprmA = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # ClassIm = sub_n_lig,*sub_n_col
        n_block_a += 0
        n_block_b += sub_n_lig * sub_n_col
        # Min = n_polar_out*Nlig*sub_n_col
        n_block_a += n_polar_out * (n_col + n_win_c)
        n_block_b += n_polar_out * n_win_l * (n_col + n_win_c)
        # Mavg = n_polar_out
        n_block_a += 0
        n_block_b += n_polar_out * sub_n_col
        # Reading Data
        n_block_b += n_col + 2 * n_col + n_polar_in * 2 * n_col + n_polar_out * n_win_l * (n_col + n_win_c)

        memory_alloc = self.check_free_memory()
        memory_alloc = max(memory_alloc, 1000)
        logging.info(f'{memory_alloc=}')
        n_lig_block = numpy.zeros(lib.util.Application.FILE_PATH_LENGTH, dtype=numpy.int32)
        nb_block = 0
        nb_block = self.memory_alloc(file_memerr, sub_n_lig, n_win_l, nb_block, n_lig_block, n_block_a, n_block_b, memory_alloc)
        logging.info(f'{n_lig_block=}')
        logging.info(f'{nb_block=}')

        # MATRIX ALLOCATION
        self.allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, n_lig_block[0], sub_n_col, sub_n_lig)

        # MASK VALID PIXELS (if there is no MaskFile)
        self.set_valid_pixels(flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)
        init_minmax = lib.util.Application.INIT_MINMAX

        # DATA PROCESSING
        logging.info('--= Started: data processing =--')
        init_time_data_processing = datetime.datetime.now()
        # Training class matrix memory allocation
        n_pp = None
        if pol_type_out in ['C3', 'T3']:
            n_pp = 3
        elif pol_type_out in ['C4', 'T4']:
            n_pp = 4

        tmp_coh_area = [[[None] * 2] * 4] * 4
        tmp_coh_area_m1 = [[[None] * 2] * 4] * 4
        det = lib.matrix.vector_float(2)
        coh = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

        n_area = 20
        for k in range(n_pp):
            for l in range(n_pp):
                tmp_coh_area[k][l][0] = lib.matrix.vector_float(n_area)
                tmp_coh_area[k][l][1] = lib.matrix.vector_float(n_area)
                tmp_coh_area_m1[k][l][0] = lib.matrix.vector_float(n_area)
                tmp_coh_area_m1[k][l][1] = lib.matrix.vector_float(n_area)
        coh_area = numpy.array(tmp_coh_area)
        coh_area_m1 = numpy.array(tmp_coh_area_m1)
        tmp_det_area = [None] * 2
        tmp_det_area[0] = lib.matrix.vector_float(n_area)
        tmp_det_area[1] = lib.matrix.vector_float(n_area)
        det_area = numpy.array(tmp_det_area)

        for area in range(1, n_area + 1):
            self.cpt_area[area] = 0.

        n_lig_g = 0
        lig_g = 0
        eps = lib.util.Application.EPS

        logging.info('--= Started: Inverse center coherency matrices computation =--')
        init_time = datetime.datetime.now()

        NNwinLigM1S2 = (n_win_l - 1) // 2
        vf_in_readingLines = [None] * nb_block
        for nb in range(nb_block):
            logging.info(f'READING NLIG LINES {nb=} {n_col=} {n_polar_out=} {n_lig_block[nb]=} {NNwinLigM1S2=} from wishart')
            vf_in_readingLines[nb] = [numba.typed.List([numpy.fromfile(in_datafile[Np], dtype=numpy.float32, count=n_col) for Np in range(n_polar_out)]) for lig in range(n_lig_block[nb] + NNwinLigM1S2)]

        for nb in range(nb_block):
            ligDone = 0
            if nb_block > 2:
                lib.util.printf_line(nb, nb_block)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            lib.util_block.read_block_matrix_float(prm_file_h, self.m_prm_h, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            lib.util_block.read_block_matrix_float(prm_file_al, self.m_prm_al, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            if pol_type == 'S2':
                lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # Case of C,T or I
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])

            area = 0
            # a1 = a2 = a3 = a4 = a5 = h1 = h2 = 0.
            # r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = 0.
            # pragma omp parallel for private(col, k, l, M_avg, M) firstprivate(ligg, area, a1, a2, a3, a4, a5, h1, h2, r1, r2, r3, r4, r5, r6, r7, r8, r9) shared(ligDone, coh_area, cpt_area)
            logging.info(f'--= Started: Inverse center coherency matrices computation loop {n_lig_block[nb]=}=--')
            for lig in range(n_lig_block[nb]):
                ligDone += 1
                if numba_get_thread_id() == 0:
                    lib.util.printf_line(ligDone, n_lig_block[nb])
                m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
                lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                lig_g = lig + n_lig_g
                # logging.info(f'--= Started: Inverse center coherency matrices computation loop 1 !!! {lig=}=--')
                average_complex_coherency_matrix_determination_1(sub_n_col, self.valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, self.m_prm_al, self.m_prm_h, n_pp, coh_area, self.cpt_area, self.class_im, lig_g)
            n_lig_g += n_lig_block[nb]

        n_area = 8
        logging.info('--= Started: Inverse center coherency matrices computation: before_inverse_center_coherency_matrices_computation_1a =--')
        prepare_coh_area_for_inverse_center_coherency_matrices_computation(n_area, self.cpt_area, n_pp, coh_area)

        coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

        # Inverse center coherency matrices computation
        logging.info('--= Started: Inverse center coherency matrices computation: inverse_center_coherency_matrices_computation_1 =--')
        inverse_center_coherency_matrices_computation_1(n_area, self.cpt_area, n_pp, coh, coh_area, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area)

        logging.info('--= Finished: Inverse center coherency matrices in: %s sec =--' % (datetime.datetime.now() - init_time))

        # START OF THE WISHART H-ALPHA CLASSIFICATION
        logging.info('--= Started: WISHART H-ALPHA CLASSIFICATION =--')
        init_time = datetime.datetime.now()
        flag_stop = 0
        nit = 0
        while flag_stop == 0:
            logging.info('--= Started: WISHART H-ALPHA CLASSIFICATION: in flag_stop =--')
            nit += 1
            for np in range(n_polar_in):
                self.rewind(in_datafile[np])
            if flag_valid is True:
                self.rewind(in_valid)

            modif = 0.
            n_lig_g = 0
            lig_g = 0
            for nb in range(nb_block):
                logging.info('--= Started: WISHART H-ALPHA CLASSIFICATION: in range(nb_block) =--')
                ligDone = 0

                if nb_block > 2:
                    lib.util.printf_line(nb, nb_block)
                if flag_valid is True:
                    lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)
                if pol_type == 'S2':
                    lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                else:  # Case of C,T or I
                    lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])

                zone = 0
                dist_min = init_minmax
                # pragma omp parallel for private(col, area, k, l, M_avg, M, coh_m1) firstprivate(ligg, distance, dist_min, zone) shared(ligDone, modif)
                for lig in range(n_lig_block[nb]):
                    # logging.info('--= Started: WISHART H-ALPHA CLASSIFICATION: in range(n_lig_block[nb] =--')
                    ligDone += 1
                    if numba_get_thread_id() == 0:
                        lib.util.printf_line(ligDone, n_lig_block[nb])
                    m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                    coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
                    lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                    lig_g = lig + n_lig_g
                    # logging.info(f'--= Started: average_complex_coherency_matrix_determination_2 loop !!! {lig=}=--')
                    modif = average_complex_coherency_matrix_determination_2(modif, sub_n_col, self.valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_area, n_pp, coh_m1, coh_area_m1, self.distance, det_area, init_minmax, self.cpt_area, self.class_im, lig_g)
                n_lig_g += n_lig_block[nb]

            logging.info('--= Finished : WISHART H-ALPHA CLASSIFICATION  2=--')

            flag_stop = 0
            if modif < pct_switch_min * (float)(sub_n_lig * sub_n_col):
                flag_stop = 1
            if nit == nit_max:
                flag_stop = 1

            print("{:.2f}%\r".format(100. * nit / nit_max), end="", flush=True)
            # lib.util.printf_line(ligDone, n_lig_block[nb])

            if flag_stop == 0:
                # Calcul des nouveaux centres de classe
                for area in range(1, n_area + 1):
                    self.cpt_area[area] = 0.
                    for k in range(n_pp):
                        for l in range(n_pp):
                            coh_area[k][l][0][area] = 0.
                            coh_area[k][l][1][area] = 0.
                for np in range(n_polar_in):
                    self.rewind(in_datafile[np])
                if flag_valid is True:
                    self.rewind(in_valid)
                modif = 0.
                n_lig_g = 0
                lig_g = 0
                for nb in range(nb_block):
                    ligDone = 0

                    if nb_block > 2:
                        lib.util.printf_line(nb, nb_block)
                    if flag_valid is True:
                        lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)
                    if pol_type == 'S2':
                        lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                    else:  # Case of C,T or I
                        lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in,  vf_in_readingLines[nb])

                    area = 0
                    # pragma omp parallel for private(col, k, l, M_avg, M) firstprivate(ligg, area) shared(ligDone, coh_area, cpt_area)
                    for lig in range(n_lig_block[nb]):
                        ligDone += 1
                        if numba_get_thread_id() == 0:
                            lib.util.printf_line(ligDone, n_lig_block[nb])
                        m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                        m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
                        lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                        lig_g = lig + n_lig_g
                        average_complex_coherency_matrix_determination_3(sub_n_col, self.valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_pp, coh_area, self.cpt_area, self.class_im, lig_g)

                    n_lig_g += n_lig_block[nb]

                prepare_coh_area_for_inverse_center_coherency_matrices_computation(n_area, self.cpt_area, n_pp, coh_area)
                coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

                # Inverse center coherency matrices computation
                inverse_center_coherency_matrices_computation_3a(n_area, self.cpt_area, n_pp, coh, coh_area, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area)
        logging.info('--= Finished: WISHART H-ALPHA CLASSIFICATION in: %s sec =--' % (datetime.datetime.now() - init_time))

        # Saving wishart_H_alpha classification results bin and bitmap
        logging.info('--= Started: Saving wishart_H_alpha classification results bin and bitmap =--')
        init_time = datetime.datetime.now()
        self.class_im[0][0] = 1.
        self.class_im[1][1] = 8.
        lib.util_block.write_block_matrix_float(w_h_alpha_file, self.class_im, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

        if bmp_flag == 1:
            file_name = os.path.join(f'{out_dir}', f'wishart_H_alpha_class_{n_win_l}x{n_win_c}')
            lib.graphics.bmp_wishart(self.class_im, sub_n_lig, sub_n_col, file_name, color_map_wishart8)

        logging.info('--= Finished: wishart_H_alpha classification results bin and bitmap in: %s sec =--' % (datetime.datetime.now() - init_time))
        # END OF THE WISHART H-ALPHA CLASSIFICATION

        logging.info('--= Started: PRE WISHART H-A-ALPHA CLASSIFICATION =--')
        init_time = datetime.datetime.now()

        for np in range(n_polar_in):
            self.rewind(in_datafile[np])
        if flag_valid is True:
            self.rewind(in_valid)

        n_area = 20
        for area in range(1, n_area + 1):
            self.cpt_area[area] = 0.
            for k in range(n_pp):
                for l in range(n_pp):
                    coh_area[k][l][0][area] = 0.
                    coh_area[k][l][1][area] = 0.

        n_lig_g = 0
        lig_g = 0

        for nb in range(nb_block):
            ligDone = 0

            if nb_block > 2:
                lib.util.printf_line(nb, nb_block)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            lib.util_block.read_block_matrix_float(prm_file_a, self.m_prm_a, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            if pol_type == 'S2':
                lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # case of c,t or i
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])

            area = 0
            # pragma omp parallel for private(col, k, l, M_avg, M) firstprivate(ligg, area) shared(ligDone, coh_area, cpt_area)
            for lig in range(n_lig_block[nb]):
                ligDone += 1
                if numba_get_thread_id() == 0:
                    lib.util.printf_line(ligDone, n_lig_block[nb])
                m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
                lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                lig_g = lig + n_lig_g
                average_complex_coherency_matrix_determination_4(sub_n_col, self.valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_pp, coh_area, self.cpt_area, self.class_im, self.m_prm_a, lig_g)
            n_lig_g += n_lig_block[nb]

            n_area = 16
            prepare_coh_area_for_inverse_center_coherency_matrices_computation(n_area, self.cpt_area, n_pp, coh_area)
            coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

            # Inverse center coherency matrices computation
            inverse_center_coherency_matrices_computation_4a(n_area, self.cpt_area, n_pp, coh, coh_area, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area)

        logging.info('--= Finished: Inverse center coherency matrices computation in: %s sec =--' % (datetime.datetime.now() - init_time))

        # START OF THE WISHART H-A-ALPHA CLASSIFICATION
        logging.info('--= Started: WISHART H-A-ALPHA CLASSIFICATION =--')
        init_time = datetime.datetime.now()

        flag_stop = 0
        nit = 0

        while flag_stop == 0:
            nit += 1
            for np in range(n_polar_in):
                self.rewind(in_datafile[np])
            if flag_valid is True:
                self.rewind(in_valid)

            modif = 0.
            lig_g = 0
            n_lig_g = 0
            for nb in range(nb_block):
                ligDone = 0

                if nb_block > 2:
                    lib.util.printf_line(nb, nb_block)
                if flag_valid is True:
                    lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

                if pol_type == 'S2':
                    lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                else:  # Case of C,T or I
                    lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])

                zone = 0
                dist_min = init_minmax
                # pragma omp parallel for private(col, area, k, l, M_avg, M, coh_m1) firstprivate(ligg, distance, dist_min, zone) shared(ligDone, modif)
                for lig in range(n_lig_block[nb]):
                    ligDone += 1
                    if numba_get_thread_id() == 0:
                        lib.util.printf_line(ligDone, n_lig_block[nb])
                    coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                    m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
                    lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                    lig_g = lig + n_lig_g
                    modif = average_complex_coherency_matrix_determination_5(modif, sub_n_col, self.valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_area, n_pp, coh_m1, coh_area_m1, self.distance, det_area,  init_minmax, self.cpt_area, self.class_im, self.m_prm_a, lig_g)
                n_lig_g += n_lig_block[nb]

            flag_stop = 0
            if modif < pct_switch_min * (float)(sub_n_lig * sub_n_col):
                flag_stop = 1
            if nit == nit_max:
                flag_stop = 1

            print("{:.2f}%\r".format(100. * nit / nit_max), end="", flush=True)

            if flag_stop == 0:
                # Calcul des nouveaux centres de classe
                for area in range(1, n_area + 1):
                    self.cpt_area[area] = 0.
                    for k in range(n_pp):
                        for l in range(n_pp):
                            coh_area[k][l][0][area] = 0.
                            coh_area[k][l][1][area] = 0.
                for np in range(n_polar_in):
                    self.rewind(in_datafile[np])
                if flag_valid is True:
                    self.rewind(in_valid)

                modif = 0.

                n_lig_g = 0
                lig_g = 0
                for nb in range(nb_block):
                    ligDone = 0
                    if nb_block > 2:
                        lib.util.printf_line(nb, nb_block)

                    if flag_valid is True:
                        lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

                    if pol_type == 'S2':
                        lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                    else:  # Case of C,T or I
                        lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])

                    area = 0
                    # pragma omp parallel for private(col, k, l, M_avg, M) firstprivate(ligg, area) shared(ligDone, coh_area, cpt_area)
                    for lig in range(n_lig_block[nb]):
                        ligDone += 1
                        if numba_get_thread_id() == 0:
                            lib.util.printf_line(ligDone, n_lig_block[nb])
                        m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
                        m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
                        lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                        lig_g = lig + n_lig_g
                        average_complex_coherency_matrix_determination_6(sub_n_col, self.valid, n_win_l_m1s2, lig, n_win_c_m1s2, eps, pol_type_out, m, m_avg, n_pp, coh_area, self.cpt_area, self.class_im, lig_g)
                    n_lig_g += n_lig_block[nb]

                prepare_coh_area_for_inverse_center_coherency_matrices_computation(n_area, self.cpt_area, n_pp, coh_area)

                coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

                # Inverse center coherency matrices computation
                inverse_center_coherency_matrices_computation_6a(n_area, self.cpt_area, n_pp, coh_area, coh, pol_type_out, coh_m1, det, eps, coh_area_m1, det_area)

        logging.info('--= Finished: WISHART H-A-ALPHA CLASSIFICATION in: %s sec =--' % (datetime.datetime.now() - init_time))

        # Saving wishart_H_A_alpha classification results bin and bitmap
        logging.info('--= Started: Saving wishart_H_alpha classification results bin and bitmap =--')
        init_time = datetime.datetime.now()
        self.class_im[0][0] = 1.
        self.class_im[1][1] = 16.

        lib.util_block.write_block_matrix_float(w_h_a_alpha_file, self.class_im, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

        if bmp_flag == 1:
            file_name = os.path.join(f'{out_dir}', f'wishart_H_A_alpha_class_{n_win_l}x{n_win_c}')
            lib.graphics.bmp_wishart(self.class_im, sub_n_lig, sub_n_col, file_name, color_map_wishart16)

        logging.info('--= Finished: Saving wishart_H_A_alpha classification results bin and bitmap in: %s sec =--' % (datetime.datetime.now() - init_time))

        # END OF THE WISHART H-A-ALPHA CLASSIFICATION
        logging.info('--= Finished: data processing in: %s sec =--' % (datetime.datetime.now() - init_time_data_processing))


def main(*args, **kwargs):
    '''Main function

    Args:
        id (str): input directory
        od (str): output directory
        iodf (str): input-output data forma
        nwr (int): Nwin Row
        nwc (int): Nwin Col
        ofr (int): Offset Row
        ofc (int): Offset Col
        fnr (int): Final Number of Row
        fnc (int): Final Number of Col
        hf (str): ininput entropy file
        af (str): ininput anisotropy file
        alf (str): ininput alpha file
        nit (int): inmaximum interation number
        pct (float): inmaximum of pixel switching classes
        bmp (int): BMP flag (0/1)
        co8 (str): ininput colormap8 file (valid if BMP flag = 1)
        co16 (str, 'input colormap16 file (valid if BMP flag = 1)
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''

    POL_TYPE_VALUES = ['S2', 'C3', 'C4', 'T3', 'T4']
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=POL_TYPE_VALUES)
    parser_args.make_def_args()
    parser_args.add_req_arg('-hf', str, 'ininput entropy file')
    parser_args.add_req_arg('-af', str, 'ininput anisotropy file')
    parser_args.add_req_arg('-alf', str, 'ininput alpha file')
    parser_args.add_req_arg('-nit', int, 'inmaximum interation number')
    parser_args.add_req_arg('-pct', float, 'inmaximum of pixel switching classes')
    parser_args.add_req_arg('-bmp', int, 'BMP flag (0/1)', {0, 1})
    parser_args.add_req_arg('-co8', str, 'ininput colormap8 file (valid if BMP flag = 1)')
    parser_args.add_req_arg('-co16', str, 'input colormap16 file (valid if BMP flag = 1)')

    parsed_args = parser_args.parse_args()
    if parsed_args.bmp == 1 and not (parsed_args.co8 or parsed_args.co16):
        parser_args.print_help()
        sys.exit(1)
    app = App(parsed_args)
    app.run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        dir_in = None
        dir_out = None
        params = {}
        if platform.system().lower().startswith('win') is True:
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\wishart_h_a_alpha_classifier\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\wishart_h_a_alpha_classifier\\py\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/wishart_h_a_alpha_classifier/'
            dir_out = '/home/krzysiek/polsarpro/out/wishart_h_a_alpha_classifier/py/'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.Termination.failure()

        # Pass params as expand dictionary with '**'
        params['id'] = dir_in
        params['od'] = dir_out
        params['iodf'] = 'T3'
        params['nwr'] = 3
        params['nwc'] = 3
        params['ofr'] = 0
        params['ofc'] = 0
        params['fnr'] = 18432
        params['fnc'] = 1248
        params['pct'] = 10
        params['nit'] = 10
        params['bmp'] = 1
        params['co8'] = os.path.join(f'{dir_in}', 'Wishart_ColorMap8.pal')
        params['co16'] = os.path.join(f'{dir_in}', 'Wishart_ColorMap16.pal')
        params['hf'] = os.path.join(f'{dir_in}', 'entropy.bin')
        params['af'] = os.path.join(f'{dir_in}', 'anisotropy.bin')
        params['alf'] = os.path.join(f'{dir_in}', 'alpha.bin')
        params['errf'] = os.path.join(f'{dir_out}', 'MemoryAllocError.txt')
        params['mask'] = os.path.join(f'{dir_in}', 'mask_valid_pixels.bin')
        main(**params)

        # Pass parasm as positional arguments
        # main(id=dir_in,
        #      od=dir_out,
        #      iodf='T3',
        #      nwr=3,
        #      nwc=3,
        #      ofr=0,
        #      ofc=0,
        #      fnr=18432,
        #      fnc=1248,
        #      pct=10,
        #      nit=10,
        #      bmp=1,
        #      co8=os.path.join(f'{dir_in}', 'Wishart_ColorMap8.pal'),
        #      co16=os.path.join(f'{dir_in}', 'Wishart_ColorMap16.pal'),
        #      hf=os.path.join(f'{dir_in}', 'entropy.bin'),
        #      af=os.path.join(f'{dir_in}', 'anisotropy.bin'),
        #      alf=os.path.join(f'{dir_in}', 'alpha.bin'),
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
        #      mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))
