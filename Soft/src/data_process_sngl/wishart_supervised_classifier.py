#!/usr/bin/env python3

'''
PolSARpro v5.0 is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 (1991) of
the License, or any later version. This program is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See the GNU General Public License (Version 2, 1991) for more details

*********************************************************************

File  : wishart_supervised_classifier.py
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

Description :  Supervised maximum likelihood classification of a
polarimetric image with a "don't know class"
- from the Wishart PDF of its coherency matrices
- from the Gaussian PDF of its target vectors
represented under the form of one look coherency matrices

********************************************************************/
'''


import os
import sys
import numpy
import platform
import math
import logging
import datetime
import numba
sys.path.append(r'../')
import lib.util  # noqa: E402
import lib.util_block  # noqa: E402
import lib.util_convert  # noqa: E402
import lib.matrix  # noqa: E402
from lib.graphics import bmp_training_set  # noqa: E402
from lib.processing import inverse_hermitian_matrix2  # noqa: E402
from lib.processing import determinant_hermitian_matrix2  # noqa: E402
from lib.processing import inverse_hermitian_matrix3  # noqa: E402
from lib.processing import determinant_hermitian_matrix3  # noqa: E402
from lib.processing import inverse_hermitian_matrix4  # noqa: E402
from lib.processing import determinant_hermitian_matrix4  # noqa: E402
from lib.processing import trace2_hm1xhm2  # noqa: E402
from lib.processing import trace3_hm1xhm2  # noqa: E402
from lib.processing import trace4_hm1xhm2  # noqa: E402


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


def create_class_map(file_name, class_map):
    with open(file_name, 'r') as f:
        n_class = None
        f.readline().strip()  # skip line
        n_class = int(f.readline().strip())
        nareacoord_l = None  # noqa: F841
        areacoord_c = None  # noqa: F841
        zone = 0
        for classe in range(n_class):
            f.readline().strip()  # skip line
            f.readline().strip()  # skip line
            f.readline().strip()  # skip line
            n_area = int(f.readline().strip())
            for area in range(n_area):
                zone += 1
                class_map[zone] = (float)(classe + 1)
                f.readline().strip()  # skip line
                f.readline().strip()  # skip line
                n_tpt = int(f.readline().strip())
                for t_pt in range(n_tpt):
                    f.readline().strip()  # skip line
                    f.readline().strip()  # skip line
                    areacoord_l = float(f.readline().strip())  # noqa: F841
                    f.readline().strip()  # skip line
                    areacoord_c = float(f.readline().strip())  # noqa: F841


@numba.njit(parallel=False)
def inverse_center_coherency_matrices_computation_ipp(n_area, cov_area_m1, cov_area, n_pp, eps, det_area):
    for area in range(1, n_area):
        for np in range(n_pp):
            cov_area_m1[np][area] = 1 / (cov_area[np][area] + eps)
        det_area[0][area] = cov_area[0][area]
        for np in range(1, n_pp + 1):
            det_area[0][area] = det_area[0][area] * cov_area[np][area]
        det_area[1][area] = 0.


@numba.njit(parallel=False)
def inverse_center_coherency_matrices_computation(n_area, n_pp, eps, det_area, coh, coh_area, pol_type_out, coh_m1, det, coh_area_m1):
    for area in range(1, n_area):
        coh[:n_pp, :n_pp, :2] = coh_area[:n_pp, :n_pp, :2, area]

        if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
            inverse_hermitian_matrix2(coh, coh_m1, eps)
            determinant_hermitian_matrix2(coh, det, eps)
        if pol_type_out in ['C3', 'T3']:
            inverse_hermitian_matrix3(coh, coh_m1)
            determinant_hermitian_matrix3(coh, det, eps)
        if pol_type_out in ['C4', 'T4']:
            inverse_hermitian_matrix4(coh, coh_m1, eps)
            determinant_hermitian_matrix4(coh, det, eps)

        coh_area_m1[:n_pp, :n_pp, :2, area] = coh_m1[:n_pp, :n_pp, :2]
        det_area[0][area] = det[0]
        det_area[1][area] = det[1]


@numba.njit(parallel=False)
def f_ipp(col, n_area, n_pp, cov_area_m1, m_avg, distance, det_area):
    for area in range(1, n_area):
        trace = 0.
        for k in range(n_pp):
            trace += cov_area_m1[k][area] * m_avg[k][col]
        distance[area] = math.log(math.sqrt(det_area[0][area] * det_area[0][area] + det_area[1][area] * det_area[1][area]))
        distance[area] = distance[area] + trace


@numba.njit(parallel=False)
def f_not_ipp(col, n_area, n_pp, coh_area_m1, m_avg, distance, det_area, pol_type_out, eps, m, coh_m1):
    if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
        # Average complex coherency matrix determination
        m[0][0][0] = eps + m_avg[0][col]
        m[0][0][1] = 0.
        m[0][1][0] = eps + m_avg[1][col]
        m[0][1][1] = eps + m_avg[2][col]
        m[1][0][0] = m[0][1][0]
        m[1][0][1] = -m[0][1][1]
        m[1][1][0] = eps + m_avg[3][col]
        m[1][1][1] = 0.
    elif pol_type_out in ['C3', 'T3']:
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
    elif pol_type_out in ['C4', 'T4']:
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
    for area in range(1, n_area):
        # for k in range(n_pp):
        #     for n in range(n_pp):
        #         coh_m1[k][n][0] = coh_area_m1[k][n][0][area]
        #         coh_m1[k][n][1] = coh_area_m1[k][n][1][area]
        coh_m1[:n_pp, :n_pp, :2] = coh_area_m1[:n_pp, :n_pp, :2, area]
        distance[area] = math.log(math.sqrt(det_area[0][area] * det_area[0][area] + det_area[1][area] * det_area[1][area]))
        if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
            distance[area] = distance[area] + trace2_hm1xhm2(coh_m1, m)
        elif pol_type_out in ['C3', 'T3']:
            distance[area] = distance[area] + trace3_hm1xhm2(coh_m1, m)
        elif pol_type_out in ['C4', 'T4']:
            distance[area] = distance[area] + trace4_hm1xhm2(coh_m1, m)


@numba.njit(parallel=False)
def f(sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, pol_type, n_area, n_pp, cov_area_m1, m_avg, distance, det_area, pol_type_out, eps, m, coh_m1, init_minmax, coh_area_m1, lig_g, tmp_class_im, tmp_dist_im, mean_dist_area, mean_dist_area2, cpt_area, class_im, class_map):
    for col in range(sub_n_col):
        if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
            if pol_type == 'IPP':
                f_ipp(col, n_area, n_pp, cov_area_m1, m_avg, distance, det_area)
                # for area in range(1, n_area):
                #     trace = 0
                #     for k in range(n_pp):
                #         trace += cov_area_m1[k][area] * m_avg[k][col]
                #     distance[area] = math.log(math.sqrt(det_area[0][area] * det_area[0][area] + det_area[1][area] * det_area[1][area]))
                #     distance[area] = distance[area] + trace
            else:
                f_not_ipp(col, n_area, n_pp, coh_area_m1, m_avg, distance, det_area, pol_type_out, eps, m, coh_m1)
                # if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                #     # Average complex coherency matrix determination
                #     m[0][0][0] = eps + m_avg[0][col]
                #     m[0][0][1] = 0.
                #     m[0][1][0] = eps + m_avg[1][col]
                #     m[0][1][1] = eps + m_avg[2][col]
                #     m[1][0][0] = m[0][1][0]
                #     m[1][0][1] = -m[0][1][1]
                #     m[1][1][0] = eps + m_avg[3][col]
                #     m[1][1][1] = 0.
                # if pol_type_out in ['C3', 'T3']:
                #     # Average complex coherency matrix determination
                #     m[0][0][0] = eps + m_avg[0][col]
                #     m[0][0][1] = 0.
                #     m[0][1][0] = eps + m_avg[1][col]
                #     m[0][1][1] = eps + m_avg[2][col]
                #     m[0][2][0] = eps + m_avg[3][col]
                #     m[0][2][1] = eps + m_avg[4][col]
                #     m[1][0][0] = m[0][1][0]
                #     m[1][0][1] = -m[0][1][1]
                #     m[1][1][0] = eps + m_avg[5][col]
                #     m[1][1][1] = 0.
                #     m[1][2][0] = eps + m_avg[6][col]
                #     m[1][2][1] = eps + m_avg[7][col]
                #     m[2][0][0] = m[0][2][0]
                #     m[2][0][1] = -m[0][2][1]
                #     m[2][1][0] = m[1][2][0]
                #     m[2][1][1] = -m[1][2][1]
                #     m[2][2][0] = eps + m_avg[8][col]
                #     m[2][2][1] = 0.
                # if pol_type_out in ['C4', 'T4']:
                #     # Average complex coherency matrix determination
                #     m[0][0][0] = eps + m_avg[0][col]
                #     m[0][0][1] = 0.
                #     m[0][1][0] = eps + m_avg[1][col]
                #     m[0][1][1] = eps + m_avg[2][col]
                #     m[0][2][0] = eps + m_avg[3][col]
                #     m[0][2][1] = eps + m_avg[4][col]
                #     m[0][3][0] = eps + m_avg[5][col]
                #     m[0][3][1] = eps + m_avg[6][col]
                #     m[1][0][0] = m[0][1][0]
                #     m[1][0][1] = -m[0][1][1]
                #     m[1][1][0] = eps + m_avg[7][col]
                #     m[1][1][1] = 0.
                #     m[1][2][0] = eps + m_avg[8][col]
                #     m[1][2][1] = eps + m_avg[9][col]
                #     m[1][3][0] = eps + m_avg[10][col]
                #     m[1][3][1] = eps + m_avg[11][col]
                #     m[2][0][0] = m[0][2][0]
                #     m[2][0][1] = -m[0][2][1]
                #     m[2][1][0] = m[1][2][0]
                #     m[2][1][1] = -m[1][2][1]
                #     m[2][2][0] = eps + m_avg[12][col]
                #     m[2][2][1] = 0.
                #     m[2][3][0] = eps + m_avg[13][col]
                #     m[2][3][1] = eps + m_avg[14][col]
                #     m[3][0][0] = m[0][3][0]
                #     m[3][0][1] = -m[0][3][1]
                #     m[3][1][0] = m[1][3][0]
                #     m[3][1][1] = -m[1][3][1]
                #     m[3][2][0] = m[2][3][0]
                #     m[3][2][1] = -m[2][3][1]
                #     m[3][3][0] = eps + m_avg[15][col]
                #     m[3][3][1] = 0.

                # # Seeking for the closest cluster center
                # for area in range(1, n_area):
                #     for k in range(n_pp):
                #         for l in range(n_pp):
                #             coh_m1[k][l][0] = coh_area_m1[k][l][0][area]
                #             coh_m1[k][l][1] = coh_area_m1[k][l][1][area]
                #     distance[area] = math.log(math.sqrt(det_area[0][area] * det_area[0][area] + det_area[1][area] * det_area[1][area]))
                #     if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                #         distance[area] = distance[area] + trace2_hm1xhm2(coh_m1, m)
                #     if pol_type_out in ['C3', 'T3']:
                #         distance[area] = distance[area] + trace3_hm1xhm2(coh_m1, m)
                #     if pol_type_out in ['C4', 'T4']:
                #         distance[area] = distance[area] + trace4_hm1xhm2(coh_m1, m)

            dist_min = init_minmax
            for area in range(1, n_area):
                if dist_min > distance[area]:
                    dist_min = distance[area]
                    tmp_class_im[lig_g][col] = area
            tmp_dist_im[lig_g][col] = dist_min
            mean_dist_area[(int)(tmp_class_im[lig_g][col])] += dist_min
            mean_dist_area2[(int)(tmp_class_im[lig_g][col])] += dist_min * dist_min
            cpt_area[(int)(tmp_class_im[lig_g][col])] += 1

            class_im[lig_g][col] = class_map[(int)(tmp_class_im[lig_g][col])]
        else:
            class_im[lig_g][col] = 0.


@numba.njit(parallel=True, fastmath=True)
def wishart_supervised_classifieri_alg(n_lig_g, n_win_l, n_win_c, n_lig_blocki_nb, m_in, n_polar_out, sub_n_col, valid, n_win_l_m1s2, n_win_c_m1s2, pol_type, n_area, n_pp, cov_area_m1, distance, det_area, pol_type_out, eps, init_minmax, coh_area_m1, lig_g, tmp_class_im, tmp_dist_im, mean_dist_area, mean_dist_area2, cpt_area, class_im, class_map):
    ligDone = 0
    for lig in numba.prange(n_lig_blocki_nb):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_blocki_nb)
        m = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
        coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
        m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        lig_g = lig + n_lig_g
        f(sub_n_col, valid, n_win_l_m1s2, lig, n_win_c_m1s2, pol_type, n_area, n_pp, cov_area_m1, m_avg, distance, det_area, pol_type_out, eps, m, coh_m1, init_minmax, coh_area_m1, lig_g, tmp_class_im, tmp_dist_im, mean_dist_area, mean_dist_area2, cpt_area, class_im, class_map)


class App(lib.util.Application):

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col, sub_n_lig):
        '''
        Allocate matrices with given dimensions
        '''
        logging.info(f'{n_col=}, {n_polar_out=}, {n_win_l=}, {n_win_c=}, {n_lig_block=}, {sub_n_col=}')
        self.vc_in = lib.matrix.vector_float(2 * n_col)
        self.vf_in = lib.matrix.vector_float(n_col)
        self.mc_in = lib.matrix.matrix_float(4, 2 * n_col)
        self.mf_in = lib.matrix.matrix3d_float(n_polar_out, n_win_l, n_col + n_win_c)

        self.valid = lib.matrix.matrix_float(n_lig_block + n_win_l, sub_n_col + n_win_c)

        self.m_in = lib.matrix.matrix3d_float(n_polar_out, n_lig_block + n_win_l, n_col + n_win_c)
        self.m_trn = lib.matrix.vector_float(n_polar_out)
        self.class_im = lib.matrix.matrix_float(sub_n_lig, n_col)
        self.tmp_class_im = lib.matrix.matrix_float(sub_n_lig, n_col)
        self.tmp_dist_im = lib.matrix.matrix_float(sub_n_lig, n_col)

    def run(self):
        logging.info('******************** Welcome in wishart supervised classifier ********************')
        logging.info(self.args)
        in_dir = self.args.id
        out_dir = self.args.od
        pol_type = self.args.iodf
        file_area = self.args.af
        n_win_l = self.args.nwr
        n_win_c = self.args.nwc
        off_lig = self.args.ofr
        off_col = self.args.ofc
        sub_n_lig = self.args.fnr
        sub_n_col = self.args.fnc
        file_cluster = self.args.cf
        bmp_flag = self.args.bmp
        color_map_training_set16 = None
        if bmp_flag == 1:
            color_map_training_set16 = self.args.col
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

        if flag_valid is True:
            self.check_file(file_valid)

        self.check_file(file_area)
        self.check_file(file_cluster)

        n_win_l_m1s2 = (n_win_l - 1) // 2
        logging.info(f'{n_win_l_m1s2=}')
        n_win_c_m1s2 = (n_win_c - 1) // 2
        logging.info(f'{n_win_c_m1s2=}')

        # INPUT/OUPUT CONFIGURATIONS
        n_lig, n_col, polar_case, polar_type = lib.util.read_config(in_dir)
        logging.info(f'{n_lig=}, {n_col=}, {polar_case=}, {polar_type=}')

        # POLAR TYPE CONFIGURATION */
        if pol_type == 'S2':
            if polar_case == 'monostatic':
                pol_type = 'S2T3'
            if polar_case == 'bistatic':
                pol_type = 'S2T4'
        if pol_type == 'SPP':
            pol_type = 'SPPC2'

        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type)
        logging.info(f'{pol_type=}, {n_polar_in=}, {pol_type_in=}, {n_polar_out=}, {pol_type_out=}')

        file_name_in = lib.util.init_file_name(pol_type_in, in_dir)

        # INPUT FILE OPENING
        in_datafile = []
        in_valid = None
        in_datafile, in_valid, flag_valid = self.open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

        trn_file = self.open_file(file_cluster, 'rb')

        # OUTPUT FILE OPENING
        file_name_out = [
            os.path.join(f'{out_dir}', f'wishart_supervised_class_{n_win_l}x{n_win_c}.bin'),
            os.path.join(f'{out_dir}', 'wishart_training_cluster_centers.txt'),
        ]
        logging.info(f'{file_name_out=}')

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # ClassIm = sub_n_lig,*sub_n_col
        n_block_a += 0
        n_block_b += sub_n_lig * sub_n_col
        # TMPclassIm = Sub_Nlig*Sub_Ncol
        n_block_a += 0
        n_block_b += sub_n_lig * sub_n_col
        # TMPdistIm = Sub_Nlig*Sub_Ncol
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

        eps = lib.util.Application.EPS
        init_minmax = lib.util.Application.INIT_MINMAX

        # DATA PROCESSING
        logging.info('--= Started: data processing =--')
        init_time = datetime.datetime.now()

        # Number of learning clusters reading
        self.m_trn = numpy.fromfile(trn_file, dtype=numpy.float32, count=1)
        n_area = (int)(self.m_trn[0] + 1)
        logging.info(f'{n_area=}')
        class_map = lib.matrix.vector_float(n_area + 1)

        create_class_map(file_area, class_map)

        # Training class matrix memory allocation
        if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3', 'IPPpp5', 'IPPpp6', 'IPPpp7']:
            n_pp = 2
        if pol_type_out in ['C3', 'T3', 'IPPpp4']:
            n_pp = 3
        if pol_type_out in ['C4', 'T4']:
            n_pp = 4

        det = None
        coh = None

        cov_area = None
        cov_area_m1 = None
        tmp_cov_area = [None] * n_pp
        tmp_cov_area_m1 = [None] * n_pp

        coh_area = None
        coh_area_m1 = None
        tmp_coh_area = [[[None] * 2] * n_pp] * n_pp
        tmp_coh_area_m1 = [[[None] * 2] * n_pp] * n_pp

        det_area = None
        tmp_det_area = [None] * 2

        # if pol_type == 'IPP':
        #     for k in range(n_pp):
        #         tmp_cov_area[k] = lib.matrix.vector_float(n_area)
        #         tmp_cov_area_m1[k] = lib.matrix.vector_float(n_area)
        #     cov_area = numpy.array(tmp_cov_area)
        #     cov_area_m1 = numpy.array(tmp_cov_area_m1)
        # else:
        #     det = lib.matrix.vector_float(2)
        #     coh = lib.matrix.matrix3d_float(n_pp, n_pp, 2)
        #     for k in range(n_pp):
        #         for l in range(n_pp):
        #             tmp_coh_area[k][l][0] = lib.matrix.vector_float(n_area)
        #             tmp_coh_area[k][l][1] = lib.matrix.vector_float(n_area)
        #             tmp_coh_area_m1[k][l][0] = lib.matrix.vector_float(n_area)
        #             tmp_coh_area_m1[k][l][1] = lib.matrix.vector_float(n_area)
        #     coh_area = numpy.array(tmp_coh_area)
        #     coh_area_m1 = numpy.array(tmp_coh_area_m1)
        logging.info(f'{n_pp=}')

        for k in range(n_pp):
            tmp_cov_area[k] = lib.matrix.vector_float(n_area)
            tmp_cov_area_m1[k] = lib.matrix.vector_float(n_area)
        cov_area = numpy.array(tmp_cov_area)
        cov_area_m1 = numpy.array(tmp_cov_area_m1)

        det = lib.matrix.vector_float(2)
        coh = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

        for k in range(n_pp):
            for n in range(n_pp):
                tmp_coh_area[k][n][0] = lib.matrix.vector_float(n_area)
                tmp_coh_area[k][n][1] = lib.matrix.vector_float(n_area)
                tmp_coh_area_m1[k][n][0] = lib.matrix.vector_float(n_area)
                tmp_coh_area_m1[k][n][1] = lib.matrix.vector_float(n_area)
        coh_area = numpy.array(tmp_coh_area)
        coh_area_m1 = numpy.array(tmp_coh_area_m1)

        tmp_det_area[0] = lib.matrix.vector_float(n_area)
        tmp_det_area[1] = lib.matrix.vector_float(n_area)
        det_area = numpy.array(tmp_det_area)

        cpt_area = lib.matrix.vector_float(100)
        mean_dist_area = lib.matrix.vector_float(100)
        mean_dist_area2 = lib.matrix.vector_float(100)
        std_dist_area = lib.matrix.vector_float(100)
        distance = lib.matrix.vector_float(100)

        #  TRAINING CLUSTER CENTERS READING
        logging.info('--= Started: TRAINING CLUSTER CENTERS READING =--')
        init_time = datetime.datetime.now()

        for area in range(1, n_area):
            if pol_type == 'IPP':
                self.m_trn = numpy.fromfile(trn_file, dtype=numpy.float32, count=n_pp)
                for np in range(n_pp):
                    cov_area[np][area] = eps + self.m_trn[np]
            else:
                self.m_trn = numpy.fromfile(trn_file, dtype=numpy.float32, count=n_polar_out)
                if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                    coh_area[0][0][0][area] = eps + self.m_trn[lib.util.C211]
                    coh_area[0][0][1][area] = 0.
                    coh_area[0][1][0][area] = eps + self.m_trn[lib.util.C212_RE]
                    coh_area[0][1][1][area] = eps + self.m_trn[lib.util.C212_IM]
                    coh_area[1][0][0][area] = eps + self.m_trn[lib.util.C212_RE]
                    coh_area[1][0][1][area] = eps - self.m_trn[lib.util.C212_IM]
                    coh_area[1][1][0][area] = eps + self.m_trn[lib.util.C222]
                    coh_area[1][1][1][area] = 0.
                elif pol_type_out in ['C3', 'T3']:
                    coh_area[0][0][0][area] = eps + self.m_trn[lib.util.X311]
                    coh_area[0][0][1][area] = 0.
                    coh_area[0][1][0][area] = eps + self.m_trn[lib.util.X312_RE]
                    coh_area[0][1][1][area] = eps + self.m_trn[lib.util.X312_IM]
                    coh_area[0][2][0][area] = eps + self.m_trn[lib.util.X313_RE]
                    coh_area[0][2][1][area] = eps + self.m_trn[lib.util.X313_IM]
                    coh_area[1][0][0][area] = eps + self.m_trn[lib.util.X312_RE]
                    coh_area[1][0][1][area] = eps - self.m_trn[lib.util.X312_IM]
                    coh_area[1][1][0][area] = eps + self.m_trn[lib.util.X322]
                    coh_area[1][1][1][area] = 0.
                    coh_area[1][2][0][area] = eps + self.m_trn[lib.util.X323_RE]
                    coh_area[1][2][1][area] = eps + self.m_trn[lib.util.X323_IM]
                    coh_area[2][0][0][area] = eps + self.m_trn[lib.util.X313_RE]
                    coh_area[2][0][1][area] = eps - self.m_trn[lib.util.X313_IM]
                    coh_area[2][1][0][area] = eps + self.m_trn[lib.util.X323_RE]
                    coh_area[2][1][1][area] = eps - self.m_trn[lib.util.X323_IM]
                    coh_area[2][2][0][area] = eps + self.m_trn[lib.util.X333]
                    coh_area[2][2][1][area] = 0.
                elif pol_type_out in ['C4', 'T4']:
                    coh_area[0][0][0][area] = eps + self.m_trn[lib.util.X411]
                    coh_area[0][0][1][area] = 0.
                    coh_area[0][1][0][area] = eps + self.m_trn[lib.util.X412_RE]
                    coh_area[0][1][1][area] = eps + self.m_trn[lib.util.X412_IM]
                    coh_area[0][2][0][area] = eps + self.m_trn[lib.util.X413_RE]
                    coh_area[0][2][1][area] = eps + self.m_trn[lib.util.X413_IM]
                    coh_area[0][3][0][area] = eps + self.m_trn[lib.util.X414_RE]
                    coh_area[0][3][1][area] = eps + self.m_trn[lib.util.X414_IM]

                    coh_area[1][0][0][area] = eps + self.m_trn[lib.util.X412_RE]
                    coh_area[1][0][1][area] = eps - self.m_trn[lib.util.X412_IM]
                    coh_area[1][1][0][area] = eps + self.m_trn[lib.util.X422]
                    coh_area[1][1][1][area] = 0.
                    coh_area[1][2][0][area] = eps + self.m_trn[lib.util.X423_RE]
                    coh_area[1][2][1][area] = eps + self.m_trn[lib.util.X423_IM]
                    coh_area[1][3][0][area] = eps + self.m_trn[lib.util.X424_RE]
                    coh_area[1][3][1][area] = eps + self.m_trn[lib.util.X424_IM]

                    coh_area[2][0][0][area] = eps + self.m_trn[lib.util.X413_RE]
                    coh_area[2][0][1][area] = eps - self.m_trn[lib.util.X413_IM]
                    coh_area[2][1][0][area] = eps + self.m_trn[lib.util.X423_RE]
                    coh_area[2][1][1][area] = eps - self.m_trn[lib.util.X423_IM]
                    coh_area[2][2][0][area] = eps + self.m_trn[lib.util.X433]
                    coh_area[2][2][1][area] = 0.
                    coh_area[2][3][0][area] = eps + self.m_trn[lib.util.X434_RE]
                    coh_area[2][3][1][area] = eps + self.m_trn[lib.util.X434_IM]

                    coh_area[3][0][0][area] = eps + self.m_trn[lib.util.X414_RE]
                    coh_area[3][0][1][area] = eps - self.m_trn[lib.util.X414_IM]
                    coh_area[3][1][0][area] = eps + self.m_trn[lib.util.X424_RE]
                    coh_area[3][1][1][area] = eps - self.m_trn[lib.util.X424_IM]
                    coh_area[3][2][0][area] = eps + self.m_trn[lib.util.X434_RE]
                    coh_area[3][2][1][area] = eps - self.m_trn[lib.util.X434_IM]
                    coh_area[3][3][0][area] = eps + self.m_trn[lib.util.X444]
                    coh_area[3][3][1][area] = 0.

            mean_dist_area[area] = 0
            mean_dist_area2[area] = 0
            std_dist_area[area] = 0

        logging.info('--= Finished: TRAINING CLUSTER CENTERS READING in: %s sec =--' % (datetime.datetime.now() - init_time))

        # save cluster center in text file
        logging.info('--= Started: save cluster center in text file =--')
        init_time = datetime.datetime.now()

        with open(file_name_out[1], mode='wb') as fp:
            for area in range(1, n_area):
                fp.write(f'cluster centre # {area}\n'.encode('ascii'))
                if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                    fp.write(f'C11 = {coh_area[0][0][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'C12 = {coh_area[0][1][0][area]:e} + j {coh_area[0][1][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C22 = {coh_area[1][1][0][area]:e}\n'.encode('ascii'))
                    fp.write('\n'.encode('ascii'))
                elif pol_type_out == 'C3':
                    fp.write(f'C11 = {coh_area[0][0][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'C12 = {coh_area[0][1][0][area]:e} + j {coh_area[0][1][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C13 = {coh_area[0][2][0][area]:e} + j {coh_area[0][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C22 = {coh_area[1][1][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'C23 = {coh_area[1][2][0][area]:e} + j {coh_area[1][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C33 = {coh_area[2][2][0][area]:e}\n'.encode('ascii'))
                    fp.write('\n'.encode('ascii'))
                elif pol_type_out == 'T3':
                    fp.write(f'T11 = {coh_area[0][0][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'T12 = {coh_area[0][1][0][area]:e} + j {coh_area[0][1][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T13 = {coh_area[0][2][0][area]:e} + j {coh_area[0][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T22 = {coh_area[1][1][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'T23 = {coh_area[1][2][0][area]:e} + j {coh_area[1][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T33 = {coh_area[2][2][0][area]:e}\n'.encode('ascii'))
                    fp.write('\n'.encode('ascii'))
                elif pol_type_out == 'C4':
                    fp.write(f'C11 = {coh_area[0][0][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'C12 = {coh_area[0][1][0][area]:e} + j {coh_area[0][1][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C13 = {coh_area[0][2][0][area]:e} + j {coh_area[0][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C14 = {coh_area[0][3][0][area]:e} + j {coh_area[0][3][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C22 = {coh_area[1][1][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'C23 = {coh_area[1][2][0][area]:e} + j {coh_area[1][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C24 = {coh_area[1][3][0][area]:e} + j {coh_area[1][3][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C33 = {coh_area[2][2][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'C34 = {coh_area[2][3][0][area]:e} + j {coh_area[2][3][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'C44 = {coh_area[3][3][0][area]:e}\n'.encode('ascii'))
                    fp.write('\n'.encode('ascii'))
                elif pol_type_out == 'T4':
                    fp.write(f'T11 = {coh_area[0][0][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'T12 = {coh_area[0][1][0][area]:e} + j {coh_area[0][1][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T13 = {coh_area[0][2][0][area]:e} + j {coh_area[0][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T14 = {coh_area[0][3][0][area]:e} + j {coh_area[0][3][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T22 = {coh_area[1][1][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'T23 = {coh_area[1][2][0][area]:e} + j {coh_area[1][2][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T24 = {coh_area[1][3][0][area]:e} + j {coh_area[1][3][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T33 = {coh_area[2][2][0][area]:e}\n'.encode('ascii'))
                    fp.write(f'T34 = {coh_area[2][3][0][area]:e} + j {coh_area[2][3][1][area]:e}\n'.encode('ascii'))
                    fp.write(f'T44 = {coh_area[3][3][0][area]:e}\n'.encode('ascii'))
                    fp.write('\n'.encode('ascii'))
                elif pol_type_out == 'IPPpp4':
                    fp.write(f'I11 = {cov_area[0][area]:e}\n'.encode('ascii'))
                    fp.write(f'I12 = {cov_area[1][area]:e}\n'.encode('ascii'))
                    fp.write(f'I22 = {cov_area[2][area]:e}\n'.encode('ascii'))
                elif pol_type_out == 'IPPpp5':
                    fp.write(f'I11 = {cov_area[0][area]:e}\n'.encode('ascii'))
                    fp.write(f'I21 = {cov_area[1][area]:e}\n'.encode('ascii'))
                elif pol_type_out == 'IPPpp6':
                    fp.write(f'I12 = {cov_area[0][area]:e}\n'.encode('ascii'))
                    fp.write(f'I22 = {cov_area[1][area]:e}\n'.encode('ascii'))
                elif pol_type_out == 'IPPpp7':
                    fp.write(f'I11 = {cov_area[0][area]:e}\n'.encode('ascii'))
                    fp.write(f'I22 = {cov_area[1][area]:e}\n'.encode('ascii'))
                elif pol_type_out == 'IPPfull':
                    fp.write(f'I11 = {cov_area[0][area]:e}\n'.encode('ascii'))
                    fp.write(f'I12 = {cov_area[1][area]:e}\n'.encode('ascii'))
                    fp.write(f'I21 = {cov_area[2][area]:e}\n'.encode('ascii'))
                    fp.write(f'I22 = {cov_area[3][area]:e}\n'.encode('ascii'))

        logging.info('--= Finished: save cluster center in text file in: %s sec =--' % (datetime.datetime.now() - init_time))

        coh_m1 = lib.matrix.matrix3d_float(n_pp, n_pp, 2)

        # Inverse center coherency matrices computation
        logging.info('--= Started: Inverse center coherency matrices computation =--')
        init_time = datetime.datetime.now()
        if pol_type == 'IPP':
            inverse_center_coherency_matrices_computation_ipp(n_area, cov_area_m1, cov_area, n_pp, eps, det_area)
        else:
            inverse_center_coherency_matrices_computation(n_area, n_pp, eps, det_area, coh, coh_area, pol_type_out, coh_m1, det, coh_area_m1)
        logging.info('--= Finished: Inverse center coherency matrices computation in: %s sec =--' % (datetime.datetime.now() - init_time))

        lig_g = 0
        n_lig_g = 0

        logging.info('--= Started: supervised classification results =--')
        init_time = datetime.datetime.now()

        for nb in range(nb_block):
            if nb_block > 2:
                lib.util.printf_line(nb, nb_block)

            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type_in in ['S2', 'SPP', 'SPPpp1', 'SPPpp2', 'SPPpp3']:
                if pol_type_in == 'S2':
                    lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                else:
                    lib.util_block.read_block_spp_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # Case of C,T or I
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            wishart_supervised_classifieri_alg(n_lig_g, n_win_l, n_win_c, n_lig_block[nb], self.m_in, n_polar_out, sub_n_col, self.valid, n_win_l_m1s2, n_win_c_m1s2, pol_type, n_area, n_pp, cov_area_m1, distance, det_area, pol_type_out, eps, init_minmax, coh_area_m1, lig_g, self.tmp_class_im, self.tmp_dist_im, mean_dist_area, mean_dist_area2, cpt_area, self.class_im, class_map)

            n_lig_g += n_lig_block[nb]

        logging.info('--= Finished: supervised classification results in: %s sec =--' % (datetime.datetime.now() - init_time))

        # Saving supervised classification results bin
        logging.info('--= Started: Saving supervised classification results bin =--')
        init_time = datetime.datetime.now()

        with open(file_name_out[0], mode='wb') as class_file:
            lib.util_block.write_block_matrix_float(class_file, self.class_im, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

        logging.info('--= Started: Create BMP file =--')
        # Create BMP file
        if bmp_flag == 1:
            file_name = os.path.join(f'{out_dir}', f'wishart_supervised_class_{n_win_l}x{n_win_c}')
            bmp_training_set(self.class_im, sub_n_lig, sub_n_col, file_name, color_map_training_set16)

        logging.info('--= Finished: Saving supervised classification results bin in: %s sec =--' % (datetime.datetime.now() - init_time))

        logging.info('--= Finished: data processing in: %s sec =--' % (datetime.datetime.now() - init_time))


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
        af (str): input area file
        cf (str): input cluster file
        bmp (int): BMP flag (0/1)
        col (str): ininput colormap file (valid if BMP flag = 1)
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''

    POL_TYPE_VALUES = ['S2', 'C2', 'C3', 'C4', 'T3', 'T4', 'SPP', 'IPP']
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=POL_TYPE_VALUES)
    parser_args.make_def_args()
    parser_args.add_req_arg('-af', str, 'input area file')
    parser_args.add_req_arg('-cf', str, 'input cluster file')
    parser_args.add_req_arg('-bmp', int, 'BMP flag (0/1)', {0, 1})
    parser_args.add_req_arg('-col', str, 'ininput colormap file (valid if BMP flag = 1)')
    parsed_args = parser_args.parse_args()
    app = App(parsed_args)
    app.run()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        dir_in = None
        dir_out = None
        params = {}
        if platform.system().lower().startswith('win') is True:
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\wishart_supervised_classifier\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\wishart_supervised_classifier\\py\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/wishart_supervised_classifier/'
            dir_out = '/home/krzysiek/polsarpro/out/wishart_supervised_classifier/py/'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()

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
        params['bmp'] = 1
        params['col'] = os.path.join(f'{dir_in}', 'Supervised_ColorMap16.pal')
        params['af'] = os.path.join(f'{dir_in}', 'wishart_training_areas_eric.txt')
        params['cf'] = os.path.join(f'{dir_in}', 'wishart_training_cluster_centers.bin')
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
        #      bmp=1,
        #      col=os.path.join(f'{dir_in}', 'Supervised_ColorMap16.pal'),
        #      af=os.path.join(f'{dir_in}', 'wishart_training_areas_eric.txt'),
        #      cf=os.path.join(f'{dir_in}', 'wishart_training_cluster_centers.bin'),
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
        #      mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))
