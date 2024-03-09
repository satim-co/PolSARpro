#!/usr/bin/env python3

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

File  : id_class_gen.py
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

Description :  Basic identification of the classes resulting of a
               Unsupervised H / A / Alpha - Wishart segmentation

********************************************************************
'''


import os
import sys
import platform
import numpy
import math
import logging
import datetime
import numba
sys.path.append(r'../')
import lib.util  # noqa: E402
import lib.util_block  # noqa: E402
import lib.util_convert  # noqa: E402
import lib.matrix  # noqa: E402
import lib.graphics  # noqa: E402

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


# CONSTANTS
LIM_H1 = 0.85
LIM_H2 = 0.5
LIM_A = 0.5

# ALIASES
NCLASS_POL = 100
ent = 0
anis = 1
al1 = 2
al2 = 3
be1 = 4
be2 = 5
pr1 = 6
pr2 = 7
CL_H_A = 0
CL_AL1 = 1
CL_AL1_AL2 = 2


@numba.njit(parallel=False, fastmath=True)
def data_processing_routine(_nb, _n_lig_block, _n_lig_g, _sub_n_col, _class_out, _class_in, _valid, _mh_in, _ma_in, _mal1_in, _mal2_in, _mbe1_in, _mbe2_in, _mp1_in, _mp2_in, _cpt_h_a, _cpt_al1, _cpt_al1_al2, _pi):
    lig_g = 0
    for lig in range(_n_lig_block[_nb]):
        lig_g = lig + _n_lig_g
        with numba.objmode():
            lib.util.printf_line(lig, _n_lig_block[_nb])
        for col in range(_sub_n_col):
            _class_out[lig_g][col] = 0.
            if _valid[lig][col] == 1.:
                h1 = (_mh_in[lig][col] <= LIM_H1)
                h2 = (_mh_in[lig][col] <= LIM_H2)
                a1 = (_ma_in[lig][col] <= LIM_A)

                # ZONE 1 (top right)
                r1 = (not h1) * (not a1)
                # ZONE 2 (bottom right)
                r2 = (not h1) * a1
                # ZONE 3 (top center)
                r3 = h1 * (not h2) * (not a1)
                # ZONE 4 (bottom center)
                r4 = h1 * (not h2) * a1
                # ZONE 1 (top left)
                r5 = h2 * (not a1)
                # ZONE 2 (bottom left)
                r6 = h2 * a1

                # segment values ranging from 1 to 9
                class_H_A = (float)(r6 * 11 + r5 * 10 + r4 * 5 + r3 * 6 + r2 * 1 + r1 * 2)
                _class_out[lig_g][col] = class_H_A
                _mal1_in[lig][col] *= _pi / 180
                _mal2_in[lig][col] *= _pi / 180
                _mbe1_in[lig][col] *= _pi / 180
                _mbe2_in[lig][col] *= _pi / 180
                class_al1 = (_mal1_in[lig][col] < _pi / 4.)
                bid1 = _mp1_in[lig][col] * math.cos(_mal1_in[lig][col]) + _mp2_in[lig][col] * math.cos(_mal2_in[lig][col])
                bid2 = _mp1_in[lig][col] * math.sin(_mal1_in[lig][col]) * math.cos(_mbe1_in[lig][col]) + _mp2_in[lig][col] * math.sin(_mal2_in[lig][col]) * math.cos(_mbe2_in[lig][col])
                class_al1_al2 = bid1 > bid2

                if class_H_A == 0:
                    class_H_A = 0.0
                if class_H_A == 1:
                    class_H_A = 2.0
                if class_H_A == 2:
                    class_H_A = 2.0
                if class_H_A == 5:
                    class_H_A = 0.0
                if class_H_A == 6:
                    class_H_A = 1.0
                if class_H_A == 10:
                    class_H_A = 0.0
                if class_H_A == 11:
                    class_H_A = 0.0

                _cpt_h_a[(int)(_class_in[lig][col])][(int)(class_H_A)] = _cpt_h_a[(int)(_class_in[lig][col])][(int)(class_H_A)] + 1.
                _cpt_al1[(int)(_class_in[lig][col])][(int)(class_al1)] = _cpt_al1[(int)(_class_in[lig][col])][(int)(class_al1)] + 1.
                _cpt_al1_al2[(int)(_class_in[lig][col])][(int)(class_al1_al2)] = _cpt_al1_al2[(int)(_class_in[lig][col])][(int)(class_al1_al2)] + 1.


@numba.njit(parallel=True, fastmath=True)
def data_processing_routine_id_class(_nb, _n_lig_block, _n_lig_g, _sub_n_col, _class_out, _class_in, _valid, _class_vec):
    lig_g = 0
    for lig in range(_n_lig_block[_nb]):
        if numba_get_thread_id() == 0:
            lib.util.printf_line(lig, _n_lig_block[_nb])
        lig_g = lig + _n_lig_g
        for col in numba.prange(_sub_n_col):
            _class_out[lig_g][col] = 0.
            if _valid[lig][col] == 1.:
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 2:
                    class_type = 1
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 0:
                    class_type = 6 - _class_vec[(int)(_class_in[lig][col])][CL_AL1]
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 1:
                    class_type = 14 - 2 * _class_vec[(int)(_class_in[lig][col])][CL_AL1_AL2]
                _class_out[lig_g][col] = class_type


@numba.njit(parallel=True, fastmath=True)
def data_processing_routine_vol_class(_nb, _n_lig_block, _n_lig_g, _sub_n_col, _class_out, _class_in, _valid, _class_vec):
    lig_g = 0
    for lig in range(_n_lig_block[_nb]):
        if numba_get_thread_id() == 0:
            lib.util.printf_line(lig, _n_lig_block[_nb])
        lig_g = lig + _n_lig_g
        for col in numba.prange(_sub_n_col):
            _class_out[lig_g][col] = 0.
            if _valid[lig][col] == 1.:
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 2:
                    class_type = 1
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 0:
                    class_type = 6 - _class_vec[(int)(_class_in[lig][col])][CL_AL1]
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 1:
                    class_type = 14 - 2 * _class_vec[(int)(_class_in[lig][col])][CL_AL1_AL2]
                _class_out[lig_g][col] = (class_type == 1) * 2


@numba.njit(parallel=True, fastmath=True)
def data_processing_routine_sgl_class(_nb, _n_lig_block, _n_lig_g, _sub_n_col, _class_out, _class_in, _valid, _class_vec):
    lig_g = 0
    for lig in range(_n_lig_block[_nb]):
        if numba_get_thread_id() == 0:
            lib.util.printf_line(lig, _n_lig_block[_nb])
        lig_g = lig + _n_lig_g
        for col in numba.prange(_sub_n_col):
            _class_out[lig_g][col] = 0.
            if _valid[lig][col] == 1.:
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 2:
                    class_type = 1
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 0:
                    class_type = 6 - _class_vec[(int)(_class_in[lig][col])][CL_AL1]
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 1:
                    class_type = 14 - 2 * _class_vec[(int)(_class_in[lig][col])][CL_AL1_AL2]
                _class_out[lig_g][col] = (class_type == 12) * 12 + (class_type == 5) * 5


@numba.njit(parallel=True, fastmath=True)
def data_processing_routine_dbl_class(_nb, _n_lig_block, _n_lig_g, _sub_n_col, _class_out, _class_in, _valid, _class_vec):
    lig_g = 0
    for lig in range(_n_lig_block[_nb]):
        if numba_get_thread_id() == 0:
            lib.util.printf_line(lig, _n_lig_block[_nb])
        lig_g = lig + _n_lig_g
        for col in numba.prange(_sub_n_col):
            _class_out[lig_g][col] = 0.
            if _valid[lig][col] == 1.:
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 2:
                    class_type = 1
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 0:
                    class_type = 6 - _class_vec[(int)(_class_in[lig][col])][CL_AL1]
                if _class_vec[(int)(_class_in[lig][col])][CL_H_A] == 1:
                    class_type = 14 - 2 * _class_vec[(int)(_class_in[lig][col])][CL_AL1_AL2]
                _class_out[lig_g][col] = (class_type == 14) * 14 + (class_type == 6) * 6


@numba.njit(parallel=False, fastmath=True)
def data_processing_routine_max(_n_class, _cpt_h_a, _class_vec, _cpt_al1, _cpt_al1_al2, _init_minmax):
    my_max = 0.0
    for lig in range(_n_class):
        my_max = -_init_minmax
        for col in range(NCLASS_POL):
            if _cpt_h_a[lig][col] > my_max:
                my_max = _cpt_h_a[lig][col]
                _class_vec[lig][CL_H_A] = col

    for lig in range(_n_class):
        my_max = -_init_minmax
        for col in range(NCLASS_POL):
            if _cpt_al1[lig][col] > my_max:
                my_max = _cpt_al1[lig][col]
                _class_vec[lig][CL_AL1] = col

    for lig in range(_n_class):
        my_max = -_init_minmax
        for col in range(NCLASS_POL):
            if _cpt_al1_al2[lig][col] > my_max:
                my_max = _cpt_al1_al2[lig][col]
                _class_vec[lig][CL_AL1_AL2] = col


class App(lib.util.Application):

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col, sub_n_lig, nclass_pol):
        '''
        Allocate matrices with given dimensions
        '''
        logging.info(f'{n_col=}, {n_polar_out=}, {n_win_l=}, {n_win_c=}, {n_lig_block=}, {sub_n_col=}, {sub_n_lig=}, {nclass_pol=}')
        self.vc_in = lib.matrix.vector_float(2 * n_col)
        self.vf_in = lib.matrix.vector_float(n_col)
        self.mc_in = lib.matrix.matrix_float(4, 2 * n_col)
        self.mf_in = lib.matrix.matrix3d_float(n_polar_out, n_win_l, n_col + n_win_c)
        self.valid = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)

        self.mh_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.ma_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.mal1_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.mal2_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.mbe1_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.mbe2_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.mp1_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.mp2_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.class_in = lib.matrix.matrix_float(n_lig_block, sub_n_col + 1)
        self.class_out = lib.matrix.matrix_float(sub_n_lig, sub_n_col + 1)

        self.cpt_h_a = lib.matrix.matrix_float(nclass_pol, nclass_pol)
        self.cpt_al1 = lib.matrix.matrix_float(nclass_pol, nclass_pol)
        self.cpt_al1_al2 = lib.matrix.matrix_float(nclass_pol, nclass_pol)
        self.class_vec = lib.matrix.matrix_float(nclass_pol, nclass_pol)

    def run(self):
        logging.info('******************** Welcome in id class gen ********************')
        logging.info(self.args)

        in_dir = self.args.id
        out_dir = self.args.od
        off_lig = self.args.ofr
        off_col = self.args.ofc
        sub_n_lig = self.args.fnr
        sub_n_col = self.args.fnc
        in_class_name = self.args.icf
        color_map_wishart = self.args.clm
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

        n_win_l = 1
        n_win_c = 1
        n_win_l_m1s2 = (n_win_l - 1) // 2
        logging.info(f'{n_win_l_m1s2=}')
        n_win_c_m1s2 = (n_win_c - 1) // 2
        logging.info(f'{n_win_c_m1s2=}')

        # INPUT/OUPUT CONFIGURATIONS
        n_lig, n_col, polar_case, polar_type = lib.util.read_config(in_dir)
        logging.info(f'{n_lig=}, {n_col=}, {polar_case=}, {polar_type=}')

        # INPUT/OUTPUT FILE CONFIGURATION
        fileH = self.open_file(os.path.join(f'{in_dir}', 'entropy.bin'), 'rb')
        fileA = self.open_file(os.path.join(f'{in_dir}', 'anisotropy.bin'), 'rb')
        fileAl1 = self.open_file(os.path.join(f'{in_dir}', 'alpha1.bin'), 'rb')
        fileAl2 = self.open_file(os.path.join(f'{in_dir}', 'alpha2.bin'), 'rb')
        fileBe1 = self.open_file(os.path.join(f'{in_dir}', 'beta1.bin'), 'rb')
        fileBe2 = self.open_file(os.path.join(f'{in_dir}', 'beta2.bin'), 'rb')
        filep1 = self.open_file(os.path.join(f'{in_dir}', 'p1.bin'), 'rb')
        filep2 = self.open_file(os.path.join(f'{in_dir}', 'p2.bin'), 'rb')
        file_class = self.open_file(f'{in_class_name}', 'rb')
        if flag_valid is True:
            in_valid = self.open_file(f'{file_valid}', 'rb')

        # # OUTPUT FILE OPENING
        # out_grd = self.open_output_file(file_name_out[0])
        # out_vol = self.open_output_file(file_name_out[1])

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # MHin = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # MAin = Nlig * n_col
        n_block_a += n_col
        n_block_b += 0
        # MAl1in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # MAl2in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # MBe1in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # MBe2in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # Mp1in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # Mp2in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0
        # class_in = Nlig*n_col
        n_block_a += n_col
        n_block_b += 0

        # 4*Cpt = 4*NCLASS_POL*NCLASS_POL
        n_block_a += 0
        n_block_b += 4 * NCLASS_POL * NCLASS_POL

        # class_out = sub_n_lig*sub_n_col
        n_block_a += 0
        n_block_b += sub_n_lig * sub_n_col

        # Reading Data
        n_polar_in = 0
        n_polar_out = 0
        n_block_b += n_col + 2 * n_col + n_polar_in * 2 * n_col + n_polar_out * n_win_l * (n_col + n_win_c)

        logging.info(f'{n_block_a=}')
        logging.info(f'{n_block_b=}')
        memory_alloc = self.check_free_memory()
        memory_alloc = max(memory_alloc, 1000)
        # logging.info(f'{memory_alloc=}')
        n_lig_block = numpy.zeros(lib.util.Application.FILE_PATH_LENGTH, dtype=int)
        nb_block = 0
        nb_block = self.memory_alloc(file_memerr, sub_n_lig, n_win_l, nb_block, n_lig_block, n_block_a, n_block_b, memory_alloc)
        logging.info(f'{n_lig_block=}')

        # MATRIX ALLOCATION
        self.allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, n_lig_block[0], sub_n_col, sub_n_lig, NCLASS_POL)

        # MASK VALID PIXELS (if there is no MaskFile
        self.set_valid_pixels(flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)

        # DATA PROCESSING
        logging.info('--= Started: data processing =--')
        self.rewind(file_class)
        if flag_valid is True:
            self.rewind(in_valid)

        n_class = -1
        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            lib.util_block.read_block_matrix_float(file_class, self.class_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            for lig in range(n_lig_block[nb]):
                lib.util.printf_line(lig, n_lig_block[nb])
                for col in range(sub_n_col):
                    if self.valid[lig][col] == 1.:
                        if (int)(self.class_in[lig][col]) > n_class:
                            n_class = (int)(self.class_in[lig][col])
        n_class += 1

        # ********************************************************************
        self.rewind(file_class)
        if flag_valid is True:
            self.rewind(in_valid)
        n_lig_g = 0
        # lig_g = 0

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: file_class  =--')
            lib.util_block.read_block_matrix_float(file_class, self.class_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: fileH  =--')
            lib.util_block.read_block_matrix_float(fileH, self.mh_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: fileA  =--')
            lib.util_block.read_block_matrix_float(fileA, self.ma_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: fileAl1  =--')
            lib.util_block.read_block_matrix_float(fileAl1, self.mal1_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: fileAl2  =--')
            lib.util_block.read_block_matrix_float(fileAl2, self.mal2_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: fileBe1  =--')
            lib.util_block.read_block_matrix_float(fileBe1, self.mbe1_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: fileBe2  =--')
            lib.util_block.read_block_matrix_float(fileBe2, self.mbe2_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: filep1  =--')
            lib.util_block.read_block_matrix_float(filep1, self.mp1_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            logging.info('--= Started: read_block_matrix_float: filep2  =--')
            lib.util_block.read_block_matrix_float(filep2, self.mp2_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            logging.info('--= Started: data_processing_routine  =--')
            data_processing_routine(nb, n_lig_block, n_lig_g, sub_n_col, self.class_out, self.class_in, self.valid, self.mh_in, self.ma_in, self.mal1_in, self.mal2_in, self.mbe1_in, self.mbe2_in, self.mp1_in, self.mp2_in, self.cpt_h_a, self.cpt_al1, self.cpt_al1_al2, lib.util.Application.PI)
            n_lig_g += n_lig_block[nb]

        lib.graphics.bmp_wishart(self.class_out, sub_n_lig, sub_n_col, os.path.join(f'{out_dir}', 'id_scatt'), color_map_wishart)

        # ********************************************************************
        logging.info('--= Started: my_max  =--')
        data_processing_routine_max(n_class, self.cpt_h_a, self.class_vec, self.cpt_al1, self.cpt_al1_al2, lib.util.Application.INIT_MINMAX)

        # ********************************************************************
        logging.info('--= Started: processing - id_class  =--')
        self.rewind(file_class)
        if flag_valid is True:
            self.rewind(in_valid)
        n_lig_g = 0
        # lig_g = 0

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            lib.util_block.read_block_matrix_float(file_class, self.class_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            # pragma omp parallel for private(col, ligg)
            data_processing_routine_id_class(nb, n_lig_block, n_lig_g, sub_n_col, self.class_out, self.class_in, self.valid, self.class_vec)
            n_lig_g += n_lig_block[nb]

        lib.graphics.bmp_wishart(self.class_out, sub_n_lig, sub_n_col, os.path.join(f'{out_dir}', 'id_class'), color_map_wishart)

        # ********************************************************************
        logging.info('--= Started: processing - vol_class  =--')
        self.rewind(file_class)
        if flag_valid is True:
            self.rewind(in_valid)
        n_lig_g = 0
        # lig_g = 0

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            lib.util_block.read_block_matrix_float(file_class, self.class_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            # pragma omp parallel for private(col, ligg)
            data_processing_routine_vol_class(nb, n_lig_block, n_lig_g, sub_n_col, self.class_out, self.class_in, self.valid, self.class_vec)
            n_lig_g += n_lig_block[nb]

        lib.graphics.bmp_wishart(self.class_out, sub_n_lig, sub_n_col, os.path.join(f'{out_dir}', 'vol_class'), color_map_wishart)

        # OUTPUT FILE
        with self.open_file(os.path.join(f'{out_dir}', 'vol_class.bin'), 'wb') as out_file:
            lib.util_block.write_block_matrix_float(out_file, self.class_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

        # ********************************************************************
        logging.info('--= Started: processing - sgl_class  =--')
        self.rewind(file_class)
        if flag_valid is True:
            self.rewind(in_valid)
        n_lig_g = 0
        # lig_g = 0

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            lib.util_block.read_block_matrix_float(file_class, self.class_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            # pragma omp parallel for private(col, ligg)
            data_processing_routine_sgl_class(nb, n_lig_block, n_lig_g, sub_n_col, self.class_out, self.class_in, self.valid, self.class_vec)
            n_lig_g += n_lig_block[nb]

        lib.graphics.bmp_wishart(self.class_out, sub_n_lig, sub_n_col, os.path.join(f'{out_dir}', 'sgl_class'), color_map_wishart)

        # OUTPUT FILE
        with self.open_file(os.path.join(f'{out_dir}', 'sgl_class.bin'), 'wb') as out_file:
            lib.util_block.write_block_matrix_float(out_file, self.class_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

        # ********************************************************************
        logging.info('--= Started: processing - dbl_class  =--')
        self.rewind(file_class)
        if flag_valid is True:
            self.rewind(in_valid)
        n_lig_g = 0
        # lig_g = 0

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)
            lib.util_block.read_block_matrix_float(file_class, self.class_in, nb, nb_block, n_lig_block[nb], sub_n_col, 1, 1, off_lig, off_col, n_col, self.vf_in)

            # pragma omp parallel for private(col, ligg)
            data_processing_routine_dbl_class(nb, n_lig_block, n_lig_g, sub_n_col, self.class_out, self.class_in, self.valid, self.class_vec)
            n_lig_g += n_lig_block[nb]

        lib.graphics.bmp_wishart(self.class_out, sub_n_lig, sub_n_col, os.path.join(f'{out_dir}', 'dbl_class'), color_map_wishart)

        # OUTPUT FILE
        with self.open_file(os.path.join(f'{out_dir}', 'dbl_class.bin'), 'wb') as out_file:
            lib.util_block.write_block_matrix_float(out_file, self.class_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)



def main(*args, **kwargs):
    '''Main function

    Args:
        id (str): input directory
        od (str): output directory
        ofr (int): Offset Row
        ofc (int): Offset Col
        fnr (int): Final Number of Row
        fnc (int): Final Number of Col
        icf (str): input class file
        clm (str): Colormap wishart 16 colors
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=[])
    parser_args.make_def_args()
    parser_args.add_req_arg('-icf', str, 'input class file')
    parser_args.add_req_arg('-clm', str, 'Colormap wishart 16 colors')
    parser_args.rem_req_arg('-iodf')
    parser_args.rem_req_arg('-nwr')
    parser_args.rem_req_arg('-nwc')
    parser_args.rem_req_arg('-data')
    parsed_args = parser_args.parse_args()
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
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\id_class_gen\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\id_class_gen\\py'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/id_class_gen/'
            dir_out = '/home/krzysiek/polsarpro/out/id_class_gen/py/'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()

        # Pass params as expand dictionary with '**'
        params['id'] = dir_in
        params['od'] = dir_out
        params['ofr'] = 0
        params['ofc'] = 0
        params['fnr'] = 18432
        params['fnc'] = 1248
        params['icf'] = os.path.join(f'{dir_in}', 'wishart_H_A_alpha_class_3x3.bin')
        params['clm'] = os.path.join(f'{dir_in}', 'Wishart_ColorMap16.pal')
        params['errf'] = os.path.join(f'{dir_out}', 'MemoryAllocError.txt')
        params['mask'] = os.path.join(f'{dir_in}', 'mask_valid_pixels.bin')
        main(**params)

        # Pass parasm as positional arguments
        # main(id=dir_in,
        #      od=dir_out,
        #      ofr=0,
        #      ofc=0,
        #      fnr=18432,
        #      fnc=1248,
        #      icf=os.path.join(f'{dir_in}', 'wishart_H_A_alpha_class_3x3.bin'),
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
        #      mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))
