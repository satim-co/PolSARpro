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

File  : freeman_decomposition.c
Project  : ESA_POLSARPRO-SATIM
Authors  : Eric POTTIER, Jacek STRZELCZYK
Version  : 2.0
Creation : 07/2015
Update  :
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

Description :  Freeman Decomposition

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
def span_determination(s_min, s_max, nb, n_lig_block, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2):
    ligDone = 0
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in range(sub_n_col):
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
                span = m_avg[lib.util.C311][col] + m_avg[lib.util.C322][col] + m_avg[lib.util.C333][col]
                if span >= s_max:
                    s_max = span
                if span <= s_min:
                    s_min = span
    return s_min, s_max


@numba.njit(parallel=False, fastmath=True)
def freeman_decomposition_algorithm(nb, n_lig_block, m_in, valid, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, eps, span_min, span_max, m_odd, m_dbl, m_vol):
    # pragma omp parallel for private(col, M_avg) firstprivate(CC11, CC13_re, CC13_im, CC22, CC33, FV, FD, FS, ALP, BET, rtemp) shared(ligDone)
    CC11 = CC13_re = CC13_im = CC22 = CC33 = FV = FD = FS = ALP = BET = rtemp = 0.
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    ligDone = 0
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m_avg.fill(0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in range(sub_n_col):
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
                CC11 = m_avg[lib.util.C311][col]
                CC13_re = m_avg[lib.util.C313_RE][col]
                CC13_im = m_avg[lib.util.C313_IM][col]
                CC22 = m_avg[lib.util.C322][col]
                CC33 = m_avg[lib.util.C333][col]

                # Freeman algorithm
                FV = 3. * CC22 / 2.
                CC11 = CC11 - FV
                CC33 = CC33 - FV
                CC13_re = CC13_re - FV / 3.

                # Case 1: Volume Scatter > Total
                if (CC11 <= eps) or (CC33 <= eps):
                    FV = 3. * (CC11 + CC22 + CC33 + 2 * FV) / 8.
                    FD = 0.
                    FS = 0.
                else:
                    # Data conditionning for non realizable ShhSvv* term
                    if (CC13_re * CC13_re + CC13_im * CC13_im) > (CC11 * CC33):
                        rtemp = CC13_re * CC13_re + CC13_im * CC13_im
                        CC13_re = CC13_re * math.sqrt(CC11 * CC33 / rtemp)
                        CC13_im = CC13_im * math.sqrt(CC11 * CC33 / rtemp)
                    # Odd Bounce
                    if CC13_re >= 0.:
                        ALP = -1.
                        FD = (CC11 * CC33 - CC13_re * CC13_re - CC13_im * CC13_im) / (CC11 + CC33 + 2 * CC13_re)
                        FS = CC33 - FD
                        BET = math.sqrt((FD + CC13_re) * (FD + CC13_re) + CC13_im * CC13_im) / FS
                    # Even Bounce
                    if CC13_re < 0.:
                        BET = 1.
                        FS = (CC11 * CC33 - CC13_re * CC13_re - CC13_im * CC13_im) / (CC11 + CC33 - 2 * CC13_re)
                        FD = CC33 - FS
                        ALP = math.sqrt((FS - CC13_re) * (FS - CC13_re) + CC13_im * CC13_im) / FD

                m_odd[lig][col] = FS * (1. + BET * BET)
                m_dbl[lig][col] = FD * (1. + ALP * ALP)
                m_vol[lig][col] = 8. * FV / 3.

                if m_odd[lig][col] < span_min:
                    m_odd[lig][col] = span_min
                if m_dbl[lig][col] < span_min:
                    m_dbl[lig][col] = span_min
                if m_vol[lig][col] < span_min:
                    m_vol[lig][col] = span_min

                if m_odd[lig][col] > span_max:
                    m_odd[lig][col] = span_max
                if m_dbl[lig][col] > span_max:
                    m_dbl[lig][col] = span_max
                if m_vol[lig][col] > span_max:
                    m_vol[lig][col] = span_max
            else:
                m_odd[lig][col] = 0.
                m_dbl[lig][col] = 0.
                m_vol[lig][col] = 0.


class App(lib.util.Application):

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col):
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
        self.m_out = lib.matrix.matrix3d_float(n_polar_out, n_lig_block, sub_n_col)
        self.m_odd = lib.matrix.matrix_float(n_lig_block, sub_n_col)
        self.m_dbl = lib.matrix.matrix_float(n_lig_block, sub_n_col)
        self.m_vol = lib.matrix.matrix_float(n_lig_block, sub_n_col)


    def run(self):
        logging.info('******************** Welcome in freeman decomposition  ********************')
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
        file_memerr = self.args.errf

        flag_valid = False
        file_valid = ''

        if self.args.mask is not None and self.args.mask:
            file_valid = self.args.mask
            flag_valid = True
        logging.info(f'{flag_valid=}, {file_valid=}')

        if pol_type == 'S2':
            pol_type = 'S2C3'
        logging.info(f'{pol_type=}')

        in_dir = self.check_dir(in_dir)
        logging.info(f'{in_dir=}')
        out_dir = self.check_dir(out_dir)
        logging.info(f'{out_dir=}')

        if flag_valid is True:
            self.check_file(file_valid)

        n_win_l_m1s2 = (n_win_l - 1) // 2
        n_win_c_m1s2 = (n_win_c - 1) // 2
        logging.info(f'{n_win_c_m1s2=}; {n_win_l_m1s2=}')

        # INPUT/OUPUT CONFIGURATIONS
        n_lig, n_col, polar_case, polar_type = lib.util.read_config(in_dir)
        logging.info(f'{n_lig=}, {n_col=}, {polar_case=}, {polar_type=}')

        # POLAR TYPE CONFIGURATION
        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type)
        logging.info(f'{pol_type=}, {n_polar_in=}, {pol_type_in=}, {n_polar_out=}, {pol_type_out=}')

        # INPUT/OUTPUT FILE CONFIGURATION
        file_name_in = lib.util.init_file_name(pol_type_in, in_dir)
        logging.info(f'{file_name_in=}')

        file_name_out = lib.util.init_file_name(pol_type_out, out_dir)
        logging.info(f'{file_name_out=}')

        # INPUT FILE OPENING
        in_datafile = []
        in_valid = None
        in_datafile, in_valid, flag_valid = self.open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

        # OUTPUT FILE OPENING
        file_name_out = [
            os.path.join(f'{out_dir}', 'Freeman_Odd.bin'),
            os.path.join(f'{out_dir}', 'Freeman_Dbl.bin'),
            os.path.join(f'{out_dir}', 'Freeman_Vol.bin'),
        ]
        out_odd = self.open_output_file(file_name_out[0])
        out_dbl = self.open_output_file(file_name_out[1])
        out_vol = self.open_output_file(file_name_out[2])


        # COPY HEADER
        self.copy_header(in_dir, out_dir)

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # Modd = Nlig*Sub_Ncol
        n_block_a += sub_n_col
        n_block_b += 0
        # Mdbl = Nlig*sub_n_col
        n_block_a += sub_n_col
        n_block_b += 0
        # Mvol = Nlig*sub_n_col
        n_block_a += sub_n_col
        n_block_b += 0
        # Min = NpolarOut*Nlig*Sub_Ncol
        n_block_a += n_polar_out * (n_col + n_win_c)
        n_block_b += n_polar_out * n_win_l * (n_col + n_win_c)
        # Mavg = n_polar_out
        n_block_a += 0
        n_block_b += n_polar_out * sub_n_col
        # Reading Data
        n_block_b += n_col + 2 * n_col + n_polar_in * 2 * n_col + n_polar_out * n_win_l * (n_col + n_win_c)
        # logging.info(f'{n_block_a=}')
        # logging.info(f'{n_block_b=}')
        memory_alloc = self.check_free_memory()
        memory_alloc = max(memory_alloc, 1000)
        logging.info(f'{memory_alloc=}')
        n_lig_block = numpy.zeros(lib.util.Application.FILE_PATH_LENGTH, dtype=int)
        nb_block = 0
        nb_block = self.memory_alloc(file_memerr, sub_n_lig, n_win_l, nb_block, n_lig_block, n_block_a, n_block_b, memory_alloc)
        logging.info(f'{n_lig_block=}')
        # MATRIX ALLOCATION
        self.allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, n_lig_block[0], sub_n_col)

        # MASK VALID PIXELS (if there is no MaskFile
        self.set_valid_pixels(flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)

        # SPANMIN / SPANMAX DETERMINATION
        for np in range(n_polar_in):
            self.rewind(in_datafile[np])
        if flag_valid is True:
            self.rewind(in_valid)

        span_min = lib.util.Application.INIT_MINMAX
        span_max = -lib.util.Application.INIT_MINMAX

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)

            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type == 'S2':
                lib.util_block.ks_read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:    # Case of C,T or I
                logging.info('--= Started: read_block_tci_noavg  =--')
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)
            if pol_type_out == 'T3':
                logging.info('--= Started: t3_to_c3  =--')
                lib.util_convert.t3_to_c3(self.m_in, n_lig_block[nb], sub_n_col + n_win_c, 0, 0)

            logging.info('--= Started: average_tci  =--')
            span_min, span_max = span_determination(span_min, span_max, nb, n_lig_block, n_polar_out, sub_n_col, self.m_in, self.valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)

        if span_min < lib.util.Application.EPS:
            span_min = lib.util.Application.EPS

        logging.info(f'{span_min=}')
        logging.info(f'{span_max=}')

        # DATA PROCESSING
        logging.info('--= Started: data processing =--')
        for np in range(n_polar_in):
            self.rewind(in_datafile[np])
        if flag_valid is True:
            self.rewind(in_valid)

        for nb in range(nb_block):
            # ligDone = 0
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type == 'S2':
                lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # Case of C,T or I
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)
            if pol_type_out == 'T3':
                lib.util_convert.t3_to_c3(self.m_in, n_lig_block[nb], sub_n_col + n_win_c, 0, 0)

            # #pragma omp parallel for private(col, M_avg) firstprivate(CC11, CC13_re, CC13_im, CC22, CC33, HHHH,HVHV,VVVV, HHVVre,HHVVim, FV, FG, RHO, x,y,z1,z2r,z2i,z3r,z3i) shared(ligDone)
            freeman_decomposition_algorithm(nb, n_lig_block, self.m_in, self.valid, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, lib.util.Application.EPS, span_min, span_max, self.m_odd, self.m_dbl, self.m_vol)
            breakpoint()
            lib.util_block.write_block_matrix_float(out_odd, self.m_odd, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
            lib.util_block.write_block_matrix_float(out_dbl, self.m_dbl, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
            lib.util_block.write_block_matrix_float(out_vol, self.m_vol, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)


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
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''
    POL_TYPE_VALUES = ['S2', 'C3', 'T3']
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=POL_TYPE_VALUES)
    parser_args.make_def_args()
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
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\freeman_decomposition\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\freeman_decomposition\\py\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/freeman_decomposition/'
            dir_out = '/home/krzysiek/polsarpro/out/freeman_decomposition/'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()

        # Pass params as expanded dictionary with '**'
        params['id'] = dir_in
        params['od'] = dir_out
        params['iodf'] = 'T3'
        params['nwr'] = 7
        params['nwc'] = 7
        params['ofr'] = 0
        params['ofc'] = 0
        params['fnr'] = 18432
        params['fnc'] = 1248
        params['errf'] = os.path.join(dir_out, 'MemoryAllocError.txt')
        params['mask'] = os.path.join(dir_in, 'mask_valid_pixels.bin')
        main(**params)

        # Pass parasm as positional arguments
        # main(id=dir_in,
        #      od=dir_out,
        #      iodf='T3',
        #      nwr=7,
        #      nwc=7,
        #      ofr=0,
        #      ofc=0,
        #      fnr=18432,
        #      fnc=1248,
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
        #      mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))



# KS """
# KS Polsarpro
# KS ===
# KS Description :  Freeman Decomposition
# KS 
# KS Input parameters:
# KS id  	input directory
# KS od  	output directory
# KS iodf	input-output data format
# KS nwr 	Nwin Row
# KS nwc 	Nwin Col
# KS ofr 	Offset Row
# KS ofc 	Offset Col
# KS fnr 	Final Number of Row
# KS fnc 	Final Number of Col
# KS 
# KS Optional Parameters:
# KS mask	mask file (valid pixels)
# KS errf	memory error file
# KS help	displays this message
# KS data	displays the help concerning Data Format parameter
# KS """
# KS 
# KS import os
# KS import sys
# KS import argparse
# KS import numpy as np
# KS from collections import namedtuple
# KS import concurrent.futures
# KS from tqdm import tqdm
# KS from multiprocessing import Pool
# KS from numba import jit
# KS import util
# KS import util_block
# KS import numba
# KS import util_convert
# KS 
# KS 
# KS def main(in_dir, out_dir, pol_type, n_win_l, n_win_c, off_lig, off_col, sub_n_lig, sub_n_col, file_memerr, file_valid):
# KS     
# KS     """
# KS     Main Function for the freeman.
# KS     Parses the input arguments, reads the input files, and processes the data using freeman filtering.
# KS     """
# KS     print("********************Welcome in freeman_decomposition********************")
# KS 
# KS     # Definitions
# KS     NPolType = ["S2", "C3", "T3"]
# KS     file_name = ''
# KS 
# KS     eps = 1.E-30
# KS 
# KS     n_lig_block = np.zeros(8192, dtype=int)
# KS     n_polar_out = 0
# KS     m_in = []
# KS     valid = 0
# KS     in_datafile = []
# KS     in_valid = 0
# KS 
# KS     # Internal variables
# KS     ii, lig, col = 0, 0, 0
# KS     ligDone = 0
# KS 
# KS     span, span_min, span_max = 0.0, 0.0, 0.0
# KS 
# KS     n_win_lm_1s2 = int((n_win_l - 1) / 2)
# KS     n_win_cm_1s2 = int((n_win_c - 1) / 2)
# KS 
# KS     # # /* INPUT/OUPUT CONFIGURATIONS */
# KS     n_lig, n_col, polar_case, polar_type = read_configuration(in_dir)  
# KS 
# KS     # # /* POLAR TYPE CONFIGURATION */
# KS     pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = configure_polar_type(pol_type)
# KS 
# KS     # # /* INPUT/OUTPUT FILE CONFIGURATION */
# KS     file_name_in = configure_input_output_files(pol_type_in, in_dir, out_dir)
# KS 
# KS     # # /* INPUT FILE OPENING*/
# KS     in_datafile, in_valid, flag_valid = open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)
# KS 
# KS     # /* OUTPUT FILE OPENING*/
# KS     out_odd, out_dbl, out_vol = open_output_files(out_dir)
# KS 
# KS     # /* COPY HEADER*/
# KS     copy_header(in_dir, out_dir)
# KS 
# KS     # /* MATRIX ALLOCATION */
# KS     util.vc_in, util.vf_in, util.mc_in, util.mf_in, valid, m_in, m_odd, m_dbl, m_vol = allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col)
# KS 
# KS     # /* MASK VALID PIXELS (if there is no MaskFile */
# KS     valid = set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l)
# KS 
# KS     # SPANMIN / SPANMAX DETERMINATION
# KS     for Np in range(n_polar_in):
# KS         in_datafile[Np].seek(0)
# KS         
# KS     if flag_valid == 1:
# KS         in_valid.seek(0)
# KS 
# KS     span = 0.0
# KS     span_min = np.inf
# KS     span_max = -np.inf
# KS 
# KS     nb_block = 1
# KS     for Nb in range(nb_block):
# KS         ligDone = 0
# KS         if nb_block > 2:
# KS             print("%f\r" % (100 * Nb / (nb_block - 1)), end="", flush = True)
# KS         if flag_valid == 1:
# KS             m_in = util_block.read_block_matrix_float(in_valid, valid, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
# KS 
# KS         if pol_type == "S2":
# KS             m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
# KS         else:
# KS             # Case of C,T or I
# KS             m_in = util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
# KS             
# KS         if pol_type_out == "T3":
# KS             m_in = util_convert.T3_to_C3(m_in, sub_n_lig, sub_n_col + n_win_c, 0, 0)
# KS         
# KS         span_min, span_max = determination(lig, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
# KS         
# KS     if span_min < eps:
# KS         span_min = eps
# KS     
# KS     # DATA PROCESSING
# KS     for Np in range(n_polar_in):
# KS         in_datafile[Np].seek(0)
# KS         if flag_valid == 1:
# KS             in_valid.seek(0)
# KS 
# KS     for Nb in range(nb_block):
# KS         ligDone = 0
# KS         if nb_block > 2:
# KS             print(f"{100. * Nb / (nb_block - 1)}", end='\r', flush=True)
# KS 
# KS         if flag_valid == 1:
# KS             m_in = util_block.read_block_matrix_float(in_valid, valid, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
# KS 
# KS         if pol_type == "S2":
# KS             m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
# KS         else:
# KS             # Case of C,T or I
# KS             m_in = util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
# KS             
# KS         if pol_type_out == "T3":
# KS             m_in = util_convert.T3_to_C3(m_in, sub_n_lig, sub_n_col + n_win_c, 0, 0)
# KS 
# KS         m_odd, m_dbl, m_vol = freeman(lig, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, span_min, span_max, m_odd, m_dbl, m_vol)
# KS 
# KS     util_block.write_block_matrix_float(out_odd, m_odd, sub_n_lig, sub_n_col, 0, 0, sub_n_col)
# KS     util_block.write_block_matrix_float(out_dbl, m_dbl, sub_n_lig, sub_n_col, 0, 0, sub_n_col)
# KS     util_block.write_block_matrix_float(out_vol, m_vol, sub_n_lig, sub_n_col, 0, 0, sub_n_col)
# KS 
# KS # %% [codecell] read_configuration
# KS def read_configuration(in_dir):
# KS     """
# KS     Read the configuration from the input directory and return the parameters.
# KS     """
# KS     n_lig, n_col, polar_case, polar_type = util.read_config(in_dir)
# KS     return n_lig, n_col, polar_case, polar_type
# KS 
# KS # %% [codecell] configure_polar_type
# KS def configure_polar_type(pol_type):
# KS     """
# KS     Configure the polar type based on the provided input-output data format and return the updated parameters.
# KS     """
# KS     polar_type_replacements = {
# KS         "SPP": "SPPC2"
# KS     }
# KS 
# KS     if pol_type in polar_type_replacements:
# KS         pol_type = polar_type_replacements[pol_type]
# KS     return util.pol_type_config(pol_type)
# KS 
# KS # %% [codecell] configure_input_output_files
# KS def configure_input_output_files(pol_type_in, in_dir, out_dir):
# KS     """
# KS     Configure the input and output files based on the provided polar types and directories.
# KS     Return the input and output file names.
# KS     """
# KS     file_name_in = util.init_file_name(pol_type_in, in_dir)
# KS     return file_name_in
# KS 
# KS # %% [codecell] open_input_files
# KS def open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid):
# KS     """
# KS     Open input files and return the file objects and a flag indicating if the 'valid' file is present.
# KS     """
# KS     flag_valid = False
# KS     for n_pol in range(n_polar_in):
# KS         try:
# KS             in_datafile.append(open(file_name_in[n_pol], "rb"))
# KS         except IOError:
# KS             print("Could not open input file : ", file_name_in[n_pol])
# KS             raise
# KS 
# KS     if file_valid:
# KS         flag_valid = True
# KS         try:
# KS             in_valid = open(file_valid, "rb")
# KS         except IOError:
# KS             print("Could not open input file: ", file_valid)
# KS             raise
# KS     return in_datafile, in_valid, flag_valid
# KS 
# KS # %% [codecell] open_output_files
# KS def open_output_files(out_dir):
# KS     """
# KS     Open output files and return the file objects.
# KS     """
# KS     try:
# KS         out_odd = open(os.path.join(out_dir, "Freeman_Odd.bin"), "wb")
# KS     except IOError:
# KS         print(f"Could not open output file : {os.path.join(out_dir, 'Freeman_Odd.bin')}")
# KS         raise
# KS 
# KS     try:
# KS         out_dbl = open(os.path.join(out_dir, "Freeman_Dbl.bin"), "wb")
# KS     except IOError:
# KS         print(f"Could not open output file : {os.path.join(out_dir, 'Freeman_Dbl.bin')}")
# KS         raise
# KS 
# KS     try:
# KS         out_vol = open(os.path.join(out_dir, "Freeman_Vol.bin"), "wb")
# KS     except IOError:
# KS         print(f"Could not open output file : {os.path.join(out_dir, 'Freeman_Vol.bin')}")
# KS         raise
# KS 
# KS     return out_odd, out_dbl, out_vol
# KS 
# KS # %% [codecell] set_valid_pixels
# KS def set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l):
# KS     """
# KS     Set the valid pixels for the freeman filter based on the provided parameters.
# KS     """
# KS     if not flag_valid:
# KS         valid[:sub_n_lig + n_win_l, :sub_n_col + n_win_c] = 1.0
# KS     return valid
# KS 
# KS # %% [codecell] allocate_matrices
# KS def allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col):
# KS     """
# KS     Allocate matrices with given dimensions
# KS     """
# KS     vc_in = np.zeros(2 * n_col, dtype=np.float32)
# KS     vf_in = np.zeros(n_col, dtype=np.float32)
# KS     mc_in = np.zeros((4, 2 * n_col), dtype=np.float32)
# KS     mf_in = np.zeros((n_polar_out, n_win_l, n_col + n_win_c), dtype=np.float32)
# KS 
# KS     valid = np.zeros((sub_n_lig + n_win_l, sub_n_col + n_win_c), dtype=np.float32)
# KS     m_in = np.zeros((n_polar_out, sub_n_lig + n_win_l, n_col + n_win_c), dtype=np.float32)
# KS 
# KS     m_odd = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)
# KS     m_dbl = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)
# KS     m_vol = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)
# KS 
# KS     return vc_in, vf_in, mc_in, mf_in, valid, m_in, m_odd, m_dbl, m_vol
# KS 
# KS # %% [codecell] freeman
# KS # @numba.njit(parallel=False)
# KS def freeman(lig, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, span_min, span_max, m_odd, m_dbl, m_vol):
# KS     """
# KS     Perform freeman filtering on the input data, updating the output matrix with the filtered values.
# KS     """
# KS     CC11 = CC13_re = CC13_im = CC22 = CC33 = FV = FD = FS = ALP = BET = rtemp = 0.0
# KS     m_avg = np.zeros((n_polar_out,sub_n_col))
# KS     util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
# KS     for col in range(sub_n_col):
# KS         if valid[n_win_lm_1s2+lig][n_win_cm_1s2+col] == 1.:
# KS             eps = 1.E-30
# KS             CC11 = m_avg[util.C311][col]
# KS             CC13_re = m_avg[util.C313_RE][col]
# KS             CC13_im = m_avg[util.C313_IM][col]
# KS             CC22 = m_avg[util.C322][col]
# KS             CC33 = m_avg[util.C333][col]
# KS 
# KS             FV = 3. * CC22 / 2.
# KS             CC11 = CC11 - FV
# KS             CC33 = CC33 - FV
# KS             CC13_re = CC13_re - FV / 3.
# KS 
# KS             if (CC11 <= eps) or (CC33 <= eps):
# KS                 FV = 3. * (CC11 + CC22 + CC33 + 2 * FV) / 8.
# KS                 FD = 0.
# KS                 FS = 0.
# KS             else:
# KS                 if (CC13_re * CC13_re + CC13_im * CC13_im) > CC11 * CC33:
# KS                     rtemp = CC13_re * CC13_re + CC13_im * CC13_im
# KS                     CC13_re = CC13_re * np.sqrt(CC11 * CC33 / rtemp)
# KS                     CC13_im = CC13_im * np.sqrt(CC11 * CC33 / rtemp)
# KS                 
# KS                 if CC13_re >= 0.:
# KS                     ALP = -1.
# KS                     FD = (CC11 * CC33 - CC13_re * CC13_re - CC13_im * CC13_im) / (CC11 + CC33 + 2 * CC13_re)
# KS                     FS = CC33 - FD
# KS                     BET = np.sqrt((FD + CC13_re) * (FD + CC13_re) + CC13_im * CC13_im) / FS
# KS                 else:
# KS                     BET = 1.
# KS                     FS = (CC11 * CC33 - CC13_re * CC13_re - CC13_im * CC13_im) / (CC11 + CC33 - 2 * CC13_re)
# KS                     FD = CC33 - FS
# KS                     ALP = np.sqrt((FS - CC13_re) * (FS - CC13_re) + CC13_im * CC13_im) / FD
# KS 
# KS             m_odd[lig][col] = FS * (1. + BET * BET)
# KS             m_dbl[lig][col] = FD * (1. + ALP * ALP)
# KS             m_vol[lig][col] = 8. * FV / 3.
# KS 
# KS             if m_odd[lig][col] < span_min:
# KS                 m_odd[lig][col] = span_min
# KS             if m_dbl[lig][col] < span_min:
# KS                 m_dbl[lig][col] = span_min
# KS             if m_vol[lig][col] < span_min:
# KS                 m_vol[lig][col] = span_min
# KS 
# KS             if m_odd[lig][col] > span_max:
# KS                 m_odd[lig][col] = span_max
# KS             if m_dbl[lig][col] > span_max:
# KS                 m_dbl[lig][col] = span_max
# KS             if m_vol[lig][col] > span_max:
# KS                 m_vol[lig][col] = span_max
# KS         else:
# KS             m_odd[lig][col] = 0.
# KS             m_dbl[lig][col] = 0.
# KS             m_vol[lig][col] = 0.
# KS 
# KS     return m_odd, m_dbl, m_vol
# KS 
# KS # %% [codecell] is_pol_type_valid
# KS def is_pol_type_valid(pol_type):
# KS     """
# KS     Check if the given pol_type is valid for processing.
# KS     Returns True if the pol_type is valid, False otherwise.
# KS     """
# KS     valid_pol_types = ["S2", "SPP", "SPPpp1", "SPPpp2", "SPPpp3"]
# KS     return pol_type in valid_pol_types
# KS 
# KS def copy_header(src_dir, dst_dir):
# KS     src_path = os.path.join(src_dir, 'config.txt')
# KS     dst_path = os.path.join(dst_dir, 'config.txt')
# KS 
# KS     if os.path.isfile(src_path):
# KS         with open(src_path, 'r') as src_file:
# KS             content = src_file.read()
# KS         
# KS         with open(dst_path, 'w') as dst_file:
# KS             dst_file.write(content)
# KS     else:
# KS         print(f"Source file {src_path} does not exist.")
# KS 
# KS # @numba.njit(parallel=False)
# KS def determination(lig, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2):
# KS     M_avg = np.zeros((n_polar_out,sub_n_col), dtype=float)
# KS 
# KS     M_avg = util_block.average_tci(m_in, valid, n_polar_out, M_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
# KS     SpanMax = -np.inf
# KS     SpanMin = np.inf
# KS     for col in range(sub_n_col):
# KS         if valid[n_win_lm_1s2+lig][n_win_cm_1s2+col] == 1.:
# KS             Span = M_avg[util.C311][col]+M_avg[util.C322][col]+M_avg[util.C333][col]
# KS             if Span >= SpanMax: 
# KS                 SpanMax = Span
# KS             if Span <= SpanMin: 
# KS                 SpanMin = Span
# KS     return SpanMin, SpanMax
# KS 
# KS if __name__ == "__main__":
# KS     main("D:\\Satim\\PolSARPro\\Datasets\\T3\\", "D:\\Satim\\PolSARPro\\Datasets\\output\\", "T3", 7, 7, 0, 0, 18432, 1248, "", "")
