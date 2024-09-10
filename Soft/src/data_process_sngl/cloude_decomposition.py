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

File  : cloude decomposition.c
Project  : ESA_POLSARPRO-SATIM
Authors  : Eric POTTIER, Jacek STRZELCZYK
Translate to python: Ryszard Wozniak
Update&Fix  : Krzysztof Smaza
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

Description :  Cloude decomposition

********************************************************************
'''


import os
import sys
import platform
import datetime
import numpy
import math
import logging
import numba
sys.path.append(r'../')
import lib.util  # noqa: E402
import lib.util_block  # noqa: E402
import lib.matrix  # noqa: E402
import lib.processing  # noqa: E402

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
def cloude_decomposition_algorithm(nb, n_lig_block, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, eps, m_out):  # , sub_n_lig, sub_n_col, n_win_lm_1s2, n_win_cm_1s2, n_win_c, m_in, valid, m_out, m, n_win_l):
    # pragma omp parallel for private(col, k, Np, M, V, lambda, M_avg) firstprivate(k1r, k1i, k2r, k2i, k3r, k3i) shared(ligDone)
    ligDone = 0
    m = lib.matrix.matrix3d_float(3, 3, 2)
    v = lib.matrix.matrix3d_float(3, 3, 2)
    lmbda = lib.matrix.vector_float(3)
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m.fill(0.0)
        v.fill(0.0)
        lmbda.fill(0.0)
        m_avg.fill(0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
        for col in range(sub_n_col):
            if valid[n_win_lm_1s2 + lig][n_win_cm_1s2 + col] == 1.:
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

                # EIGENVECTOR/EIGENVALUE DECOMPOSITION
                # V complex eigenvecor matrix, lambda real vector
                lib.processing.diagonalisation(3, m, v, lmbda)

                for k in range(3):
                    if lmbda[k] < 0.:
                        lmbda[k] = 0.

                # Cloude algorithm
                k1r = math.sqrt(lmbda[0]) * v[0][0][0]
                k1i = math.sqrt(lmbda[0]) * v[0][0][1]
                k2r = math.sqrt(lmbda[0]) * v[1][0][0]
                k2i = math.sqrt(lmbda[0]) * v[1][0][1]
                k3r = math.sqrt(lmbda[0]) * v[2][0][0]
                k3i = math.sqrt(lmbda[0]) * v[2][0][1]

                m_out[0][lig][col] = k1r * k1r + k1i * k1i
                m_out[1][lig][col] = k1r * k2r + k1i * k2i
                m_out[2][lig][col] = k1i * k2r - k1r * k2i
                m_out[3][lig][col] = k1r * k3r + k1i * k3i
                m_out[4][lig][col] = k1i * k3r - k1r * k3i
                m_out[5][lig][col] = k2r * k2r + k2i * k2i
                m_out[6][lig][col] = k2r * k3r + k2i * k3i
                m_out[7][lig][col] = k2i * k3r - k2r * k3i
                m_out[8][lig][col] = k3r * k3r + k3i * k3i
            else:
                for np in range(n_polar_out):
                    m_out[np][lig][col] = 0.


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

    def run(self):
        logging.info('******************** Welcome in cloude decomposition ********************')
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
        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type, polar_type)
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
        out_datafile = self.open_output_files(file_name_out, n_polar_out)

        # COPY HEADER
        self.copy_header(in_dir, out_dir)

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # Mout = NpolarOut*Nlig*Sub_Ncol
        n_block_a += n_polar_out * sub_n_col
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

        # DATA PROCESSING
        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)

            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type_in == 'S2':
                lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # Case of C, T, or I
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            cloude_decomposition_algorithm(nb, n_lig_block, n_polar_out, sub_n_col, self.m_in, self.valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, lib.util.Application.EPS, self.m_out)

            lib.util_block.write_block_matrix3d_float(out_datafile, n_polar_out, self.m_out, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)


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
    POL_TYPE_VALUES = ['S2C3', 'S2T3', 'C3', 'T3']
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
        # For manual test
        dir_in = None
        dir_out = None
        params = {}
        module_name = os.path.splitext(f'{os.path.basename(sys.argv[0])}')[0]
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        if platform.system().lower().startswith('win') is True:
            home = os.environ['USERPROFILE']
            dir_in = f'{home}\\polsarpro\\in\\{module_name}\\'
            dir_out = f'{home}\\polsarpro\\out\\artifacts\\{timestamp}\\{module_name}\\out\\'
        elif platform.system().lower().startswith('lin') is True:
            home = os.environ["HOME"]
            dir_in = f'{home}/polsarpro/in/{module_name}/'
            dir_out = f'{home}/polsarpro/out/artifacts/{timestamp}/{module_name}/out'
            params['v'] = None
        else:
            print(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
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
        params['mask'] = os.path.join(dir_in, 'mask_valid_pixels.bin')  # optional param
        lib.util.dump_dict(params)
        main(**params)

        # Pass params as positional arguments
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
