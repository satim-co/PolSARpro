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

File   : lee_refined_filter.c
Project  : ESA_POLSARPRO-SATIM
Authors  : Eric POTTIER, Jacek STRZELCZYK
Translate to python: Ryszard Wozniak
Update&Fix  : Krzysztof Smaza
Version  : 2.0
Creation : 07/2015
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

Description :  J.S. LEE refined fully polarimetric speckle filter

********************************************************************
'''


import os
import sys
import platform
import numpy
import math
import logging
import numba
sys.path.append(r'../')
import lib.util  # noqa: E402
import lib.util_block  # noqa: E402
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
def make_mask(mask, n_win):
    '''
    Routine  : make_Mask
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Translate to python: Ryszard Wozniak
    Update  :
    *--------------------------------------------------------------------
    Description :  Creates a set of 8 Nwin*Nwin pixel directional mask
            (0 or 1)
    *--------------------------------------------------------------------
    '''
    Nmax = 8
    mask[:Nmax, :n_win, :n_win] = 0.0

    Nmax = 0
    mask[Nmax, :n_win, int((n_win - 1) / 2):n_win] = 1.0

    Nmax = 4
    mask[Nmax, :n_win, :int(1 + (n_win - 1) / 2)] = 1.0

    Nmax = 1
    for k in range(n_win):
        mask[Nmax, k, k:n_win] = 1.0

    Nmax = 5
    for k in range(n_win):
        mask[Nmax][k][:k + 1] = [1.] * (k + 1)

    Nmax = 2
    mask[Nmax, :int(1 + (n_win - 1) / 2), :n_win] = 1.

    Nmax = 6
    mask[Nmax][int((n_win - 1) / 2): n_win, :n_win] = 1.0

    Nmax = 3
    for k in range(n_win):
        mask[Nmax][k, :n_win - k] = 1.

    Nmax = 7
    for k in range(n_win):
        mask[Nmax][k, int(n_win - 1 - k): n_win] = 1.


@numba.njit(parallel=False, fastmath=True)
def make_coeff(sigma2, deplct, nnwin, nwin_m1_s2, sub_n_lig, sub_ncol, span, mask, nmax, coeff, eps, init_minmax):
    '''
    Routine  : make_Coeff
    Authors  : Eric POTTIER
    Creation : 08/2009
    Translate to python: Ryszard Wozniak
    Update  : Krzysztof Smaza
    *--------------------------------------------------------------------
    Description :  Creates the Filtering Coefficient
    *--------------------------------------------------------------------
    '''
    # Internal variables
    subwin = lib.matrix.matrix_float(3, 3)
    Dist = lib.matrix.vector_float(4)
    MaxDist = 0.0
    Npoints = 0.0

    # FILTERING
    divisor = 1.0 / (nnwin * nnwin)
    for lig in range(sub_n_lig):
        for col in range(sub_ncol):
            # 3*3 average SPAN Sub_window calculation for directional gradient determination
            subwin.fill(0.0)
            for k in range(3):
                row_start = k * deplct + lig
                for n in range(3):
                    col_start = n * deplct + col
                    for kk in range(nnwin):
                        for ll in range(nnwin):
                            subwin[k][n] += span[row_start + kk, col_start + ll] * divisor
            # Directional gradient computation
            Dist[0] = -subwin[0][0] + subwin[0][2] - subwin[1][0] + subwin[1][2] - subwin[2][0] + subwin[2][2]
            Dist[1] = subwin[0][1] + subwin[0][2] - subwin[1][0] + subwin[1][2] - subwin[2][0] - subwin[2][1]
            Dist[2] = subwin[0][0] + subwin[0][1] + subwin[0][2] - subwin[2][0] - subwin[2][1] - subwin[2][2]
            Dist[3] = subwin[0][0] + subwin[0][1] + subwin[1][0] - subwin[1][2] - subwin[2][1] - subwin[2][2]

            # Choice of a directional mask according to the maximum gradient
            MaxDist = -init_minmax
            for k in range(4):
                if MaxDist < abs(Dist[k]):
                    MaxDist = abs(Dist[k])
                    nmax[lig][col] = k
            if Dist[nmax[lig][col]] > 0:
                nmax[lig][col] += 4

            # Within window statistics
            m_span = 0.
            m_span2 = 0.
            Npoints = 0.

            nmax_select = nmax[lig][col]
            for k in range(-nwin_m1_s2, 1 + nwin_m1_s2):
                r = nwin_m1_s2 + k
                for n in range(-nwin_m1_s2, 1 + nwin_m1_s2):
                    c = nwin_m1_s2 + n
                    if mask[nmax_select][r][c] == 1:
                        rr = r + lig
                        cc = c + col
                        m_span += span[rr][cc]
                        m_span2 += span[rr][cc] ** 2
                        Npoints += 1.

            m_span /= Npoints
            m_span2 /= Npoints

            # SPAN variation coefficient cv_span
            v_span = m_span2 - m_span * m_span  # Var(x) = E(x^2)-E(x)^2
            cv_span = math.sqrt(abs(v_span)) / (eps + m_span)

            # Linear filter coefficient
            coeff[lig][col] = (cv_span * cv_span - sigma2) / (cv_span * cv_span * (1 + sigma2) + eps)
            if coeff[lig][col] < 0:
                coeff[lig][col] = 0


def gradient_window_cal_params(n_win, window_parameters):
    if n_win in window_parameters:
        nn_win, deplct = window_parameters[n_win]
        return nn_win, deplct
    else:
        raise ValueError('The window width Nwin must be set to 3 to 31')


def span_determination(sub_n_lig, sub_n_col, n_win, pol_type_out, span, m_in):
    pol_type_indices = {
        'C2': [0, 3], 'C2pp1': [0, 3], 'C2pp2': [0, 3], 'C2pp3': [0, 3],
        'T2': [0, 3], 'T2pp1': [0, 3], 'T2pp2': [0, 3], 'T2pp3': [0, 3],
        'C3': [0, 5, 8], 'T3': [0, 5, 8],
        'C4': [0, 7, 12, 15], 'T4': [0, 7, 12, 15]
    }
    indices = pol_type_indices.get(pol_type_out)
    if indices is not None:
        span[:sub_n_lig + n_win, :sub_n_col + n_win] = numpy.sum(m_in[indices, :sub_n_lig + n_win, :sub_n_col + n_win], axis=0)
    return span


@numba.njit(parallel=True, fastmath=True)
def lee_refined(sub_n_lig, sub_n_col, n_polar_out, m_out, n_win_m1s2, valid, n_max, mask, coeff, m_in):
    m_out.fill(0.0)
    for lig in numba.prange(sub_n_lig):
        rr = n_win_m1s2 + lig
        for col in range(sub_n_col):
            cc = n_win_m1s2 + col
            if valid[rr][cc] != 1.0:
                continue
            nmax_select = n_max[lig][col]
            for np in range(n_polar_out):
                mean = 0.0
                n_points = 0.0
                for k in range(-n_win_m1s2, 1 + n_win_m1s2):
                    r = n_win_m1s2 + k
                    for n in range(-n_win_m1s2, 1 + n_win_m1s2):
                        c = n_win_m1s2 + n
                        if mask[nmax_select][r][c] != 1:
                            continue
                        mean += m_in[np][r + lig][c + col]
                        n_points += 1.0
                mean /= n_points

                # Filtering f(x)=E(x)+k*(x-E(x))
                # a = (m_in[np][n_win_m1s2 + lig][n_win_m1s2 + col] - mean)
                # b = coeff[lig][col]
                # c = mean
                m_out[np][lig][col] = mean + coeff[lig][col] * (m_in[np][rr][cc] - mean)


class App(lib.util.Application):

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win, n_win_l, n_win_c, n_lig_block, sub_n_col):
        '''
        Allocate matrices with given dimensions
        '''
        logging.info(f'{n_col=}, {n_polar_out=}, {n_win=}, {n_win_l=}, {n_win_c=}, {n_lig_block=}, {sub_n_col=}')
        self.vc_in = lib.matrix.vector_float(2 * n_col)
        self.vf_in = lib.matrix.vector_float(n_col)
        self.mc_in = lib.matrix.matrix_float(4, 2 * n_col)
        self.mf_in = lib.matrix.matrix3d_float(n_polar_out, n_win_l, n_col + n_win_c)
        self.valid = lib.matrix.matrix_float(n_lig_block + n_win_l, sub_n_col + n_win_c)
        self.mask = lib.matrix.matrix3d_float(8, n_win, n_win)
        self.span = lib.matrix.matrix_float(n_lig_block + n_win, n_col + n_win)
        self.coeff = lib.matrix.matrix_float(n_lig_block, sub_n_col)
        self.n_max = lib.matrix.matrix_int(n_lig_block + n_win, n_col + n_win)
        self.m_in = lib.matrix.matrix3d_float(n_polar_out, n_lig_block + n_win, n_col + n_win)
        self.m_out = lib.matrix.matrix3d_float(n_polar_out, n_lig_block, sub_n_col)

    def run(self):
        logging.info('******************** Welcome in lee refined filter ********************')
        logging.info(self.args)
        in_dir = self.args.id
        out_dir = self.args.od
        pol_type = self.args.iodf
        n_win = self.args.nw
        n_look = self.args.nlk
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

        n_win_m1s2 = (n_win - 1) // 2
        n_win_l = n_win
        n_win_c = n_win
        logging.info(f'{n_win=}; {n_win_l=}; {n_win_c=} {n_win_m1s2=}')

        # INPUT/OUPUT CONFIGURATIONS
        if pol_type == 'SPP':
            pol_type = 'SPPC2'
        logging.info(f'{pol_type=}')
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
        # Mout = n_polar_out*Nlig*sub_n_col
        n_block_a += n_polar_out * sub_n_col
        n_block_b += 0
        # Min = n_polar_out*(Nlig+n_win_l)*(n_col+n_win_c)
        n_block_a += n_polar_out * (n_col + n_win)
        n_block_b += n_polar_out * n_win * (n_col + n_win)
        # Mask = 8*n_win*n_win
        n_block_b += 8 * n_win * n_win
        # span = (Nlig + n_win)*(n_col + n_win)
        n_block_a += n_col + n_win
        n_block_b += n_win * (n_col + n_win)
        # coeff = Nlig * sub_n_col
        n_block_a += sub_n_col
        # Nmax = (Nlig + n_win)*(n_col + n_win)
        n_block_a += n_col + n_win
        n_block_b += n_win * (n_col + n_win)

        # Reading Data
        n_block_b += n_col + 2 * n_col + n_polar_in * 2 * n_col + n_polar_out * n_win_l * (n_col + n_win_c)

        memory_alloc = self.check_free_memory()
        memory_alloc = max(memory_alloc, 1000)
        logging.info(f'{memory_alloc=}')
        n_lig_block = numpy.zeros(lib.util.Application.FILE_PATH_LENGTH, dtype=int)
        nb_block = 0
        nb_block = self.memory_alloc(file_memerr, sub_n_lig, n_win_l, nb_block, n_lig_block, n_block_a, n_block_b, memory_alloc)
        logging.info(f'{n_lig_block=}')

        # MATRIX ALLOCATION
        self.allocate_matrices(n_col, n_polar_out, n_win, n_win_l, n_win_c, n_lig_block[0], sub_n_col)

        # MASK VALID PIXELS (if there is no MaskFile
        self.set_valid_pixels(flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)

        # Speckle variance given by the input data number of looks
        sigma2 = 1.0 / float(n_look)

        # Gradient window calculation parameters
        window_parameters = {
            3: (1, 1),
            5: (3, 1),
            7: (3, 2),
            9: (5, 2),
            11: (5, 3),
            13: (5, 4),
            15: (7, 4),
            17: (7, 5),
            19: (7, 6),
            21: (9, 6),
            23: (9, 7),
            25: (9, 8),
            27: (11, 8),
            29: (11, 9),
            31: (11, 10)
        }
        nn_win, deplct = gradient_window_cal_params(n_win, window_parameters)

        # Create Mask
        make_mask(self.mask, n_win)

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
            if pol_type_in in ['S2', 'SPP', 'SPPpp1', 'SPPpp2', 'SPPpp3']:
                if pol_type_in == 'S2':
                    lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                else:
                    lib.util_block.read_block_spp_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # Case of C, T, or I
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            # Span Determination
            logging.info('span_determination')
            span = span_determination(sub_n_lig, sub_n_col, n_win, pol_type_out, self.span, self.m_in)

            # Filtering Coeff determination
            logging.info(f'Filtering Coeff determination {nn_win=},{n_win_m1s2=}, {n_lig_block[nb]=}, {sub_n_col=}')
            make_coeff(sigma2, deplct, nn_win, n_win_m1s2, n_lig_block[nb], sub_n_col, span, self.mask, self.n_max, self.coeff, lib.util.Application.EPS, lib.util.Application.INIT_MINMAX)

            # Filtering Element per Element
            logging.info('Filtering Element per Element')
            lee_refined(sub_n_lig, sub_n_col, n_polar_out, self.m_out, n_win_m1s2, self.valid, self.n_max, self.mask, self.coeff, self.m_in)

            logging.info('write_block_matrix3d_float')
            lib.util_block.write_block_matrix3d_float(out_datafile, n_polar_out, self.m_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)


def main(*args, **kwargs):
    '''Main function

    Args:
        id (str): input directory
        od (str): output directory
        iodf (str): input-output data forma
        nw (int): Nwin Row and Col
        nlk (int): Nlook
        ofr (int): Offset Row
        ofc (int): Offset Col
        fnr (int): Final Number of Row
        fnc (int): Final Number of Col
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''
    POL_TYPE_VALUES = ['S2C3', 'S2C4', 'S2T3', 'S2T4', 'C2', 'C3', 'C4', 'T2', 'T3', 'T4', 'SPP', 'IPP']
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=POL_TYPE_VALUES)
    parser_args.make_def_args()
    parser_args.rem_req_arg('-nwr')
    parser_args.rem_req_arg('-nwc')
    parser_args.add_req_arg('-nw', int, 'Nwin Row and Col')
    parser_args.add_req_arg('-nlk', int, 'Nlook')
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
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\lee_refined_filter\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\lee_refined_filter\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/lee_refined_filter/'
            dir_out = '/home/krzysiek/polsarpro/out/lee_refined_filter/'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()

        # Pass params as expanded dictionary with '**'
        params['id'] = dir_in
        params['od'] = dir_out
        params['iodf'] = 'T3'
        params['nw'] = 7
        params['nlk'] = 7
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
        #      nw=7,
        #      nlk=7,
        #      ofr=0,
        #      ofc=0,
        #      fnr=18432,
        #      fnc=1248,
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
        #      mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))
