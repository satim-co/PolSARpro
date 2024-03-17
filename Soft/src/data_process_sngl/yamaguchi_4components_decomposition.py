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

File  : yamaguchi_4components_decomposition.c
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

Description :  Yamaguchi 4 components Decomposition

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
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m_avg.fill(0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in range(sub_n_col):
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
                span = m_avg[lib.util.C311][col] + m_avg[lib.util.C322][col] + m_avg[lib.util.C333][col]
                s_max = max(s_max, span)
                s_min = min(s_min, span)
    return s_min, s_max


@numba.njit(parallel=False, fastmath=True)
def unitary_rotation(TT, teta):
    t11 = TT[lib.util.T311]
    t12_re, t12_im = TT[lib.util.T312_RE], TT[lib.util.T312_IM]
    t13_re, t13_im = TT[lib.util.T313_RE], TT[lib.util.T313_IM]
    t22 = TT[lib.util.T322]
    t23_re, t23_im = TT[lib.util.T323_RE], TT[lib.util.T323_IM]
    t33 = TT[lib.util.T333]

    TT[lib.util.T311] = t11
    TT[lib.util.T312_RE] = t12_re * math.cos(teta) + t13_re * math.sin(teta)
    TT[lib.util.T312_IM] = t12_im * math.cos(teta) + t13_im * math.sin(teta)
    TT[lib.util.T313_RE] = -t12_re * math.sin(teta) + t13_re * math.cos(teta)
    TT[lib.util.T313_IM] = -t12_im * math.sin(teta) + t13_im * math.cos(teta)
    TT[lib.util.T322] = t22 * math.cos(teta)**2 + 2. * t23_re * math.cos(teta) * math.sin(teta) + t33 * math.sin(teta) ** 2
    TT[lib.util.T323_RE] = -t22 * math.cos(teta) * math.sin(teta) + t23_re * math.cos(teta)**2 - t23_re * math.sin(teta) ** 2 + t33 * math.cos(teta) * math.sin(teta)
    TT[lib.util.T323_IM] = t23_im * math.cos(teta) ** 2 + t23_im * math.sin(teta) ** 2
    TT[lib.util.T333] = t22 * math.sin(teta)**2 + t33 * math.cos(teta)**2 - 2. * t23_re * math.cos(teta) * math.sin(teta)


@numba.njit(parallel=False, fastmath=True)
def data_processing(yam_mode, nb, n_lig_block, n_polar_out, m_in, valid, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, m_odd, m_dbl, m_vol, m_hlx, span_min, span_max, eps):
    HV_type = 0
    FS = FD = FV = ALPre = ALPim = BETre = BETim = 0.
    HHHH = HVHV = VVVV = HHVVre = HHVVim = 0.
    rtemp = ratio = S = D = Cre = Cim = CO = C1 = teta = 0.
    Ps = Pd = Pv = Pc = TP = 0.
    # pragma omp parallel for private(col, M_avg) firstprivate(CC11, CC13_re, CC13_im, CC22, CC33, FS, FD, FV, ALPre, ALPim, BETre, BETim, HHHH, HVHV, VVVV, HHVVre, HHVVim, rtemp, ratio) shared(ligDone)
    ligDone = 0
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    TT = lib.matrix.vector_float(n_polar_out)
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m_avg.fill(0)
        TT.fill(0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in numba.prange(sub_n_col):
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1:

                for np in range(n_polar_out):
                    TT[np] = m_avg[np][col]
                teta = 0.
                if (yam_mode == "Y4R") or (yam_mode == "S4R"):
                    sub_res = TT[lib.util.T322] - TT[lib.util.T333]
                    if sub_res != 0:
                        teta = 0.5 * math.atan(2 * TT[lib.util.T323_RE] / (sub_res))
                    else:
                        teta = 0.5 * math.atan(math.inf)
                    unitary_rotation(TT, teta)
                Pc = 2. * math.fabs(TT[lib.util.T323_IM])
                HV_type = 1  # Surface scattering
                if yam_mode == "S4R":
                    C1 = TT[lib.util.T311] - TT[lib.util.T322] + (7. / 8.) * TT[lib.util.T333] + (Pc / 16.)
                    if C1 > 0.:
                        HV_type = 1  # Surface scattering
                    else:
                        HV_type = 2  # Double bounce scattering
                # Surface scattering
                if HV_type == 1:
                    ratio = 10. * math.log10((TT[lib.util.T311] + TT[lib.util.T322] - 2. * TT[lib.util.T312_RE]) / (TT[lib.util.T311] + TT[lib.util.T322] + 2. * TT[lib.util.T312_RE]))
                    if (ratio > -2.) and (ratio <= 2.):
                        Pv = 2. * (2. * TT[lib.util.T333] - Pc)
                    else:
                        Pv = (15. / 8.) * (2. * TT[lib.util.T333] - Pc)
                # Double bounce scattering
                if HV_type == 2:
                    Pv = (15. / 16.) * (2. * TT[lib.util.T333] - Pc)

                TP = TT[lib.util.T311] + TT[lib.util.T322] + TT[lib.util.T333]

                if Pv < 0.:  # Freeman - Yamaguchi 3-components algorithm
                    HHHH = (TT[lib.util.T311] + 2 * TT[lib.util.T312_RE] + TT[lib.util.T322]) / 2.
                    HHVVre = (TT[lib.util.T311] - TT[lib.util.T322]) / 2.
                    HHVVim = -TT[lib.util.T312_IM]
                    HVHV = TT[lib.util.T333] / 2.
                    VVVV = (TT[lib.util.T311] - 2. * TT[lib.util.T312_RE] + TT[lib.util.T322]) / 2.

                    ratio = 10. * math.log10(VVVV / HHHH)
                    if ratio <= -2.:
                        FV = 15. * (HVHV / 4.)
                        HHHH = HHHH - 8. * (FV / 15.)
                        VVVV = VVVV - 3. * (FV / 15.)
                        HHVVre = HHVVre - 2. * (FV / 15.)
                    if ratio > 2.:
                        FV = 15. * (HVHV / 4.)
                        HHHH = HHHH - 3. * (FV / 15.)
                        VVVV = VVVV - 8. * (FV / 15.)
                        HHVVre = HHVVre - 2. * (FV / 15.)
                    if (ratio > -2.) and (ratio <= 2.):
                        FV = 8. * (HVHV / 2.)
                        HHHH = HHHH - 3. * (FV / 8.)
                        VVVV = VVVV - 3. * (FV / 8.)
                        HHVVre = HHVVre - 1. * (FV / 8.)

                    # Case 1: Volume Scatter > Total
                    if (HHHH <= eps) or (VVVV <= eps):
                        FD = 0.
                        FS = 0.
                        if (ratio > -2.) and (ratio <= 2.):
                            FV = (HHHH + 3. * (FV / 8.)) + HVHV + (VVVV + 3. * (FV / 8.))
                        if ratio <= -2.:
                            FV = (HHHH + 8. * (FV / 15.)) + HVHV + (VVVV + 3. * (FV / 15.))
                        if ratio > 2.:
                            FV = (HHHH + 3. * (FV / 15.)) + HVHV + (VVVV + 8. * (FV / 15.))
                    else:
                        # Data conditionning for non realizable ShhSvv* term
                        if (HHVVre * HHVVre + HHVVim * HHVVim) > HHHH * VVVV:
                            rtemp = HHVVre * HHVVre + HHVVim * HHVVim
                            HHVVre = HHVVre * math.sqrt((HHHH * VVVV) / rtemp)
                            HHVVim = HHVVim * math.sqrt((HHHH * VVVV) / rtemp)
                        # Odd Bounce
                        if HHVVre >= 0.:
                            ALPre = -1.
                            ALPim = 0.
                            FD = (HHHH * VVVV - HHVVre * HHVVre - HHVVim * HHVVim) / (HHHH + VVVV + 2 * HHVVre)
                            FS = VVVV - FD
                            BETre = (FD + HHVVre) / FS
                            BETim = HHVVim / FS
                        # Even Bounce
                        if HHVVre < 0.:
                            BETre = 1.
                            BETim = 0.
                            FS = (HHHH * VVVV - HHVVre * HHVVre - HHVVim * HHVVim) / (HHHH + VVVV - 2 * HHVVre)
                            FD = VVVV - FS
                            ALPre = (HHVVre - FS) / FD
                            ALPim = HHVVim / FD

                    m_odd[lig][col] = FS * (1. + BETre * BETre + BETim * BETim)
                    m_dbl[lig][col] = FD * (1. + ALPre * ALPre + ALPim * ALPim)
                    m_vol[lig][col] = FV
                    m_hlx[lig][col] = 0.

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

                else:  # Yamaguchi 4-Components algorithm
                    # Surface scattering
                    if HV_type == 1:
                        S = TT[lib.util.T311] - (Pv / 2.)
                        D = TP - Pv - Pc - S
                        Cre = TT[lib.util.T312_RE] + TT[lib.util.T313_RE]
                        Cim = TT[lib.util.T312_IM] + TT[lib.util.T313_IM]
                        if ratio <= -2.:
                            Cre = Cre - (Pv / 6.)
                        if ratio > 2.:
                            Cre = Cre + (Pv / 6.)
                        if (Pv + Pc) > TP:
                            Ps = 0.
                            Pd = 0.
                            Pv = TP - Pc
                        else:
                            CO = 2. * TT[lib.util.T311] + Pc - TP
                            if CO > 0.:
                                Ps = S + (Cre * Cre + Cim * Cim) / S
                                Pd = D - (Cre * Cre + Cim * Cim) / S
                            else:
                                Pd = D + (Cre * Cre + Cim * Cim) / D
                                Ps = S - (Cre * Cre + Cim * Cim) / D
                        if Ps < 0.:
                            if Pd < 0.:
                                Ps = 0.
                                Pd = 0.
                                Pv = TP - Pc
                            else:
                                Ps = 0.
                                Pd = TP - Pv - Pc
                        else:
                            if Pd < 0.:
                                Pd = 0.
                                Ps = TP - Pv - Pc
                    # Double bounce scattering
                    if HV_type == 2:
                        S = TT[lib.util.T311]
                        D = TP - Pv - Pc - S
                        Cre = TT[lib.util.T312_RE] + TT[lib.util.T313_RE]
                        Cim = TT[lib.util.T312_IM] + TT[lib.util.T313_IM]

                        Pd = D + (Cre * Cre + Cim * Cim) / D
                        Ps = S - (Cre * Cre + Cim * Cim) / D
                        if Ps < 0.:
                            if Pd < 0.:
                                Ps = 0.
                                Pd = 0.
                                Pv = TP - Pc
                            else:
                                Ps = 0.
                                Pd = TP - Pv - Pc
                        else:
                            if Pd < 0.:
                                Pd = 0.
                                Ps = TP - Pv - Pc
                    if Ps < span_min:
                        Ps = span_min
                    if Pd < span_min:
                        Pd = span_min
                    if Pv < span_min:
                        Pv = span_min
                    if Pc < span_min:
                        Pc = span_min

                    if Ps > span_max:
                        Ps = span_max
                    if Pd > span_max:
                        Pd = span_max
                    if Pv > span_max:
                        Pv = span_max
                    if Pc > span_max:
                        Pc = span_max

                    m_odd[lig][col] = Ps
                    m_dbl[lig][col] = Pd
                    m_vol[lig][col] = Pv
                    m_hlx[lig][col] = Pc

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
        self.m_odd = lib.matrix.matrix_float(n_lig_block, sub_n_col)
        self.m_dbl = lib.matrix.matrix_float(n_lig_block, sub_n_col)
        self.m_vol = lib.matrix.matrix_float(n_lig_block, sub_n_col)
        self.m_hlx = lib.matrix.matrix_float(n_lig_block, sub_n_col)

    def run(self):
        logging.info('******************** Welcome in yamaguchi 4components decomposition ********************')
        logging.info(self.args)
        in_dir = self.args.id
        yam_mode = self.args.mod
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
            pol_type = 'S2T3'
        logging.info(f'{pol_type=}')

        in_dir = self.check_dir(in_dir)
        logging.info(f'{in_dir=}')
        out_dir = self.check_dir(out_dir)
        logging.info(f'{out_dir=}')

        if flag_valid is True:
            self.check_file(file_valid)

        n_win_l_m1s2 = (n_win_l - 1) // 2
        logging.info(f'{n_win_l_m1s2=}')
        n_win_c_m1s2 = (n_win_c - 1) // 2
        logging.info(f'{n_win_c_m1s2=}')

        # INPUT/OUPUT CONFIGURATIONS
        n_lig, n_col, polar_case, polar_type = lib.util.read_config(in_dir)
        logging.info(f'{n_lig=}, {n_col=}, {polar_case=}, {polar_type=}')

        # POLAR TYPE CONFIGURATION
        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type)
        logging.info(f'{pol_type=}, {n_polar_in=}, {pol_type_in=}, {n_polar_out=}, {pol_type_out=}')

        # INPUT/OUTPUT FILE CONFIGURATION
        file_name_in = lib.util.init_file_name(pol_type_in, in_dir)
        logging.info(f'{file_name_in=}')

        file_name_out = [
            os.path.join(f'{out_dir}', f'Yamaguchi4_{yam_mode}_Odd.bin'),
            os.path.join(f'{out_dir}', f'Yamaguchi4_{yam_mode}_Dbl.bin'),
            os.path.join(f'{out_dir}', f'Yamaguchi4_{yam_mode}_Vol.bin'),
            os.path.join(f'{out_dir}', f'Yamaguchi4_{yam_mode}_Hlx.bin'),
        ]
        logging.info(f'{file_name_out=}')

        # INPUT FILE OPENING
        in_datafile = []
        in_valid = None
        in_datafile, in_valid, flag_valid = self.open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

        # OUTPUT FILE OPENING
        out_odd = self.open_output_file(file_name_out[0])
        out_dbl = self.open_output_file(file_name_out[1])
        out_vol = self.open_output_file(file_name_out[2])
        out_hlx = self.open_output_file(file_name_out[3])

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # Modd = Nlig*Sub_Ncol
        n_block_a += sub_n_col
        n_block_b += 0
        # Mdbl = Nlig*Sub_Ncol
        n_block_a += sub_n_col
        n_block_b += 0
        # Mvol = Nlig*Sub_Ncol
        n_block_a += sub_n_col
        n_block_b += 0
        # Mhlx = Nlig*Sub_Ncol
        n_block_a += sub_n_col
        n_block_b += 0
        # Min = NpolarOut*Nlig*Sub_Ncol
        n_block_a += n_polar_out * (n_col + n_win_c)
        n_block_b += n_polar_out * n_win_l * (n_col + n_win_c)
        # Mavg = NpolarOut
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

        vf_in_readingLines = [None] * nb_block

        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)

            logging.info(f'READING NLIG LINES {nb=} {n_col=} {n_polar_out=} {n_lig_block[nb]=} {n_win_c_m1s2=} from yamaguchi_4components_decomposition')
            for np in range(n_polar_in):
                self.rewind(in_datafile[np])
            if flag_valid is True:
                self.rewind(in_valid)
            vf_in_readingLines[nb] = [numba.typed.List([numpy.fromfile(in_datafile[Np], dtype=numpy.float32, count=n_col) for Np in range(n_polar_out)]) for lig in range(n_lig_block[nb] + n_win_c_m1s2)]

            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type == 'S2':
                lib.util_block.ks_read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:    # Case of C,T or I
                logging.info('--= Started: read_block_tci_noavg  =--')
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])
            if pol_type_out == 'C3':
                logging.info('--= Started: c3_to_t3  =--')
                lib.util_convert.c3_to_t3(self.m_in, n_lig_block[nb], sub_n_col + n_win_c, 0, 0)

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
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in, vf_in_readingLines[nb])
            if pol_type_out == 'C3':
                lib.util_convert.c3_to_t3(self.m_in, n_lig_block[nb], sub_n_col + n_win_c, 0, 0)

            data_processing(yam_mode, nb, n_lig_block, n_polar_out, self.m_in, self.valid, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, self.m_odd, self.m_dbl, self.m_vol, self.m_hlx, span_min, span_max, lib.util.Application.EPS)

            lib.util_block.write_block_matrix_float(out_odd, self.m_odd, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
            lib.util_block.write_block_matrix_float(out_dbl, self.m_dbl, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
            lib.util_block.write_block_matrix_float(out_vol, self.m_vol, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
            lib.util_block.write_block_matrix_float(out_hlx, self.m_hlx, n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)


def main(*args, **kwargs):
    '''Main function

    Args:
        id (str): input directory
        od (str): output directory
        iodf (str): input-output data forma
        mod (str): decomposition mode (Y4O, Y4R, S4R)
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
    parser_args.add_req_arg('-mod', str, 'decomposition mode (Y4O, Y4R, S4R)', {'Y40', 'Y4R', 'S4R'})
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
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\yamaguchi_3components_decomposition\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\yamaguchi_3components_decomposition\\py\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/yamaguchi_3components_decomposition/'
            dir_out = '/home/krzysiek/polsarpro/out/freeman_2components_decomposition/'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()

        # Pass params as expanded dictionary with '**'
        params['id'] = dir_in
        params['od'] = dir_out
        params['iodf'] = 'T3'
        params['mod'] = 'Y4R'
        params['nwr'] = 3
        params['nwc'] = 3
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
        #      mod='Y4R',
        #      nwr=3,
        #      nwc=3,
        #      ofr=0,
        #      ofc=0,
        #      fnr=18432,
        #      fnc=1248,
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
