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

File  : OPCE.py
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
  laurent.ferro-famil@univ-rennes1.fr
*--------------------------------------------------------------------

Description :  Sampling of full polar coherency matrices from an
               image using user defined pixel coordinates, then apply
               the OPCE procedure on the Target and Clutter cluster
               centers

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
import lib.util
import lib.util_block
import lib.util_convert
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


def read_coord(file_name):
    '''
    Routine  : read_coord
    Authors  : Laurent FERRO-FAMIL
    Translate to python: Krzysztof Smaza
    Creation : 07/2003
    Update  : 2024-01-28
    *-------------------------------------------------------------------
    Description :  Read training area coordinates
    *-------------------------------------------------------------------
    Inputs arguments :

    Returned values  :
    '''

    file = None
    try:
        file = open(file_name, 'r')
    except IOError:
        print('Could not open configuration file: ', file_name)
        raise

    tmp = None
    n_class = None
    tmp = file.readline()
    n_class = (int)(file.readline())

    n_area = lib.matrix.vector_int(n_class)

    n_t_pt = [None] * n_class
    area_coord_l = [None] * n_class
    area_coord_c = [None] * n_class

    zone = 0

    for classe in range(n_class):
        tmp = file.readline()
        tmp = file.readline()
        tmp = file.readline()
        n_area[classe] = numpy.int32(file.readline())

        n_t_pt[classe] = lib.matrix.vector_int(n_area[classe])
        area_coord_l[classe] = [None] * n_area[classe]
        area_coord_c[classe] = [None] * n_area[classe]

        for area in range(n_area[classe]):
            zone += 1
            tmp = file.readline()
            tmp = file.readline()
            n_t_pt[classe][area] = numpy.int32(file.readline())
            area_coord_l[classe][area] = lib.matrix.vector_float(n_t_pt[classe][area] + 1)
            area_coord_c[classe][area] = lib.matrix.vector_float(n_t_pt[classe][area] + 1)
            t_pt = 0
            for t_pt in range(n_t_pt[classe][area]):
                tmp = file.readline()
                tmp = file.readline()
                area_coord_l[classe][area][t_pt] = numpy.float32(file.readline())
                tmp = file.readline()
                area_coord_c[classe][area][t_pt] = numpy.float32(file.readline())
            area_coord_l[classe][area][t_pt + 1] = area_coord_l[classe][area][0]
            area_coord_c[classe][area][t_pt + 1] = area_coord_c[classe][area][0]
    class_map = lib.matrix.vector_float(zone + 1)
    class_map[0] = 0
    zone = 0
    for classe in range(n_class):
        for area in range(n_area[classe]):
            zone += 1
            class_map[zone] = numpy.float32(classe + 1.)
    file.close()
    return n_class, n_area, n_t_pt, area_coord_l, area_coord_c


def create_borders(border_map, n_class, n_area, n_t_pt, area_coord_c, area_coord_l):
    '''
    Routine  : create_borders
    Authors  : Laurent FERRO-FAMIL
    Translate to python: Krzysztof Smaza
    Creation : 07/2003
    Update  : 2024-01-28
    *-------------------------------------------------------------------
    Description : Create borders
    *-------------------------------------------------------------------
    Inputs arguments :

    Returned values  :

    *******************************************************************/
    '''
    label_area = -1.0

    for classe in range(n_class):
        for area in range(n_area[classe]):
            label_area += 1
            for t_pt in range(n_t_pt[classe][area]):
                x0 = area_coord_c[classe][area][t_pt]
                y0 = area_coord_l[classe][area][t_pt]
                x1 = area_coord_c[classe][area][t_pt + 1]
                y1 = area_coord_l[classe][area][t_pt + 1]
                x = x0
                y = y0
                sig_x = (float)(x1 > x0) - (float)(x1 < x0)
                sig_y = (float)(y1 > y0) - (float)(y1 < y0)
                logging.info(f'{sig_x=}, {sig_y=}')
                border_map[(int)(y)][(int)(x)] = label_area
                if x0 == x1:
                    # Vertical segment
                    while y != y1:
                        y += sig_y
                        border_map[(int)(y)][(int)(x)] = label_area
                else:
                    if y0 == y1:
                        # Horizontal segment
                        while x != x1:
                            x += sig_x
                            border_map[(int)(y)][(int)(x)] = label_area
                    else:
                        # Non horizontal & Non vertical segment
                        A = (y1 - y0) / (x1 - x0)  # Segment slope
                        B = y0 - A * x0  # Segment offset
                        while (x != x1) or (y != y1):
                            y_sol = math.floor(A * (x + sig_x) + B)
                            if math.fabs(y_sol - y) > 1:
                                sig_y_sol = (y_sol > y) - (y_sol < y)
                                while y != y_sol:
                                    y += sig_y_sol
                                    x = math.floor((y - B) / A)
                                    border_map[(int)(y)][(int)(x)] = label_area
                            else:
                                y = y_sol
                                x += sig_x
                            border_map[(int)(y)][(int)(x)] = label_area


def create_areas(border_map, n_lig, n_col, n_class, n_area, n_t_pt, area_coord_c, area_coord_l):
    '''
    Routine  : create_areas
    Authors  : Laurent FERRO-FAMIL
    Translate to python: Krzysztof Smaza
    Creation : 07/2003
    Update  : 2024-01-28
    *-------------------------------------------------------------------
    Description : Create areas
    *-------------------------------------------------------------------
    Inputs arguments :

    Returned values  :

    ********************************************************************/

    '''
    # Avoid recursive algorithm due to problems encountered under Windows
    # int change_tot, change, classe, area, t_pt;
    # float x, y, x_min, x_max, y_min, y_max, label_area;
    # Pix *P_top, *P1, *P2;

    tmp_map = lib.matrix.matrix_float(n_lig, n_col)
    P_top = None
    label_area = -1.0

    for classe in range(n_class):
        for area in range(n_area[classe]):
            label_area += 1
            x_min = n_col
            y_min = n_lig
            x_max = -1.0
            y_max = -1.0
            # Determine a square zone containing the area under study
            for t_pt in range(n_t_pt[classe][area]):
                x = area_coord_c[classe][area][t_pt]
                y = area_coord_l[classe][area][t_pt]
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
            for x in numpy.arange(x_min, x_max + 1):
                for y in numpy.arange(y_min, y_max + 1):
                    tmp_map[(int)(y)][(int)(x)] = 0.0

            for x in numpy.arange(x_min, x_max + 1):
                tmp_map[(int)(y_min)][(int)(x)] = -(float)(border_map[(int)(y_min)][(int)(x)] != label_area)
                y = y_min
                while (y <= y_max) and (border_map[(int)(y)][(int)(x)] != label_area):
                    tmp_map[(int)(y)][(int)(x)] = -1.0
                    y += 1.0
                tmp_map[(int)(y_max)][(int)(x)] = -(float)(border_map[(int)(y_max)][(int)(x)] != label_area)
                y = y_max
                while (y >= y_min) and (border_map[(int)(y)][(int)(x)] != label_area):
                    tmp_map[(int)(y)][(int)(x)] = -1.0
                    y -= 1.0

            for y in numpy.arange(y_min, y_max + 1):
                tmp_map[(int)(y)][(int)(x_min)] = -(float)(border_map[(int)(y)][(int)(x_min)] != label_area)
                x = x_min
                while (x <= x_max) and (border_map[(int)(y)][(int)(x)] != label_area):
                    tmp_map[(int)(y)][(int)(x)] = -1.0
                    x += 1.0
                tmp_map[(int)(y)][(int)(x_max)] = -(float)(border_map[(int)(y)][(int)(x_max)] != label_area)
                x = x_max
                while (x >= x_min) and (border_map[(int)(y)][(int)(x)] != label_area):
                    tmp_map[(int)(y)][(int)(x)] = -1
                    x -= 1.0

            change = 0
            for x in numpy.arange(x_min, x_max + 1):
                for y in numpy.arange(y_min, y_max + 1):
                    change = 0
                    if tmp_map[(int)(y)][(int)(x)] == -1:
                        if (x - 1) >= x_min:
                            if (tmp_map[(int)(y)][(int)(x - 1)] != 0) or (border_map[(int)(y)][(int)(x - 1)] == label_area):
                                change += 1
                        else:
                            change += 1
                        if (x + 1) <= x_max:
                            if (tmp_map[(int)(y)][(int)(x + 1)] != 0) or (border_map[(int)(y)][(int)(x + 1)] == label_area):
                                change += 1
                        else:
                            change += 1
                        if (y - 1) >= y_min:
                            if (tmp_map[(int)(y - 1)][(int)(x)] != 0) or (border_map[(int)(y - 1)][(int)(x)] == label_area):
                                change += 1
                        else:
                            change += 1
                        if (y + 1) <= y_max:
                            if (tmp_map[(int)(y + 1)][(int)(x)] != 0) or (border_map[(int)(y + 1)][(int)(x)] == label_area):
                                change += 1
                        else:
                            change += 1
                    if (border_map[(int)(y)][(int)(x)] != label_area) and (change < 4):
                        P2 = None
                        P2 = lib.util.Create_Pix(P2, x, y)
                        if change == 0:
                            P_top = P2
                            P1 = P_top
                            change = 1
                        else:
                            P1.next = P2
                            P1 = P2
            change_tot = 1
            while change_tot == 1:
                change_tot = 0
                P1 = P_top
                while P1 is not None:
                    x = P1.x
                    y = P1.y
                    change = 0
                    if tmp_map[(int)(y)][(int)(x)] == -1:
                        if (x - 1) >= x_min:
                            if (border_map[(int)(y)][(int)(x - 1)] != label_area) and (tmp_map[(int)(y)][(int)(x - 1)] != -1):
                                tmp_map[(int)(y)][(int)(x - 1)] = -1.0
                                change = 1
                        if (x + 1) <= x_max:
                            if (border_map[(int)(y)][(int)(x + 1)] != label_area) and (tmp_map[(int)(y)][(int)(x + 1)] != -1):
                                tmp_map[(int)(y)][(int)(x + 1)] = -1.0
                                change = 1
                        if (y - 1) >= y_min:
                            if (border_map[(int)(y - 1)][(int)(x)] != label_area) and (tmp_map[(int)(y - 1)][(int)(x)] != -1):
                                tmp_map[(int)(y - 1)][(int)(x)] = -1.0
                                change = 1
                        if (y + 1) <= y_max:
                            if (border_map[(int)(y + 1)][(int)(x)] != label_area) and (tmp_map[(int)(y + 1)][(int)(x)] != -1):
                                tmp_map[(int)(y + 1)][(int)(x)] = -1.0
                                change = 1
                        if change == 1:
                            change_tot = 1
                        change = 0

                        if (x - 1) >= x_min:
                            if (tmp_map[(int)(y)][(int)(x - 1)] != 0) or (border_map[(int)(y)][(int)(x - 1)] == label_area):
                                change += 1
                        else:
                            change += 1

                        if (x + 1) <= x_max:
                            if (tmp_map[(int)(y)][(int)(x + 1)] != 0) or (border_map[(int)(y)][(int)(x + 1)] == label_area):
                                change += 1
                        else:
                            change += 1

                        if (y - 1) >= y_min:
                            if (tmp_map[(int)(y - 1)][(int)(x)] != 0) or (border_map[(int)(y - 1)][(int)(x)] == label_area):
                                change += 1
                        else:
                            change += 1

                        if (y + 1) <= y_max:
                            if (tmp_map[(int)(y + 1)][(int)(x)] != 0) or (border_map[(int)(y + 1)][(int)(x)] == label_area):
                                change += 1
                        else:
                            change += 1

                        if change == 4:
                            change_tot = 1
                            if P_top == P1:
                                P_top = lib.util.Remove_Pix(P_top, P1)
                            else:
                                P1 = lib.util.Remove_Pix(P_top, P1)
                    P1 = P1.next

            for x in numpy.arange(x_min, x_max + 1):
                for y in numpy.arange(y_min, y_max + 1):
                    if tmp_map[(int)(y)][(int)(x)] == 0:
                        border_map[(int)(y)][(int)(x)] = label_area


@numba.njit(parallel=True, fastmath=True)
def determine_the_opce_puissance(ligDone, nb, n_lig_block, sub_n_col_opce, m_in, valid, n_polar_out, lig, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, eps, g0, g1, g2, g3, h0, h1, h2, h3, m_out):
    # pragma omp parallel for private(col, M_avg) firstprivate(KT1, KT2, KT3, KT4, KT5, KT6, KT7, KT8, KT9, KT10, A0, A1, A2, A3) shared(ligDone)
    # m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col_opce)
    for lig in numba.prange(n_lig_block[nb]):
        ligDone += 1
        # logging.info(f'--= Started: Determine the OPCE Puissance 1 {lig=}=--')
        # print(lig)
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        # m_avg.fill(0.0)
        m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col_opce)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col_opce, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in range(sub_n_col_opce):
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
                # Average Kennaugh matrix determination
                KT1 = eps + 0.5 * (m_avg[lib.util.T311][col] + m_avg[lib.util.T322][col] + m_avg[lib.util.T333][col])
                KT2 = eps + m_avg[lib.util.T312_RE][col]
                KT3 = eps + m_avg[lib.util.T313_RE][col]
                KT4 = eps + m_avg[lib.util.T323_IM][col]
                KT5 = eps + 0.5 * (m_avg[lib.util.T311][col] + m_avg[lib.util.T322][col] - m_avg[lib.util.T333][col])
                KT6 = eps + m_avg[lib.util.T323_RE][col]
                KT7 = eps + m_avg[lib.util.T313_IM][col]
                KT8 = eps + 0.5 * (m_avg[lib.util.T311][col] - m_avg[lib.util.T322][col] + m_avg[lib.util.T333][col])
                KT9 = eps - m_avg[lib.util.T312_IM][col]
                KT10 = eps + 0.5 * (-m_avg[lib.util.T311][col] + m_avg[lib.util.T322][col] + m_avg[lib.util.T333][col])

                A0 = g0 * KT1 + g1 * KT2 + g2 * KT3 + g3 * KT4
                A1 = g0 * KT2 + g1 * KT5 + g2 * KT6 + g3 * KT7
                A2 = g0 * KT3 + g1 * KT6 + g2 * KT8 + g3 * KT9
                A3 = g0 * KT4 + g1 * KT7 + g2 * KT9 + g3 * KT10
                m_out[lig][col] = h0 * A0 + h1 * A1 + h2 * A2 + h3 * A3
            else:
                m_out[lig][col] = 0.


class App(lib.util.Application):

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col_opce, n_lig):
        '''
        Allocate matrices with given dimensions
        '''
        logging.info(f'{n_col=}, {n_polar_out=}, {n_win_l=}, {n_win_c=}, {n_lig_block=}, {sub_n_col_opce=}')
        self.vc_in = lib.matrix.vector_float(2 * n_col)
        self.vf_in = lib.matrix.vector_float(n_col)
        self.mc_in = lib.matrix.matrix_float(4, 2 * n_col)
        self.mf_in = lib.matrix.matrix3d_float(n_polar_out, n_win_l, n_col + n_win_c)

        self.valid = lib.matrix.matrix_float(n_lig_block + n_win_l, sub_n_col_opce + n_win_c)

        self.m_in = lib.matrix.matrix3d_float(n_polar_out, n_lig_block + n_win_l, n_col + n_win_c)
        self.m_out = lib.matrix.matrix_float(n_lig_block, sub_n_col_opce)

        self.im = lib.matrix.matrix_float(n_lig, n_col)
        self.border_map = lib.matrix.matrix_float(n_lig, n_col)

        self.s_tmp = lib.matrix.matrix3d_float(4, 4, 2 * (n_col - 1))
        self.m_tmp = lib.matrix.matrix3d_float(16, 2, (n_col - 1))

    def run(self):
        logging.info('******************** Welcome in OPCE ********************')
        logging.info(self.args)
        in_dir = self.args.id
        out_dir = self.args.od
        pol_type = self.args.iodf
        file_area = self.args.af
        n_win_l = self.args.nwr
        n_win_c = self.args.nwc
        off_lig_opce = self.args.ofr
        off_col_opce = self.args.ofc
        sub_n_lig_opce = self.args.fnr
        sub_n_col_opce = self.args.fnc
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

        n_win_l_m1s2 = (n_win_l - 1) // 2
        logging.info(f'{n_win_l_m1s2=}')
        n_win_c_m1s2 = (n_win_c - 1) // 2
        logging.info(f'{n_win_c_m1s2=}')

        # INPUT/OUPUT CONFIGURATIONS
        n_lig, n_col, polar_case, polar_type = lib.util.read_config(in_dir)
        logging.info(f'{n_lig=}, {n_col=}, {polar_case=}, {polar_type=}')

        # POLAR TYPE CONFIGURATION */
        if pol_type == 'S2':
            pol_type = 'S2T3'
        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type)
        logging.info(f'{pol_type=}, {n_polar_in=}, {pol_type_in=}, {n_polar_out=}, {pol_type_out=}')

        file_name_in = lib.util.init_file_name(pol_type_in, in_dir)

        # INPUT FILE OPENING
        in_datafile = []
        in_valid = None
        in_datafile, in_valid, flag_valid = self.open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

        # OUTPUT FILE OPENING
        file_name_out = [
            os.path.join(f'{out_dir}', 'OPCE_results.txt'),
        ]
        logging.info(f'{file_name_out=}')

        # MEMORY ALLOCATION
        sub_n_col = 0
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # im = Nlig*Ncol
        n_block_a += 0
        n_block_b += n_lig * n_col
        # border = Nlig*Ncol
        n_block_a += 0
        n_block_b += n_lig * n_col
        # Mout = Nlig*Sub_Ncol
        n_block_a += sub_n_col_opce
        n_block_b += 0
        # Min = n_polar_out*Nlig*sub_n_col
        n_block_a += n_polar_out * (n_col + n_win_c)
        n_block_b += n_polar_out * n_win_l * (n_col + n_win_c)
        # Mavg = n_polar_out
        n_block_a += 0
        n_block_b += n_polar_out * sub_n_col_opce
        # Reading Data
        n_block_b += n_col + 2 * n_col + n_polar_in * 2 * n_col + n_polar_out * n_win_l * (n_col + n_win_c)

        memory_alloc = self.check_free_memory()
        memory_alloc = max(memory_alloc, 1000)
        logging.info(f'{memory_alloc=}')
        n_lig_block = numpy.zeros(lib.util.Application.FILE_PATH_LENGTH, dtype=numpy.int32)
        nb_block = 0
        nb_block = self.memory_alloc(file_memerr, sub_n_lig_opce, n_win_l, nb_block, n_lig_block, n_block_a, n_block_b, memory_alloc)
        logging.info(f'{n_lig_block=}')

        # MATRIX ALLOCATION
        t = [[[None] * 2] * 3] * 3
        coh_area = [[[None] * 2] * 3] * 3
        self.allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, n_lig_block[0], sub_n_col_opce, n_lig)

        # MASK VALID PIXELS (if there is no MaskFile)
        self.set_valid_pixels(flag_valid, n_lig_block, sub_n_col_opce, n_win_c, n_win_l)

        eps = lib.util.Application.EPS
        init_minmax = lib.util.Application.INIT_MINMAX

        # DATA PROCESSING
        logging.info('--= Started: data processing =--')
        init_time = datetime.datetime.now()

        # Training Area coordinates reading
        n_class, n_area, n_t_pt, area_coord_l, area_coord_c = read_coord(file_area)

        for lig in range(n_lig):
            for col in range(n_col):
                self.border_map[lig][col] = -1.0

        logging.info('--= Started: create_borders =--')
        create_borders(self.border_map, n_class, n_area, n_t_pt, area_coord_c, area_coord_l)
        logging.info('--= Started: create_areas processing =--')
        create_areas(self.border_map, n_lig, n_col, n_class, n_area, n_t_pt, area_coord_c, area_coord_l)

        # Training class matrix memory allocation
        logging.info('--= Started: Training class matrix memory allocation prepare =--')
        n_zones = 0
        for cls in range(n_class):
            n_zones += n_area[cls]

        for k in range(3):
            for l in range(3):
                coh_area[k][l][0] = lib.matrix.vector_float(n_zones)
                coh_area[k][l][1] = lib.matrix.vector_float(n_zones)
        cpt_zones = lib.matrix.vector_float(n_zones)

        zone = -1
        border_error_flag = 0
        logging.info(f'--= Started: Training class matrix memory allocation {n_class}=--')
        for cls in range(n_class):
            logging.info(f'--= Started: Training class matrix memory allocation {cls=}=--')
            for area in range(n_area[cls]):
                zone += 1
                off_lig = 2 * n_lig
                off_col = 2 * n_col
                sub_n_lig = -1
                sub_n_col = -1

                logging.info(f'--= Started: Training class matrix memory allocation {n_t_pt[cls][area]=} =--')
                for t_pt in range(n_t_pt[cls][area]):
                    if area_coord_l[cls][area][t_pt] < off_lig:
                        off_lig = area_coord_l[cls][area][t_pt]
                    if area_coord_c[cls][area][t_pt] < off_col:
                        off_col = area_coord_c[cls][area][t_pt]
                    if area_coord_l[cls][area][t_pt] > sub_n_lig:
                        sub_n_lig = area_coord_l[cls][area][t_pt]
                    if area_coord_c[cls][area][t_pt] > sub_n_col:
                        sub_n_col = area_coord_c[cls][area][t_pt]

                sub_n_lig = sub_n_lig - off_lig + 1
                sub_n_col = sub_n_col - off_col + 1

                cpt_zones[zone] = 0.0

                for np in range(n_polar_in):
                    self.rewind(in_datafile[np])

                for lig in numpy.arange(off_lig):
                    if pol_type_in == 'S2':
                        for np in range(n_polar_in):
                            self.s_tmp[np][0] = numpy.fromfile(in_datafile[np], dtype=numpy.float32, count=2 * n_col)
                    else:
                        for np in range(n_polar_in):
                            self.m_tmp[np][0] = numpy.fromfile(in_datafile[np], dtype=numpy.float32, count=n_col)

                for lig in numpy.arange(sub_n_lig):
                    if pol_type_in == 'S2':
                        for np in range(n_polar_in):
                            self.s_tmp[np][0] = numpy.fromfile(in_datafile[np], dtype=numpy.float32, count=2 * n_col)
                        lib.util_convert.s2_to_t3(self.s_tmp, self.m_tmp, 1, n_col, 0, 0)
                    else:
                        for np in range(n_polar_in):
                            self.m_tmp[np][0] = numpy.fromfile(in_datafile[np], dtype=numpy.float32, count=n_col)
                        if pol_type_in == 'C3':
                            lib.util_convert.c3_to_t3(self.m_tmp, 1, n_col, 0, 0)
                        if pol_type_in == 'C4':
                            lib.util_convert.c4_to_t3(self.m_tmp, 1, n_col, 0, 0)
                        if pol_type_in == 'T4':
                            lib.util_convert.t4_to_t3(self.m_tmp, 1, n_col, 0, 0)

                    for col in numpy.arange(sub_n_col):
                        if self.border_map[(int)(lig + off_lig)][(int)(col + off_col)] == zone:
                            # Average complex coherency matrix determination
                            t[0][0][0] = eps + self.m_tmp[lib.util.T311][0][(int)(col + off_col)]
                            t[0][0][1] = 0.
                            t[0][1][0] = eps + self.m_tmp[lib.util.T312_RE][0][(int)(col + off_col)]
                            t[0][1][1] = eps + self.m_tmp[lib.util.T312_IM][0][(int)(col + off_col)]
                            t[0][2][0] = eps + self.m_tmp[lib.util.T313_RE][0][(int)(col + off_col)]
                            t[0][2][1] = eps + self.m_tmp[lib.util.T313_IM][0][(int)(col + off_col)]
                            t[1][0][0] = eps + self.m_tmp[lib.util.T312_RE][0][(int)(col + off_col)]
                            t[1][0][1] = eps - self.m_tmp[lib.util.T312_IM][0][(int)(col + off_col)]
                            t[1][1][0] = eps + self.m_tmp[lib.util.T322][0][(int)(col + off_col)]
                            t[1][1][1] = 0.
                            t[1][2][0] = eps + self.m_tmp[lib.util.T323_RE][0][(int)(col + off_col)]
                            t[1][2][1] = eps + self.m_tmp[lib.util.T323_IM][0][(int)(col + off_col)]
                            t[2][0][0] = eps + self.m_tmp[lib.util.T313_RE][0][(int)(col + off_col)]
                            t[2][0][1] = eps - self.m_tmp[lib.util.T313_IM][0][(int)(col + off_col)]
                            t[2][1][0] = eps + self.m_tmp[lib.util.T323_RE][0][(int)(col + off_col)]
                            t[2][1][1] = eps - self.m_tmp[lib.util.T323_IM][0][(int)(col + off_col)]
                            t[2][2][0] = eps + self.m_tmp[lib.util.T333][0][(int)(col + off_col)]
                            t[2][2][1] = 0.

                            # Assigning T to the corresponding training coherency matrix
                            for k in range(3):
                                for l in range(3):
                                    coh_area[k][l][0][zone] = coh_area[k][l][0][zone] + t[k][l][0]
                                    coh_area[k][l][1][zone] = coh_area[k][l][1][zone] + t[k][l][1]
                            cpt_zones[zone] = cpt_zones[zone] + 1.

                            # Check if the pixel has already been assigned to a previous class
                            # Avoid overlapped classes
                            if self.im[(int)(lig + off_lig)][(int)(col + off_col)] != 0:
                                border_error_flag = 1
                            self.im[(int)(lig + off_lig)][(int)(col + off_col)] = zone + 1

                for k in range(3):
                    for l in range(3):
                        if cpt_zones[zone] == 0.0:
                            coh_area[k][l][0][zone] = -math.nan
                            coh_area[k][l][1][zone] = -math.nan
                        else:
                            coh_area[k][l][0][zone] = coh_area[k][l][0][zone] / cpt_zones[zone]
                            coh_area[k][l][1][zone] = coh_area[k][l][1][zone] / cpt_zones[zone]

        logging.info('--= Started: OPCE_results =--')
        if border_error_flag == 0:
            file_name = os.path.join(f'{out_dir}', 'OPCE_results.txt')
            fp = None
            try:
                fp = open(file_name, 'wb')
            except IOError:
                print('Could not open output file: ', file_name)
                raise

            fp.write('Target cluster centre\n'.encode('ascii'))
            fp.write(f'T11 = {coh_area[0][0][0][0]:e}\n'.encode('ascii'))
            fp.write(f'T12 = {coh_area[0][1][0][0]:e} + j {coh_area[0][1][1][0]:e}\n'.encode('ascii'))
            fp.write(f'T13 = {coh_area[0][2][0][0]:e} + j {coh_area[0][2][1][0]:e}\n'.encode('ascii'))
            fp.write(f'T22 = {coh_area[1][1][0][0]:e}\n'.encode('ascii'))
            fp.write(f'T23 = {coh_area[1][2][0][0]:e} + j {coh_area[1][2][1][0]:e}\n'.encode('ascii'))
            fp.write(f'T33 = {coh_area[2][2][0][0]:e}\n'.encode('ascii'))
            fp.write('\n'.encode('ascii'))
            fp.write('Clutter cluster centre\n'.encode('ascii'))
            fp.write(f'T11 = {coh_area[0][0][0][1]:e}\n'.encode('ascii'))
            fp.write(f'T12 = {coh_area[0][1][0][1]:e} + j {coh_area[0][1][1][1]:e}\n'.encode('ascii'))
            fp.write(f'T13 = {coh_area[0][2][0][1]:e} + j {coh_area[0][2][1][1]:e}\n'.encode('ascii'))
            fp.write(f'T22 = {coh_area[1][1][0][1]:e}\n'.encode('ascii'))
            fp.write(f'T23 = {coh_area[1][2][0][1]:e} + j {coh_area[1][2][1][1]:e}\n'.encode('ascii'))
            fp.write(f'T33 = {coh_area[2][2][0][1]:e}\n'.encode('ascii'))
            fp.write('\n'.encode('ascii'))

            # OPCE PROCEDURE
            #  Maximise the ratio (ht.KT.g)/(ht.KC.g)

            # Target Kennaugh Matrix Elements
            zone = 0
            KT1 = 0.5 * (coh_area[0][0][0][zone] + coh_area[1][1][0][zone] + coh_area[2][2][0][zone])
            KT2 = coh_area[0][1][0][zone]
            KT3 = coh_area[0][2][0][zone]
            KT4 = coh_area[1][2][1][zone]
            KT5 = 0.5 * (coh_area[0][0][0][zone] + coh_area[1][1][0][zone] - coh_area[2][2][0][zone])
            KT6 = coh_area[1][2][0][zone]
            KT7 = coh_area[0][2][1][zone]
            KT8 = 0.5 * (coh_area[0][0][0][zone] - coh_area[1][1][0][zone] + coh_area[2][2][0][zone])
            KT9 = -coh_area[0][1][1][zone]
            KT10 = 0.5 * (-coh_area[0][0][0][zone] + coh_area[1][1][0][zone] + coh_area[2][2][0][zone])

            # Clutter Kennaugh Matrix Elements
            zone = 1
            KC1 = 0.5 * (coh_area[0][0][0][zone] + coh_area[1][1][0][zone] + coh_area[2][2][0][zone])
            KC2 = coh_area[0][1][0][zone]
            KC3 = coh_area[0][2][0][zone]
            KC4 = coh_area[1][2][1][zone]
            KC5 = 0.5 * (coh_area[0][0][0][zone] + coh_area[1][1][0][zone] - coh_area[2][2][0][zone])
            KC6 = coh_area[1][2][0][zone]
            KC7 = coh_area[0][2][1][zone]
            KC8 = 0.5 * (coh_area[0][0][0][zone] - coh_area[1][1][0][zone] + coh_area[2][2][0][zone])
            KC9 = -coh_area[0][1][1][zone];
            KC10 = 0.5 * (-coh_area[0][0][0][zone] + coh_area[1][1][0][zone] + coh_area[2][2][0][zone])

            # Transmission / Reception Stokes Vector Initialisation
            #   g0p = 1.;
            g1p = 0.
            g2p = 0.
            g3p = 1.
            g0 = 1.
            g1 = 1.
            g2 = 0.
            g3 = 0.
            #   h0p = 1.;
            h1p = 0.
            h2p = 0.
            h3p = 1.
            h0 = 1.
            h1 = 1.
            h2 = 0.
            h3 = 0.

            # Initial Contrast
            A0 = g0 * KT1 + g1 * KT2 + g2 * KT3 + g3 * KT4
            A1 = g0 * KT2 + g1 * KT5 + g2 * KT6 + g3 * KT7
            A2 = g0 * KT3 + g1 * KT6 + g2 * KT8 + g3 * KT9
            A3 = g0 * KT4 + g1 * KT7 + g2 * KT9 + g3 * KT10
            Pnum = h0 * A0 + h1 * A1 + h2 * A2 + h3 * A3
            B0 = g0 * KC1 + g1 * KC2 + g2 * KC3 + g3 * KC4
            B1 = g0 * KC2 + g1 * KC5 + g2 * KC6 + g3 * KC7
            B2 = g0 * KC3 + g1 * KC6 + g2 * KC8 + g3 * KC9
            B3 = g0 * KC4 + g1 * KC7 + g2 * KC9 + g3 * KC10
            Pden = h0 * B0 + h1 * B1 + h2 * B2 + h3 * B3

            fp.write(f'Initial Target Power = {Pnum:e}\n'.encode('ascii'))
            fp.write(f'Initial Clutter Power = {Pden:e}\n'.encode('ascii'))
            fp.write(f'Initial Contrast = {Pnum / Pden:e}\n'.encode('ascii'))
            fp.write('\n'.encode('ascii'));

            x0 = g0
            x1 = g1
            x2 = g2
            x3 = g3

            arret = 0
            iteration = 0
            epsilon = 1.E-05
            logging.info('--= Started: arret =--')
            while arret == 0:
                lib.util.printf_line(iteration, 100)
                h1p = h1
                h2p = h2
                h3p = h3
                g1p = g1
                g2p = g2
                g3p = g3

                iteration += 1

                A0 = x0 * KT1 + x1 * KT2 + x2 * KT3 + x3 * KT4
                A1 = x0 * KT2 + x1 * KT5 + x2 * KT6 + x3 * KT7
                A2 = x0 * KT3 + x1 * KT6 + x2 * KT8 + x3 * KT9
                A3 = x0 * KT4 + x1 * KT7 + x2 * KT9 + x3 * KT10
                B0 = x0 * KC1 + x1 * KC2 + x2 * KC3 + x3 * KC4
                B1 = x0 * KC2 + x1 * KC5 + x2 * KC6 + x3 * KC7
                B2 = x0 * KC3 + x1 * KC6 + x2 * KC8 + x3 * KC9
                B3 = x0 * KC4 + x1 * KC7 + x2 * KC9 + x3 * KC10
                z1 = A0 * A0 - A1 * A1 - A2 * A2 - A3 * A3
                z2 = B0 * B0 - B1 * B1 - B2 * B2 - B3 * B3
                z12 = A0 * B0 - A1 * B1 - A2 * B2 - A3 * B3
                rm = (z12 + math.sqrt(z12 * z12 - z1 * z2)) / z2
                den = math.sqrt((A1 - rm * B1) * (A1 - rm * B1) + (A2 - rm * B2) * (A2 - rm * B2) + (A3 - rm * B3) * (A3 - rm * B3))
                if math.fmod(iteration, 2) == 1:
                    h1 = (A1 - rm * B1) / den
                    h2 = (A2 - rm * B2) / den
                    h3 = (A3 - rm * B3) / den
                    x0 = h0
                    x1 = h1
                    x2 = h2
                    x3 = h3
                else:
                    g1 = (A1 - rm * B1) / den
                    g2 = (A2 - rm * B2) / den
                    g3 = (A3 - rm * B3) / den
                    x0 = g0
                    x1 = g1
                    x2 = g2
                    x3 = g3

                normh = math.fabs(h1 - h1p) + math.fabs(h2 - h2p) + math.fabs(h3 - h3p)
                normg = math.fabs(g1 - g1p) + math.fabs(g2 - g2p) + math.fabs(g3 - g3p)
                # if ((normh < epsilon)&&(normg < epsilon)) arret = 1;
                arret = 1
                if normg > epsilon:
                    arret = 0
                elif normh > epsilon:
                    arret = 0

                if iteration == 100:
                    arret = 1

            fp.write(f'Initial Contrast = {Pnum / Pden:e}\n'.encode('ascii'))
            # Save results
            fp.write('Optimal Transmit Polarization\n'.encode('ascii'))
            fp.write(f'g0 = {g0:e}\n'.encode('ascii'))
            fp.write(f'g1 = {g1:e}\n'.encode('ascii'))
            fp.write(f'g2 = {g2:e}\n'.encode('ascii'))
            fp.write(f'g3 = {g3:e}\n'.encode('ascii'))
            fp.write('\n'.encode('ascii'))
            fp.write('Optimal Receive Polarization\n'.encode('ascii'))
            fp.write(f'h0 = {h0:e}\n'.encode('ascii'))
            fp.write(f'h1 = {h1:e}\n'.encode('ascii'))
            fp.write(f'h2 = {h2:e}\n'.encode('ascii'))
            fp.write(f'h3 = {h3:e}\n'.encode('ascii'))
            fp.write('\n'.encode('ascii'))
            fp.write(f'iteration = {iteration}:d\n'.encode('ascii'))
            fp.write('\n'.encode('ascii'))

            # Final Contrast
            A0 = g0 * KT1 + g1 * KT2 + g2 * KT3 + g3 * KT4
            A1 = g0 * KT2 + g1 * KT5 + g2 * KT6 + g3 * KT7
            A2 = g0 * KT3 + g1 * KT6 + g2 * KT8 + g3 * KT9
            A3 = g0 * KT4 + g1 * KT7 + g2 * KT9 + g3 * KT10
            Pnum = h0 * A0 + h1 * A1 + h2 * A2 + h3 * A3
            B0 = g0 * KC1 + g1 * KC2 + g2 * KC3 + g3 * KC4
            B1 = g0 * KC2 + g1 * KC5 + g2 * KC6 + g3 * KC7
            B2 = g0 * KC3 + g1 * KC6 + g2 * KC8 + g3 * KC9
            B3 = g0 * KC4 + g1 * KC7 + g2 * KC9 + g3 * KC10
            Pden = h0 * B0 + h1 * B1 + h2 * B2 + h3 * B3
            fp.write(f'Final Target Power = {Pnum:e}\n'.encode('ascii'))
            fp.write(f'Final Clutter Power = {Pden:e}\n'.encode('ascii'))
            fp.write(f'Final Contrast = {Pnum / Pden:e}\n'.encode('ascii'))
            fp.close()

            logging.info('--= Started: Determine the OPCE Puissance =--')
            # Determine the OPCE Puissance

            file_name = os.path.join(f'{out_dir}', 'OPCE.bin')
            fbin = None
            try:
                fbin = open(file_name, 'wb')
            except IOError:
                print('Could not open output file: ', file_name)
                raise

            for np in range(n_polar_in):
                self.rewind(in_datafile[np])
            if flag_valid is True:
                self.rewind(in_valid)

            for nb in range(nb_block):
                logging.info(f'--= Started: Determine the OPCE Puissance {nb=}=--')
                ligDone = 0
                if nb_block > 2:
                    logging.debug("%f\r" % (100 * nb / (nb_block - 1)), end="", flush=True)

                if flag_valid is True:
                    lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col_opce, n_win_l, n_win_c, off_lig_opce, off_col_opce, n_col, self.vf_in)

                if pol_type == 'S2':
                    lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col_opce, n_win_l, n_win_c, off_lig_opce, off_col_opce, n_col, self.mc_in)
                else:  # Case of C,T or I
                    lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col_opce, n_win_l, n_win_c, off_lig_opce, off_col_opce, n_col, self.vf_in)

                if pol_type_out == 'C3':
                    lib.util_convert.c3_to_t3(self.m_in, n_lig_block[nb], sub_n_col_opce + n_win_c, 0, 0)
                elif pol_type_out == 'C4':
                    lib.util_convert.c4_to_t3(self.m_in, n_lig_block[nb], sub_n_col_opce + n_win_c, 0, 0)
                elif pol_type_out == 'T4':
                    lib.util_convert.t4_to_t3(self.m_in, n_lig_block[nb], sub_n_col_opce + n_win_c, 0, 0)

                KT1 = KT2 = KT3 = KT4 = KT5 = KT6 = KT7 = KT8 = KT9 = KT10 = A0 = A1 = A2 = A3 = 0.
                # pragma omp parallel for private(col, M_avg) firstprivate(KT1, KT2, KT3, KT4, KT5, KT6, KT7, KT8, KT9, KT10, A0, A1, A2, A3) shared(ligDone)
                logging.info(f'--= Started: Determine the OPCE Puissance {n_lig_block[nb]=}=--')
                determine_the_opce_puissance(ligDone, nb, n_lig_block, sub_n_col_opce, self.m_in, self.valid, n_polar_out, lig, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, eps, g0, g1, g2, g3, h0, h1, h2, h3, self.m_out)
                # m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col_opce)
                # for lig in range(n_lig_block[nb]):
                #     ligDone += 1
                #     logging.info(f'--= Started: Determine the OPCE Puissance 1 {lig=}=--')
                #     determine_the_opce_puissance(ligDone, nb, n_lig_block, m_avg, sub_n_col_opce, self.m_in, self.valid, n_polar_out, lig, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, eps, g0, g1, g2, g3, h0, h1, h2, h3, self.m_out)
                #     # if numba_get_thread_id() == 0:
                #     #     lib.util.printf_line(ligDone, n_lig_block[nb])
                #     # m_avg.fill(0.0)
                #     # lib.util_block.average_tci(self.m_in, self.valid, n_polar_out, m_avg, lig, sub_n_col_opce, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
                #     # logging.info(f'--= Started: Determine the OPCE Puissance 2 {lig=} {sub_n_col_opce=} =--')
                #     # for col in range(sub_n_col_opce):
                #     #     if self.valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:
                #     #         # Average Kennaugh matrix determination
                #     #         KT1 = eps + 0.5 * (m_avg[lib.util.T311][col] + m_avg[lib.util.T322][col] + m_avg[lib.util.T333][col])
                #     #         KT2 = eps + m_avg[lib.util.T312_RE][col]
                #     #         KT3 = eps + m_avg[lib.util.T313_RE][col]
                #     #         KT4 = eps + m_avg[lib.util.T323_IM][col]
                #     #         KT5 = eps + 0.5 * (m_avg[lib.util.T311][col] + m_avg[lib.util.T322][col] - m_avg[lib.util.T333][col])
                #     #         KT6 = eps + m_avg[lib.util.T323_RE][col]
                #     #         KT7 = eps + m_avg[lib.util.T313_IM][col]
                #     #         KT8 = eps + 0.5 * (m_avg[lib.util.T311][col] - m_avg[lib.util.T322][col] + m_avg[lib.util.T333][col])
                #     #         KT9 = eps - m_avg[lib.util.T312_IM][col]
                #     #         KT10 = eps + 0.5 * (-m_avg[lib.util.T311][col] + m_avg[lib.util.T322][col] + m_avg[lib.util.T333][col])

                #     #         A0 = g0 * KT1 + g1 * KT2 + g2 * KT3 + g3 * KT4
                #     #         A1 = g0 * KT2 + g1 * KT5 + g2 * KT6 + g3 * KT7
                #     #         A2 = g0 * KT3 + g1 * KT6 + g2 * KT8 + g3 * KT9
                #     #         A3 = g0 * KT4 + g1 * KT7 + g2 * KT9 + g3 * KT10
                #     #         self.m_out[lig][col] = h0 * A0 + h1 * A1 + h2 * A2 + h3 * A3
                #     #     else:
                #     #         self.m_out[lig][col] = 0.
                lib.util_block.write_block_matrix_float(fbin, self.m_out, n_lig_block[nb], sub_n_col_opce, 0, 0, sub_n_col_opce)

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
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''
    POL_TYPE_VALUES = ['S2', 'C3', 'C4', 'T3', 'T4']
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=POL_TYPE_VALUES)
    parser_args.make_def_args()
    parser_args.add_req_arg('-af', str, 'input area file')
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
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\opce\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\opce\\py\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/opce/'
            dir_out = '/home/krzysiek/polsarpro/out/opce/py'
            params['v'] = None
        else:
            logging.error(f'unknown platform: {platform.system()}')
            lib.util.exit_failure()

        # Pass params as expand dictionary with '**'
        params['id'] = dir_in
        params['od'] = dir_out
        params['iodf'] = 'T3'
        params['nwr'] = 1000
        params['nwc'] = 1000
        params['ofr'] = 0
        params['ofc'] = 0
        params['fnr'] = 18432
        params['fnc'] = 1248
        params['af'] = os.path.join(f'{dir_in}', 'OPCE_areas.txt')
        params['errf'] = os.path.join(f'{dir_out}', 'MemoryAllocError.txt')
        params['mask'] = os.path.join(f'{dir_in}', 'mask_valid_pixels.bin')
        main(**params)

        # Pass parasm as positional arguments
        main(id=dir_in,
             od=dir_out,
             iodf='T3',
             nwr=1000,
             nwc=1000,
             ofr=0, 
             ofc=0,
             fnr=18432,
             fnc=1248,
             af=os.path.join(f'{dir_in}', 'OPCE_areas.txt'),
             errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
             mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))
