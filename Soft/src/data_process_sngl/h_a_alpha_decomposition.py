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

File  : h_a_alpha_decomposition.c
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

Description :  Cloude-Pottier eigenvector/eigenvalue based
               decomposition

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

SpecIndexes = [('alpha', numba.types.int32),
               ('beta', numba.types.int32),
               ('gamma', numba.types.int32),
               ('delta', numba.types.int32),
               ('lmda', numba.types.int32),
               ('epsi', numba.types.int32),
               ('a', numba.types.int32),
               ('h', numba.types.int32),
               ('nhu', numba.types.int32),
               ('combHA', numba.types.int32),
               ('combH1mA', numba.types.int32),
               ('comb1mHA', numba.types.int32),
               ('comb1mH1mA', numba.types.int32)]


@numba.experimental.jitclass(SpecIndexes)
class Indexes:
    def __init__(self):
        self.alpha = -1
        self.beta = -1
        self.gamma = -1
        self.delta = -1
        self.lmda = -1
        self.epsi = -1
        self.a = -1
        self.h = -1
        self.nhu = -1
        self.combHA = -1
        self.combH1mA = -1
        self.comb1mHA = -1
        self.comb1mH1mA = -1


SpecMatrices = [('alpha', numba.types.float32[:]),
                ('beta', numba.types.float32[:]),
                ('epsilon', numba.types.float32[:]),
                ('delta', numba.types.float32[:]),
                ('gamma', numba.types.float32[:]),
                ('nhu', numba.types.float32[:]),
                ('phase', numba.types.float32[:]),
                ('p', numba.types.float32[:])]


@numba.experimental.jitclass(SpecMatrices)
class Matrices:
    def __init__(self):
        self.alpha = lib.matrix.vector_float(4)
        self.beta = lib.matrix.vector_float(4)
        self.epsilon = lib.matrix.vector_float(4)
        self.delta = lib.matrix.vector_float(4)
        self.gamma = lib.matrix.vector_float(4)
        self.nhu = lib.matrix.vector_float(4)
        self.phase = lib.matrix.vector_float(4)
        self.p = lib.matrix.vector_float(4)


@numba.njit(parallel=False, fastmath=True)
def process_c2(nb, n_lig_block, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, n_out, m_out, eps, flag, matrices, indexes, pi):
    # pragma omp parallel for private(col, k, M, V, lambda, M_avg) firstprivate(alpha, delta, phase, p) shared(ligDone)
    ligDone = 0
    m = lib.matrix.matrix3d_float(2, 2, 2)
    v = lib.matrix.matrix3d_float(2, 2, 2)
    m_lambda = lib.matrix.vector_float(2)
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    log_2 = math.log(2.)
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m.fill(0.0)
        v.fill(0.0)
        m_lambda.fill(0.0)
        m_avg.fill(0.0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in range(sub_n_col):
            for k in range(n_out):
                m_out[k][lig][col] = 0.
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:

                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[1][0][0] = m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[3][col]
                m[1][1][1] = 0.

                # EIGENVECTOR/EIGENVALUE DECOMPOSITION
                # V complex eigenvecor matrix, lambda real vector
                lib.processing.diagonalisation(2, m, v, m_lambda)

                for k in range(2):
                    if m_lambda[k] < 0.:
                        m_lambda[k] = 0.
                for k in range(2):
                    matrices.alpha[k] = math.acos(math.sqrt(v[0][k][0] * v[0][k][0] + v[0][k][1] * v[0][k][1]))
                    matrices.phase[k] = math.atan2(v[0][k][1], eps + v[0][k][0])
                    matrices.delta[k] = math.atan2(v[1][k][1], eps + v[1][k][0]) - matrices.phase[k]
                    matrices.delta[k] = math.atan2(math.sin(matrices.delta[k]), math.cos(matrices.delta[k]) + eps)
                    # Scattering mechanism probability of occurence
                    matrices.p[k] = m_lambda[k] / (eps + m_lambda[0] + m_lambda[1])
                    if matrices.p[k] < 0.:
                        matrices.p[k] = 0.
                    if matrices.p[k] > 1.:
                        matrices.p[k] = 1.

                # Mean scattering mechanism
                if flag[indexes.alpha] != -1:
                    m_out[flag[indexes.alpha]][lig][col] = 0
                if flag[indexes.delta] != -1:
                    m_out[flag[indexes.delta]][lig][col] = 0
                if flag[indexes.lmda] != -1:
                    m_out[flag[indexes.lmda]][lig][col] = 0
                if flag[indexes.h] != -1:
                    m_out[flag[indexes.h]][lig][col] = 0

                for k in range(2):
                    if flag[indexes.alpha] != -1:
                        m_out[flag[indexes.alpha]][lig][col] += matrices.alpha[k] * matrices.p[k]
                    if flag[indexes.delta] != -1:
                        m_out[flag[indexes.delta]][lig][col] += matrices.delta[k] * matrices.p[k]
                    if flag[indexes.lmda] != -1:
                        m_out[flag[indexes.lmda]][lig][col] += m_lambda[k] * matrices.p[k]
                    if flag[indexes.h] != -1:
                        m_out[flag[indexes.h]][lig][col] -= matrices.p[k] * math.log(matrices.p[k] + eps)

                # Scaling
                if flag[indexes.alpha] != -1:
                    m_out[flag[indexes.alpha]][lig][col] *= 180. / pi
                if flag[indexes.delta] != -1:
                    m_out[flag[indexes.delta]][lig][col] *= 180. / pi
                if flag[indexes.h] != -1:
                    m_out[flag[indexes.h]][lig][col] /= log_2

                if flag[indexes.a] != -1:
                    m_out[flag[indexes.a]][lig][col] = (matrices.p[0] - matrices.p[1]) / (matrices.p[0] + matrices.p[1] + eps)

                if flag[indexes.combHA] != -1:
                    m_out[flag[indexes.combHA]][lig][col] = m_out[flag[indexes.h]][lig][col] * m_out[flag[indexes.a]][lig][col]
                if flag[indexes.combH1mA] != -1:
                    m_out[flag[indexes.combH1mA]][lig][col] = m_out[flag[indexes.h]][lig][col] * (1. - m_out[flag[indexes.a]][lig][col])
                if flag[indexes.comb1mHA] != -1:
                    m_out[flag[indexes.comb1mHA]][lig][col] = (1. - m_out[flag[indexes.h]][lig][col]) * m_out[flag[indexes.a]][lig][col]
                if flag[indexes.comb1mH1mA] != -1:
                    m_out[flag[indexes.comb1mH1mA]][lig][col] = (1. - m_out[flag[indexes.h]][lig][col]) * (1. - m_out[flag[indexes.a]][lig][col])


@numba.njit(parallel=False, fastmath=True)
def process_t3_c3(nb, n_lig_block, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, n_out, m_out, eps, flag, matrices, indexes, pi):
    # pragma omp parallel for private(col, k, M, V, lambda, M_avg) firstprivate(alpha, beta, delta, gamma, phase, p) shared(ligDone)
    ligDone = 0
    m = lib.matrix.matrix3d_float(3, 3, 2)
    v = lib.matrix.matrix3d_float(3, 3, 2)
    m_lambda = lib.matrix.vector_float(3)
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    log_3 = math.log(3.)
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m.fill(0.0)
        v.fill(0.0)
        m_lambda.fill(0.0)
        m_avg.fill(0.0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)
        for col in range(sub_n_col):
            for k in range(n_out):
                m_out[k][lig][col] = 0.
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:

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
                lib.processing.diagonalisation(3, m, v, m_lambda)

                for k in range(3):
                    if m_lambda[k] < 0.:
                        m_lambda[k] = 0.
                for k in range(3):
                    matrices.alpha[k] = math.acos(math.sqrt(v[0][k][0] * v[0][k][0] + v[0][k][1] * v[0][k][1]))
                    matrices.phase[k] = math.atan2(v[0][k][1], eps + v[0][k][0])
                    matrices.beta[k] = math.atan2(math.sqrt(v[2][k][0] * v[2][k][0] + v[2][k][1] * v[2][k][1]), eps + math.sqrt(v[1][k][0] * v[1][k][0] + v[1][k][1] * v[1][k][1]))
                    matrices.delta[k] = math.atan2(v[1][k][1], eps + v[1][k][0]) - matrices.phase[k]
                    matrices.delta[k] = math.atan2(math.sin(matrices.delta[k]), math.cos(matrices.delta[k]) + eps)
                    matrices.gamma[k] = math.atan2(v[2][k][1], eps + v[2][k][0]) - matrices.phase[k]
                    matrices.gamma[k] = math.atan2(math.sin(matrices.gamma[k]), math.cos(matrices.gamma[k]) + eps)
                    # Scattering mechanism probability of occurence
                    matrices.p[k] = m_lambda[k] / (eps + m_lambda[0] + m_lambda[1] + m_lambda[2])
                    if matrices.p[k] < 0.:
                        matrices.p[k] = 0.
                    if matrices.p[k] > 1.:
                        matrices.p[k] = 1.

                # Mean scattering mechanism
                if flag[indexes.alpha] != -1:
                    m_out[flag[indexes.alpha]][lig][col] = 0
                if flag[indexes.beta] != -1:
                    m_out[flag[indexes.beta]][lig][col] = 0
                if flag[indexes.delta] != -1:
                    m_out[flag[indexes.delta]][lig][col] = 0
                if flag[indexes.gamma] != -1:
                    m_out[flag[indexes.gamma]][lig][col] = 0
                if flag[indexes.lmda] != -1:
                    m_out[flag[indexes.lmda]][lig][col] = 0
                if flag[indexes.h] != -1:
                    m_out[flag[indexes.h]][lig][col] = 0

                for k in range(3):
                    if flag[indexes.alpha] != -1:
                        m_out[flag[indexes.alpha]][lig][col] += matrices.alpha[k] * matrices.p[k]
                    if flag[indexes.beta] != -1:
                        m_out[flag[indexes.beta]][lig][col] += matrices.beta[k] * matrices.p[k]
                    if flag[indexes.delta] != -1:
                        m_out[flag[indexes.delta]][lig][col] += matrices.delta[k] * matrices.p[k]
                    if flag[indexes.gamma] != -1:
                        m_out[flag[indexes.gamma]][lig][col] += matrices.gamma[k] * matrices.p[k]
                    if flag[indexes.lmda] != -1:
                        m_out[flag[indexes.lmda]][lig][col] += m_lambda[k] * matrices.p[k]
                    if flag[indexes.h] != -1:
                        m_out[flag[indexes.h]][lig][col] -= matrices.p[k] * math.log(matrices.p[k] + eps)

                # Scaling
                if flag[indexes.alpha] != -1:
                    m_out[flag[indexes.alpha]][lig][col] *= 180. / pi
                if flag[indexes.beta] != -1:
                    m_out[flag[indexes.beta]][lig][col] *= 180. / pi
                if flag[indexes.delta] != -1:
                    m_out[flag[indexes.delta]][lig][col] *= 180. / pi
                if flag[indexes.gamma] != -1:
                    m_out[flag[indexes.gamma]][lig][col] *= 180. / pi
                if flag[indexes.h] != -1:
                    m_out[flag[indexes.h]][lig][col] /= log_3

                if flag[indexes.a] != -1:
                    m_out[flag[indexes.a]][lig][col] = (matrices.p[1] - matrices.p[2]) / (matrices.p[1] + matrices.p[2] + eps)

                if flag[indexes.combHA] != -1:
                    m_out[flag[indexes.combHA]][lig][col] = m_out[flag[indexes.h]][lig][col] * m_out[flag[indexes.a]][lig][col]
                if flag[indexes.combH1mA] != -1:
                    m_out[flag[indexes.combH1mA]][lig][col] = m_out[flag[indexes.h]][lig][col] * (1. - m_out[flag[indexes.a]][lig][col])
                if flag[indexes.comb1mHA] != -1:
                    m_out[flag[indexes.comb1mHA]][lig][col] = (1. - m_out[flag[indexes.h]][lig][col]) * m_out[flag[indexes.a]][lig][col]
                if flag[indexes.comb1mH1mA] != -1:
                    m_out[flag[indexes.comb1mH1mA]][lig][col] = (1. - m_out[flag[indexes.h]][lig][col]) * (1. - m_out[flag[indexes.a]][lig][col])


@numba.njit(parallel=False, fastmath=True)
def process_t4_c4(nb, n_lig_block, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, n_out, m_out, eps, flag, matrices, indexes, pi):
    # pragma omp parallel for private(col, k, M, V, lambda, M_avg) firstprivate(alpha, beta, delta, gamma, epsilon, phase,nhu, p) shared(ligDone)
    ligDone = 0
    m = lib.matrix.matrix3d_float(4, 4, 2)
    v = lib.matrix.matrix3d_float(4, 4, 2)
    m_lambda = lib.matrix.vector_float(4)
    m_avg = lib.matrix.matrix_float(n_polar_out, sub_n_col)
    log_4 = math.log(4.)
    for lig in range(n_lig_block[nb]):
        ligDone += 1
        if numba_get_thread_id() == 0:
            lib.util.printf_line(ligDone, n_lig_block[nb])
        m.fill(0.0)
        v.fill(0.0)
        m_lambda.fill(0.0)
        m_avg.fill(0.0)
        lib.util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2)

        for col in range(sub_n_col):
            for k in range(n_out):
                m_out[k][lig][col] = 0.
            if valid[n_win_l_m1s2 + lig][n_win_c_m1s2 + col] == 1.:

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

                # EIGENVECTOR/EIGENVALUE DECOMPOSITION
                # V complex eigenvecor matrix, lambda real vector
                lib.processing.diagonalisation(4, m, v, m_lambda)

                for k in range(4):
                    if m_lambda[k] < 0.:
                        m_lambda[k] = 0.
                for k in range(4):
                    matrices.alpha[k] = math.acos(math.sqrt(v[0][k][0] * v[0][k][0] + v[0][k][1] * v[0][k][1]))
                    matrices.phase[k] = math.atan2(v[0][k][1], eps + v[0][k][0])
                    matrices.beta[k] = math.atan2(math.sqrt(v[2][k][0] * v[2][k][0] + v[2][k][1] * v[2][k][1] + v[3][k][0] * v[3][k][0] + v[3][k][1] * v[3][k][1]), eps + math.sqrt(v[1][k][0] * v[1][k][0] + v[1][k][1] * v[1][k][1]))
                    matrices.epsilon[k] = math.atan2(math.sqrt(v[3][k][0] * v[3][k][0] + v[3][k][1] * v[3][k][1]), eps + math.sqrt(v[2][k][0] * v[2][k][0] + v[2][k][1] * v[2][k][1]))
                    matrices.delta[k] = math.atan2(v[1][k][1], eps + v[1][k][0]) - matrices.phase[k]
                    matrices.delta[k] = math.atan2(math.sin(matrices.delta[k]), math.cos(matrices.delta[k]) + eps)
                    matrices.gamma[k] = math.atan2(v[2][k][1], eps + v[2][k][0]) - matrices.phase[k]
                    matrices.gamma[k] = math.atan2(math.sin(matrices.gamma[k]), math.cos(matrices.gamma[k]) + eps)
                    matrices.nhu[k] = math.atan2(v[3][k][1], eps + v[3][k][0]) - matrices.phase[k]
                    matrices.nhu[k] = math.atan2(math.sin(matrices.nhu[k]), math.cos(matrices.nhu[k]) + eps)
                    # Scattering mechanism probability of occurence
                    matrices.p[k] = m_lambda[k] / (eps + m_lambda[0] + m_lambda[1] + m_lambda[2] + m_lambda[3])
                    if matrices.p[k] < 0.:
                        matrices.p[k] = 0.
                    if matrices.p[k] > 1.:
                        matrices.p[k] = 1.

                # Mean scattering mechanism
                if flag[indexes.alpha] != -1:
                    m_out[flag[indexes.alpha]][lig][col] = 0
                if flag[indexes.beta] != -1:
                    m_out[flag[indexes.beta]][lig][col] = 0
                if flag[indexes.epsi] != -1:
                    m_out[flag[indexes.epsi]][lig][col] = 0
                if flag[indexes.delta] != -1:
                    m_out[flag[indexes.delta]][lig][col] = 0
                if flag[indexes.gamma] != -1:
                    m_out[flag[indexes.gamma]][lig][col] = 0
                if flag[indexes.nhu] != -1:
                    m_out[flag[indexes.nhu]][lig][col] = 0
                if flag[indexes.lmda] != -1:
                    m_out[flag[indexes.lmda]][lig][col] = 0
                if flag[indexes.h] != -1:
                    m_out[flag[indexes.h]][lig][col] = 0

                for k in range(4):
                    if flag[indexes.alpha] != -1:
                        m_out[flag[indexes.alpha]][lig][col] += matrices.alpha[k] * matrices.p[k]
                    if flag[indexes.beta] != -1:
                        m_out[flag[indexes.beta]][lig][col] += matrices.beta[k] * matrices.p[k]
                    if flag[indexes.epsi] != -1:
                        m_out[flag[indexes.epsi]][lig][col] += matrices.epsilon[k] * matrices.p[k]
                    if flag[indexes.delta] != -1:
                        m_out[flag[indexes.delta]][lig][col] += matrices.delta[k] * matrices.p[k]
                    if flag[indexes.gamma] != -1:
                        m_out[flag[indexes.gamma]][lig][col] += matrices.gamma[k] * matrices.p[k]
                    if flag[indexes.nhu] != -1:
                        m_out[flag[indexes.nhu]][lig][col] += matrices.nhu[k] * matrices.p[k]
                    if flag[indexes.lmda] != -1:
                        m_out[flag[indexes.lmda]][lig][col] += m_lambda[k] * matrices.p[k]
                    if flag[indexes.h] != -1:
                        m_out[flag[indexes.h]][lig][col] -= matrices.p[k] * math.log(matrices.p[k] + eps)

                # Scaling
                if flag[indexes.alpha] != -1:
                    m_out[flag[indexes.alpha]][lig][col] *= 180. / pi
                if flag[indexes.beta] != -1:
                    m_out[flag[indexes.beta]][lig][col] *= 180. / pi
                if flag[indexes.epsi] != -1:
                    m_out[flag[indexes.epsi]][lig][col] *= 180. / pi
                if flag[indexes.delta] != -1:
                    m_out[flag[indexes.delta]][lig][col] *= 180. / pi
                if flag[indexes.gamma] != -1:
                    m_out[flag[indexes.gamma]][lig][col] *= 180. / pi
                if flag[indexes.nhu] != -1:
                    m_out[flag[indexes.nhu]][lig][col] *= 180. / pi
                if flag[indexes.h] != -1:
                    m_out[flag[indexes.h]][lig][col] /= log_4

                if flag[indexes.a] != -1:
                    m_out[flag[indexes.a]][lig][col] = (matrices.p[1] - matrices.p[2]) / (matrices.p[1] + matrices.p[2] + eps)

                if flag[indexes.combHA] != -1:
                    m_out[flag[indexes.combHA]][lig][col] = m_out[flag[indexes.h]][lig][col] * m_out[flag[indexes.a]][lig][col]
                if flag[indexes.combH1mA] != -1:
                    m_out[flag[indexes.combH1mA]][lig][col] = m_out[flag[indexes.h]][lig][col] * (1. - m_out[flag[indexes.a]][lig][col])
                if flag[indexes.comb1mHA] != -1:
                    m_out[flag[indexes.comb1mHA]][lig][col] = (1. - m_out[flag[indexes.h]][lig][col]) * m_out[flag[indexes.a]][lig][col]
                if flag[indexes.comb1mH1mA] != -1:
                    m_out[flag[indexes.comb1mH1mA]][lig][col] = (1. - m_out[flag[indexes.h]][lig][col]) * (1. - m_out[flag[indexes.a]][lig][col])


class App(lib.util.Application):

    def __init__(self, args):
        super().__init__(args)

    def allocate_matrices(self, n_col, n_polar_out, n_win_l, n_win_c, n_lig_block, sub_n_col, n_out):
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
        self.m_out = lib.matrix.matrix3d_float(n_out, n_lig_block, sub_n_col)

    def decomposition_parameters_c2(self, indexes, flag, flag_para):
        # Decomposition parameters
        indexes.alpha = 0
        indexes.delta = 1
        indexes.lmda = 2
        indexes.h = 3
        indexes.a = 4
        indexes.combHA = 5
        indexes.combH1mA = 6
        indexes.comb1mHA = 7
        indexes.comb1mH1mA = 8
        # M = matrix3d_float(2, 2, 2);
        # V = matrix3d_float(2, 2, 2);
        # lambda = vector_float(2);
        n_para = 9
        flag[:n_para] = -1
        n_out = 0
        # Flag Parameters
        if flag_para == 1:
            flag[indexes.alpha] = n_out
            n_out += 1
            flag[indexes.delta] = n_out
            n_out += 1
            flag[indexes.lmda] = n_out
            n_out += 1
        return n_out, n_para

    def decomposition_parameters_t3_c3(self, indexes, flag, flag_para):
        #  Decomposition parameters
        indexes.alpha = 0
        indexes.beta = 1
        indexes.delta = 2
        indexes.gamma = 3
        indexes.lmda = 4
        indexes.h = 5
        indexes.a = 6
        indexes.combHA = 7
        indexes.combH1mA = 8
        indexes.comb1mHA = 9
        indexes.comb1mH1mA = 10
        # M = matrix3d_float(3, 3, 2);
        # V = matrix3d_float(3, 3, 2);
        # lambda = vector_float(3);
        n_para = 11
        flag[:n_para] = -1
        n_out = 0
        # Flag Parameters
        if flag_para == 1:
            flag[indexes.alpha] = n_out
            n_out += 1
            flag[indexes.beta] = n_out
            n_out += 1
            flag[indexes.delta] = n_out
            n_out += 1
            flag[indexes.gamma] = n_out
            n_out += 1
            flag[indexes.lmda] = n_out
            n_out += 1
        return n_out, n_para

    def decomposition_parameters_t4_c4(self, indexes, flag, flag_para):
        # Decomposition parameters
        indexes.alpha = 0
        indexes.beta = 1
        indexes.epsi = 2
        indexes.delta = 3
        indexes.gamma = 4
        indexes.nhu = 5
        indexes.lmda = 6
        indexes.h = 7
        indexes.a = 8
        indexes.combHA = 9
        indexes.combH1mA = 10
        indexes.comb1mHA = 11
        indexes.comb1mH1mA = 12
        # M = matrix3d_float(4, 4, 2);
        # V = matrix3d_float(4, 4, 2);
        # lambda = vector_float(4);
        n_para = 13
        flag[:n_para] = -1
        n_out = 0
        # Flag Parameters
        if flag_para == 1:
            flag[indexes.alpha] = n_out
            n_out += 1
            flag[indexes.beta] = n_out
            n_out += 1
            flag[indexes.epsi] = n_out
            n_out += 1
            flag[indexes.delta] = n_out
            n_out += 1
            flag[indexes.gamma] = n_out
            n_out += 1
            flag[indexes.nhu] = n_out
            n_out += 1
            flag[indexes.lmda] = n_out
            n_out += 1
        return n_out, n_para

    def run(self):
        logging.info('******************** Welcome in h a alpha decomposition ********************')
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
        flag_para = self.args.fl1
        flag_lambda = self.args.fl2
        flag_alpha = self.args.fl3
        flag_entropy = self.args.fl4
        flag_anisotropy = self.args.fl5
        flag_combHA = self.args.fl6
        flag_combH1mA = self.args.fl7
        flag_comb1mHA = self.args.fl8
        flag_comb1mH1mA = self.args.fl9
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
        pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = lib.util.pol_type_config(pol_type)
        logging.info(f'{pol_type=}, {n_polar_in=}, {pol_type_in=}, {n_polar_out=}, {pol_type_out=}')

        # INPUT/OUTPUT FILE CONFIGURATION
        file_name_in = lib.util.init_file_name(pol_type_in, in_dir)
        logging.info(f'{file_name_in=}')

        # INPUT FILE OPENING
        in_datafile = []
        in_valid = None
        in_datafile, in_valid, flag_valid = self.open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

        # OUTPUT FILE OPENING
        file_out2 = ['alpha.bin', 'delta.bin', 'lambda.bin',
                     'entropy.bin', 'anisotropy.bin',
                     'combination_HA.bin', 'combination_H1mA.bin',
                     'combination_1mHA.bin', 'combination_1mH1mA.bin']

        file_out3 = ['alpha.bin', 'beta.bin', 'delta.bin',
                     'gamma.bin', 'lambda.bin',
                     'entropy.bin', 'anisotropy.bin',
                     'combination_HA.bin', 'combination_H1mA.bin',
                     'combination_1mHA.bin', 'combination_1mH1mA.bin']

        file_out4 = ['alpha.bin', 'beta.bin', 'epsilon.bin', 'delta.bin',
                     'gamma.bin', 'nhu.bin', 'lambda.bin',
                     'entropy.bin', 'anisotropy.bin',
                     'combination_HA.bin', 'combination_H1mA.bin',
                     'combination_1mHA.bin', 'combination_1mH1mA.bin']

        flag = lib.matrix.vector_int(13)
        n_out = 0
        indexes = Indexes()
        if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
            n_out, n_para = self.decomposition_parameters_c2(indexes, flag, flag_para)
        elif pol_type_out in ['T3', 'C3']:
            n_out, n_para = self.decomposition_parameters_t3_c3(indexes, flag, flag_para)
        elif pol_type_out in ['T4', 'C4']:
            n_out, n_para = self.decomposition_parameters_t4_c4(indexes, flag, flag_para)

        # Flag Lambda  (must keep the previous selection)
        if flag_lambda == 1:
            if flag[indexes.lmda] == -1:
                flag[indexes.lmda] = n_out
                n_out += 1

        # Flag Alpha  (must keep the previous selection)
        if flag_alpha == 1:
            if flag[indexes.alpha] == -1:
                flag[indexes.alpha] = n_out
                n_out += 1

        # Flag Entropy
        if flag_entropy == 1:
            flag[indexes.h] = n_out
            n_out += 1

        # Flag Anisotropy
        if flag_anisotropy == 1:
            flag[indexes.a] = n_out
            n_out += 1

        # Flag Combinations HA
        if flag_combHA == 1:
            flag[indexes.combHA] = n_out
            n_out += 1

        if flag_combH1mA == 1:
            flag[indexes.combH1mA] = n_out
            n_out += 1

        if flag_comb1mHA == 1:
            flag[indexes.comb1mHA] = n_out
            n_out += 1

        if flag_comb1mH1mA == 1:
            flag[indexes.comb1mH1mA] = n_out
            n_out += 1

        out_file_2 = [None] * 9
        out_file_3 = [None] * 11
        out_file_4 = [None] * 13

        for k in range(n_para):
            if flag[k] != -1:
                if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                    out_file_2[k] = self.open_output_file(os.path.join(out_dir, file_out2[k]))
                if pol_type_out in ['T3', 'C3']:
                    out_file_3[k] = self.open_output_file(os.path.join(out_dir, file_out3[k]))
                if pol_type_out in ['T4', 'C4']:
                    out_file_4[k] = self.open_output_file(os.path.join(out_dir, file_out4[k]))

        # MEMORY ALLOCATION
        n_block_a = 0
        n_block_b = 0
        # Mask
        n_block_a += sub_n_col + n_win_c
        n_block_b += n_win_l * (sub_n_col + n_win_c)
        # Mout = Nout*Nlig*Sub_Ncol
        n_block_a += n_out * sub_n_col
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
        self.allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, n_lig_block[0], sub_n_col, n_out)

        # MASK VALID PIXELS (if there is no MaskFile
        self.set_valid_pixels(flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)

        # DATA PROCESSING
        logging.info('--= Started: data processing =--')
        for np in range(n_polar_in):
            self.rewind(in_datafile[np])
        if flag_valid is True:
            self.rewind(in_valid)

        matrices = Matrices()
        eps = lib.util.Application.EPS
        pi = lib.util.Application.PI
        for nb in range(nb_block):
            if nb_block > 2:
                logging.debug('%f\r' % (100 * nb / (nb_block - 1)), end='', flush=True)
            if flag_valid is True:
                lib.util_block.read_block_matrix_float(in_valid, self.valid, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type in ['S2', 'SPP', 'SPPpp1', 'SPPpp2', 'SPPpp3', 'S2']:
                if pol_type == 'S2':
                    lib.util_block.read_block_s2_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
                else:
                    lib.util_block.read_block_spp_noavg(in_datafile, self.m_in, pol_type_out, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.mc_in)
            else:  # Case of C,T or I
                lib.util_block.read_block_tci_noavg(in_datafile, self.m_in, n_polar_out, nb, nb_block, n_lig_block[nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col, self.vf_in)

            if pol_type_in == 'C3' and pol_type_out == 'T3':
                lib.util_convert.c3_to_t3(self.m_in, n_lig_block[nb], sub_n_col + n_win_c, 0, 0)
            elif pol_type_in == 'C4' and pol_type_out == 'T4':
                lib.util_convert.c4_to_t4(self.m_in, n_lig_block[nb], sub_n_col + n_win_c, 0, 0)

            if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                process_c2(nb, n_lig_block, n_polar_out, sub_n_col, self.m_in, self.valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, n_out, self.m_out, eps, flag, matrices, indexes, pi)
            elif pol_type_out in ['T3', 'C3']:
                process_t3_c3(nb, n_lig_block, n_polar_out, sub_n_col, self.m_in, self.valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, n_out, self.m_out, eps, flag, matrices, indexes, pi)
            elif pol_type_in == 'T4' and pol_type_out == 'C4':
                process_t4_c4(nb, n_lig_block, n_polar_out, sub_n_col, self.m_in, self.valid, n_win_l, n_win_c, n_win_l_m1s2, n_win_c_m1s2, n_out, self.m_out, eps, flag, matrices, indexes, pi)
            for k in range(n_para):
                if flag[k] != -1:
                    if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                        lib.util_block.write_block_matrix_matrix3d_float(out_file_2[flag[k]], self.m_out, flag[k], n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
                    if pol_type_out in ['T3', 'C3']:
                        lib.util_block.write_block_matrix_matrix3d_float(out_file_3[flag[k]], self.m_out, flag[k], n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)
                    if pol_type_in == 'T4' and pol_type_out == 'C4':
                        lib.util_block.write_block_matrix_matrix3d_float(out_file_4[flag[k]], self.m_out, flag[k], n_lig_block[nb], sub_n_col, 0, 0, sub_n_col)


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
        fl1 Flag Parameters (0/1)
        fl2 Flag Lambda (0/1)
        fl3 Flag Alpha (0/1)
        fl4 Flag Entropy (0/1)
        fl5 Flag Anisotropy (0/1)
        fl6 Flag Comb HA (0/1)
        fl7 Flag Comb H1mA (0/1)
        fl8 Flag Comb 1mHA (0/1)
        fl9 Flag Comb 1mH1mA (0/1)
        errf (str): memory error file
        mask (str): mask file (valid pixels)'
    '''
    POL_TYPE_VALUES = ['S2T3', 'S2C3', 'S2T4', 'S2C4', 'SPPC2', 'C2', 'C3', 'C3T3', 'C4', 'C4T4', 'T3', 'T4']
    local_args = lib.util.ParseArgs.get_args(*args, **kwargs)
    parser_args = lib.util.ParseArgs(args=local_args, desc=f'{os.path.basename(sys.argv[0])}', pol_types=POL_TYPE_VALUES)
    parser_args.make_def_args()
    parser_args.add_req_arg('-fl1', int, 'Flag Parameters (0/1)', {0, 1})
    parser_args.add_req_arg('-fl2', int, 'Flag Lambda (0/1)', {0, 1})
    parser_args.add_req_arg('-fl3', int, 'Flag Alpha (0/1)', {0, 1})
    parser_args.add_req_arg('-fl4', int, 'Flag Entropy (0/1)', {0, 1})
    parser_args.add_req_arg('-fl5', int, 'Flag Anisotropy (0/1)', {0, 1})
    parser_args.add_req_arg('-fl6', int, 'Flag Comb HA (0/1)', {0, 1})
    parser_args.add_req_arg('-fl7', int, 'Flag Comb H1mA (0/1)', {0, 1})
    parser_args.add_req_arg('-fl8', int, 'Flag Comb 1mHA (0/1)', {0, 1})
    parser_args.add_req_arg('-fl9', int, 'Flag Comb 1mH1mA (0/1)', {0, 1})
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
            dir_in = 'c:\\Projekty\\polsarpro.svn\\in\\h_a_alpha_decomposition\\'
            dir_out = 'c:\\Projekty\\polsarpro.svn\\out\\h_a_alpha_decomposition\\py\\'
        elif platform.system().lower().startswith('lin') is True:
            dir_in = '/home/krzysiek/polsarpro/in/h_a_alpha_decomposition/'
            dir_out = '/home/krzysiek/polsarpro/out/h_a_alpha_decomposition/'
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
        params['fnc'] = 1248
        params['fl1'] = 1
        params['fl2'] = 0
        params['fl3'] = 0
        params['fl4'] = 0
        params['fl5'] = 0
        params['fl6'] = 0
        params['fl7'] = 0
        params['fl8'] = 0
        params['fl9'] = 0
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
        #      fl1 = 1,
        #      fl2 = 0,
        #      fl3 = 0,
        #      fl4 = 0,
        #      fl5 = 0,
        #      fl6 = 0,
        #      fl7 = 0,
        #      fl8 = 0,
        #      fl9 = 0,
        #      errf=os.path.join(f'{dir_out}', 'MemoryAllocError.txt'),
        #      mask=os.path.join(f'{dir_in}', 'mask_valid_pixels.bin'))
