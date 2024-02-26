"""
Polsarpro
===
Description :  Cloude-Pottier eigenvector/eigenvalue based decomposition

Input parameters:
id  	input directory
od  	output directory
iodf	input-output data format
nwr 	Nwin Row
nwc 	Nwin Col
ofr 	Offset Row
ofc 	Offset Col
fnr 	Final Number of Row
fnc 	Final Number of Col
fl1 	Flag Parameters (0/1)
fl2 	Flag Lambda (0/1)
fl3 	Flag Alpha (0/1)
fl4 	Flag Entropy (0/1)
fl5 	Flag Anisotropy (0/1)
fl6 	Flag Comb HA (0/1)
fl7 	Flag Comb H1mA (0/1)
fl8 	Flag Comb 1mHA (0/1)
fl9 	Flag Comb 1mH1mA (0/1)

Optional Parameters:
mask	mask file (valid pixels)
errf	memory error file
help	displays this message
data	displays the help concerning Data Format parameter
"""

# %% [codecell] import
import math
import os
import sys
import argparse
import numpy as np
from collections import namedtuple
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool
from numba import jit
from numpy import linalg as LA
from math import acos, sqrt, atan2, sin, cos, pi, log

import processing
import util
import util_block

# %% [codecell] main
def main():
    """
    Main Function for the BoxCar fully polarimetric speckle filter.
    Parses the input arguments, reads the input files, and processes the data using boxcar filtering.
    """
    print("********************Welcome in boxcar_filter********************")

    Config = 0
    pol_type_conf = ["S2T3", "S2C3", "S2T4", "S2C4", "SPPC2", "C2", "C3", "C3T3", "C4", "C4T4", "T3", "T4"]
    file_name = ""
    eps = 1.E-30
    n_lig_block = np.zeros(8192, dtype=int)

    flag = [0]*13
    n_out = 0
    n_Para = 0

    out_file_2 = [None] * 9
    out_file_3 = [None] * 11
    out_file_4 = [None] * 13

    file_out_2 = [
        "alpha.bin", "delta.bin", "lambda.bin",
        "entropy.bin", "anisotropy.bin",
        "combination_HA.bin", "combination_H1mA.bin",
        "combination_1mHA.bin", "combination_1mH1mA.bin"]

    file_out_3 = [
        "alpha.bin", "beta.bin", "delta.bin",
        "gamma.bin", "lambda.bin",
        "entropy.bin", "anisotropy.bin",
        "combination_HA.bin", "combination_H1mA.bin",
        "combination_1mHA.bin", "combination_1mH1mA.bin"]

    file_out_4 = [
        "alpha.bin", "beta.bin", "epsilon.bin", "delta.bin",
        "gamma.bin", "nhu.bin", "lambda.bin",
        "entropy.bin", "anisotropy.bin",
        "combination_HA.bin", "combination_H1mA.bin",
        "combination_1mHA.bin", "combination_1mH1mA.bin"]

    flag_para = flag_lambda = flag_alpha = 0
    flag_entropy = flag_anisotropy = 0
    flag_comb_ha = flag_comb_h1ma = flag_comb_1mha = flag_comb_1mh1ma = 0

    Alpha = Beta = Epsi = Delta = Gamma = Nhu = Lambda = 0
    H = A = comb_ha = comb_h1ma = comb_1mha = comb_1mh1ma = 0

    # Internal variables
    ii, lig, col, k = 0, 0, 0, 0
    ligDone = 0


    # Matrix arrays - we will define these as empty lists for now as we don't know their sizes
    m_avg = []
    m_in = []
    m_out = []

    lambda_ = []

    #   ********************************************************************
    #   ********************************************************************/
    #   /* USAGE */
    args = parse_arguments(pol_type_conf)
    in_dir = args.id
    out_dir = args.od
    pol_type = args.iodf
    n_win_l = args.nwr
    n_win_c = args.nwc
    off_lig = args.ofr
    off_col = args.ofc
    sub_n_lig = args.fnr
    sub_n_col = args.fnc

    flag_para = args.fl1
    flag_lambda = args.fl2
    flag_alpha = args.fl3
    flag_entropy = args.fl4
    flag_anisotropy = args.fl5
    flag_comb_ha = args.fl6
    flag_comb_h1ma = args.fl7
    flag_comb_1mha = args.fl8
    flag_comb_1mh1ma = args.fl9

    file_memerr = args.errf
    file_valid = args.mask
    data_help = args.data

    n_win_lm1s2 = int((n_win_l - 1) / 2)
    n_win_cm1s2 = int((n_win_c - 1) / 2)

    # # /* INPUT/OUPUT CONFIGURATIONS */
    n_lig, n_col, polar_case, polar_type = read_configuration(in_dir)

    # # /* POLAR TYPE CONFIGURATION */
    pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = configure_polar_type(pol_type)

    # # /* INPUT/OUTPUT FILE CONFIGURATION */
    file_name_in, file_name_out = configure_input_output_files(pol_type_in, pol_type_out, in_dir, out_dir)

    # # /* INPUT FILE OPENING*/
    in_datafile, in_valid, flag_valid = open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

    # /* OUTPUT FILE OPENING*/
    flag, Alpha, Beta, Epsi, Delta, Gamma, Nhu, Lambda, n_out, n_para = open_output_files(pol_type_out, n_out, flag, flag_para, flag_lambda, flag_alpha, flag_entropy, flag_anisotropy, flag_comb_ha, flag_comb_h1ma, flag_comb_1mha, flag_comb_1mh1ma, out_dir, file_out_2, out_file_2, out_file_3, out_file_4)

    # /* MATRIX ALLOCATION */
    util.vc_in, util.vf_in, util.mc_in, util.mf_in, valid, m_in, m_out = allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col, n_out)

    # /* MASK VALID PIXELS (if there is no MaskFile */
    valid = set_valid_pixels(valid, flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)

    # /********************************************************************
    # ********************************************************************/
    # /* DATA PROCESSING */
    nb_block = 1
    for block in range(nb_block):
        ligDone = 0

        if flag_valid == 1:
            util_block.read_block_matrix_float(in_valid, valid, block, nb_block, n_lig_block[block], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        if (pol_type_in in ["S2", "SPP", "SPPpp1", "SPPpp2", "SPPpp3"]):
            if pol_type_in == "S2":
                util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, block, nb_block, n_lig_block[block], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
            else:
                util_block.read_block_spp_noavg(in_datafile, m_in, pol_type_out, n_polar_out, block, nb_block, n_lig_block[block], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
        else:
            # Case of C,T or I 
            util_block.read_block_tci_noavg(in_datafile, m_in, n_polar_out, block, nb_block, n_lig_block[block], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        if (pol_type_in == "C3" and pol_type_out == "T3"):
            util_block.C3_to_T3(m_in, n_lig_block[block], sub_n_col + n_win_c, 0, 0)
        if (pol_type_in == "C4" and pol_type_out == "T4"):
            util_block.C4_to_T4(m_in, n_lig_block[block], sub_n_col + n_win_c, 0, 0)

        if pol_type_out in ["C2", "C2pp1", "C2pp2", "C2pp3"]:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                m_out_futures = [executor.submit(process_c2, lig, n_lig_block, block, sub_n_col, n_polar_out, m_out, n_win_lm1s2, n_win_cm1s2, valid, m_in, flag, eps, Alpha, Beta, Delta, Gamma, Lambda, H, A, comb_ha, comb_h1ma, comb_1mha, comb_1mh1ma) for lig in range(sub_n_lig)]
                for future in tqdm(concurrent.futures.as_completed(m_out_futures), total=len(m_out_futures), desc="Processing"):
                    pass
        
        elif pol_type_out in ["T3", "C3"]:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                m_out_futures = [executor.submit(process_t3_c3, lig, n_out, n_win_l, n_win_c, sub_n_col, n_polar_out, m_out, phase, n_win_lm1s2, n_win_cm1s2, valid, m_in, flag, eps, Alpha, Beta, Delta, Gamma, Lambda, H, A, comb_ha, comb_h1ma, comb_1mha, comb_1mh1ma) for lig in range(sub_n_lig)]
                for future in tqdm(concurrent.futures.as_completed(m_out_futures), total=len(m_out_futures), desc="Processing"):
                    pass
        

        elif pol_type_out in ["T4", "C4"]:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                            m_out_futures = [executor.submit(process_t4_c4, lig, n_out, n_win_l, n_win_c, sub_n_col, n_polar_out, m_out, phase, n_win_lm1s2, n_win_cm1s2, valid, m_in, flag, eps, Alpha, Beta, Delta, Gamma, Lambda, Epsi, Nhu, H, A, comb_ha, comb_h1ma, comb_1mha, comb_1mh1ma) for lig in range(sub_n_lig)]
                            for future in tqdm(concurrent.futures.as_completed(m_out_futures), total=len(m_out_futures), desc="Processing"):
                                pass

        for k in range(n_para):
            if flag[k] != -1:
                if pol_type_out in ["C2", "C2pp1", "C2pp2", "C2pp3"]:
                    util_block.write_block_matrix_matrix3d_float(out_file_2[flag[k]], m_out, flag[k], n_lig_block[0], sub_n_col, 0, 0, sub_n_col)
                if pol_type_out in ["T3", "C3"]:
                    util_block.write_block_matrix_matrix3d_float(out_file_3[flag[k]], m_out, flag[k], n_lig_block[0], sub_n_col, 0, 0, sub_n_col)
                if pol_type_out in ["T4", "C4"]:
                    util_block.write_block_matrix_matrix3d_float(out_file_4[flag[k]], m_out, flag[k], n_lig_block[0], sub_n_col, 0, 0, sub_n_col)

        
# %% [codecell] parse_arguments
def parse_arguments(pol_type_conf):
    """
    Parse command line arguments and return them as an 'args' object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type=str, required=True, help="input directory")
    parser.add_argument("-od", type=str, required=True, help="output directory")
    parser.add_argument("-iodf", type=str, required=True, choices=pol_type_conf, help="input-output data format")
    parser.add_argument("-nwr", type=int, required=True, help="Nwin Row")
    parser.add_argument("-nwc", type=int, required=True, help="Nwin Col")
    parser.add_argument("-ofr", type=int, required=True, help="Offset Row")
    parser.add_argument("-ofc", type=int, required=True, help="Offset Col")
    parser.add_argument("-fnr", type=int, required=True, help="Final Number of Row")
    parser.add_argument("-fnc", type=int, required=True, help="Final Number of Col")

    parser.add_argument("-fl1", type=int, required=True, help="Flag Parameters (0/1)")
    parser.add_argument("-fl2", type=int, required=True, help="Flag Lambda (0/1)")
    parser.add_argument("-fl3", type=int, required=True, help="Flag Alpha (0/1)")
    parser.add_argument("-fl4", type=int, required=True, help="Flag Entropy (0/1)")
    parser.add_argument("-fl5", type=int, required=True, help="Flag Anisotropy (0/1)")
    parser.add_argument("-fl6", type=int, required=True, help="Flag Comb HA (0/1)")
    parser.add_argument("-fl7", type=int, required=True, help="Flag Comb H1mA (0/1)")
    parser.add_argument("-fl8", type=int, required=True, help="Flag Comb 1mHA (0/1)")
    parser.add_argument("-fl9", type=int, required=True, help="Flag Comb 1mH1mA (0/1)")

    parser.add_argument("-mask", type=str, required=False, help="Optional - mask file (valid pixels)")
    parser.add_argument("-errf", type=str, required=False, help="Optional - memory error file")
    parser.add_argument("-data", action='store_true', required=False, help="Optional - displays the help concerning Data Format parameter")
    
    args = parser.parse_args()

    return args

# %% [codecell] read_configuration
def read_configuration(in_dir):
    """
    Read the configuration from the input directory and return the parameters.
    """
    n_lig, n_col, polar_case, polar_type = util.read_config(in_dir)
    return n_lig, n_col, polar_case, polar_type

# %% [codecell] configure_polar_type
def configure_polar_type(pol_type):
    """
    Configure the polar type based on the provided input-output data format and return the updated parameters.
    """
    return util.pol_type_config(pol_type)

# %% [codecell] configure_input_output_files
def configure_input_output_files(pol_type_in, in_dir):
    """
    Configure the input and output files based on the provided polar types and directories.
    Return the input and output file names.
    """
    init_file_name = util.init_file_name(pol_type_in, in_dir)
    return init_file_name

# %% [codecell] open_input_files
def open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid):
    """
    Open input files and return the file objects and a flag indicating if the 'valid' file is present.
    """
    flag_valid = False
    for n_pol in range(n_polar_in):
        try:
            in_datafile.append(open(file_name_in[n_pol], "rb"))
        except IOError:
            print("Could not open input file : ", file_name_in[n_pol])
            raise

    if file_valid:
        flag_valid = True
        try:
            in_valid = open(file_valid, "rb")
        except IOError:
            print("Could not open input file: ", file_valid)
            raise
    return in_datafile, in_valid, flag_valid

# %% [codecell] open_output_files
def open_output_files(pol_type_out, n_out, flag, flag_para, flag_lambda, flag_alpha, flag_entropy, flag_anisotropy, flag_comb_ha, flag_comb_h1ma, flag_comb_1mha, flag_comb_1mh1ma, out_dir, file_out_2, out_file_2, out_file_3, out_file_4):
    """
    Open output files and return the file objects.
    """
    if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
        # Decomposition parameters
        Alpha, Delta, Lambda = 0, 1, 2
        H, A, CombHA, CombH1mA, Comb1mHA, Comb1mH1mA = 3, 4, 5, 6, 7, 8

        # M = matrix3d_float(2, 2, 2)
        # V = matrix3d_float(2, 2, 2)
        # lambda = vector_float(2)
        n_para = 9
        for k in range(n_para):
            flag[k] = -1
        # Flag Parameters
        if flag_para == 1:
            flag[Alpha] = n_out; n_out += 1
            flag[Delta] = n_out; n_out += 1
            flag[Lambda] = n_out; n_out += 1

    # T3, C3
    if pol_type_out in ['T3', 'C3']:
        # Decomposition parameters
        Alpha, Beta, Delta, Gamma, Lambda = 0, 1, 2, 3, 4
        H, A, CombHA, CombH1mA, Comb1mHA, Comb1mH1mA = 5, 6, 7, 8, 9, 10

        # M = matrix3d_float(3, 3, 2)
        # V = matrix3d_float(3, 3, 2)
        # lambda = vector_float(3)

        n_para = 11
        for k in range(n_para):
            flag[k] = -1
        n_out = 0
        # Flag Parameters
        if flag_para == 1:
            flag[Alpha] = n_out; n_out += 1
            flag[Beta] = n_out; n_out += 1
            flag[Delta] = n_out; n_out += 1
            flag[Gamma] = n_out; n_out += 1
            flag[Lambda] = n_out; n_out += 1

    # T4, C4
    if pol_type_out in ['T4', 'C4']:
        # Decomposition parameters
        Alpha, Beta, Epsi, Delta, Gamma, Nhu, Lambda = 0, 1, 2, 3, 4, 5, 6
        H, A, CombHA, CombH1mA, Comb1mHA, Comb1mH1mA = 7, 8, 9, 10, 11, 12

        # M = matrix3d_float(4, 4, 2)
        # V = matrix3d_float(4, 4, 2)
        # lambda = vector_float(4)

        n_para = 13
        for k in range(n_para):
            flag[k] = -1
        n_out = 0
        # Flag Parameters
        if flag_para == 1:
            flag[Alpha] = n_out; n_out += 1
            flag[Beta] = n_out; n_out += 1
            flag[Epsi] = n_out; n_out += 1
            flag[Delta] = n_out; n_out += 1
            flag[Gamma] = n_out; n_out += 1
            flag[Nhu] = n_out; n_out += 1
            flag[Lambda] = n_out; n_out += 1

    # Flag Lambda (must keep the previous selection)
    if flag_lambda == 1:
        if flag[Lambda] == -1:
            flag[Lambda] = n_out
            n_out += 1

    # Flag Alpha (must keep the previous selection)
    if flag_alpha == 1:
        if flag[Alpha] == -1:
            flag[Alpha] = n_out
            n_out += 1

    # Flag Entropy
    if flag_entropy == 1:
        flag[H] = n_out
        n_out += 1

    # Flag Anisotropy
    if flag_anisotropy == 1:
        flag[A] = n_out
        n_out += 1

    # Flag Combinations HA
    if flag_comb_ha == 1:
        flag[CombHA] = n_out
        n_out += 1

    if flag_comb_h1ma == 1:
        flag[CombH1mA] = n_out
        n_out += 1

    if flag_comb_1mha == 1:
        flag[Comb1mHA] = n_out
        n_out += 1

    if flag_comb_1mh1ma == 1:
        flag[Comb1mH1mA] = n_out
        n_out += 1

    for k in range(n_para):
        if flag[k] != -1:
            # C2, C2pp1, C2pp2, C2pp3
            if pol_type_out in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
                file_name = os.path.join(out_dir, file_out_2[k])
                try:
                    out_file_2[flag[k]] = open(file_name, "wb")
                except IOError:
                    raise Exception("Could not open input file : ", file_name)
            # T3, C3
            elif pol_type_out in ['T3', 'C3']:
                file_name = os.path.join(out_dir, out_file_3[k])
                try:
                    out_file_3[flag[k]] = open(file_name, "wb")
                except IOError:
                    raise Exception("Could not open input file : ", file_name)
            # T4, C4
            elif pol_type_out in ['T4', 'C4']:
                file_name = os.path.join(out_dir, out_file_4[k])
                try:
                    out_file_4[flag[k]] = open(file_name, "wb")
                except IOError:
                    raise Exception("Could not open input file : ", file_name)
    return flag, Alpha, Beta, Epsi, Delta, Gamma, Nhu, Lambda, H, A, CombHA, CombH1mA, Comb1mHA, Comb1mH1mA, n_out, n_para

# %% [codecell] set_valid_pixels
def set_valid_pixels(valid, flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l):
    """
    Set the valid pixels for the h_a_alpha based on the provided parameters.
    """
    if not flag_valid:
        valid[:n_lig_block[0] + n_win_l, :sub_n_col + n_win_c] = 1.0
    return valid

# %% [codecell] allocate_matrices
def allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col, n_out):
    """
    Allocate matrices with given dimensions
    """
    vc_in = np.zeros(2 * n_col, dtype=float)
    vf_in = np.zeros(n_col, dtype=float)
    mc_in = np.zeros((4, 2 * n_col), dtype=float)
    mf_in = np.zeros((n_polar_out, n_win_l, n_col + n_win_c), dtype=float)

    valid = np.zeros((sub_n_lig + n_win_l, sub_n_col + n_win_c), dtype=float)
    m_in = np.zeros((n_polar_out, sub_n_lig + n_win_l, n_col + n_win_c), dtype=float)
    m_out = np.zeros((n_polar_out, sub_n_lig, sub_n_col), dtype=float)

    return vc_in, vf_in, mc_in, mf_in, valid, m_in, m_out

# %% [codecell] process_C2
@jit(nopython=True)
def process_c2(lig, n_out, n_lig_block, block, sub_n_col, n_polar_out, m_out, n_win_lm1s2, n_win_cm1s2, valid, m_in, flag, eps, Alpha, Beta, Delta, Gamma, Lambda, H, A, comb_ha, comb_h1ma, comb_1mha, comb_1mh1ma):
    alpha = np.zeros(4)
    beta = np.zeros(4)
    delta = np.zeros(4)
    gamma = np.zeros(4)
    p = np.zeros(4)
    M = np.zeros((2, 2, 2))
    V = np.zeros((2, 2, 2))
    lambda_ = np.zeros(2)
    m_avg = np.zeros((n_polar_out, sub_n_col))
    # Assuming average_TCI() is another function to be implemented
    m_avg = util_block.average_TCI(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_lm1s2, n_win_cm1s2)

    for col in range(sub_n_col):
        for k in range(n_out):
            m_out[k][lig][col] = 0.0
        if valid[n_win_lm1s2+lig][n_win_cm1s2+col] == 1.0:
            M[0][0][0] = eps + m_avg[0, col]
            M[0][0][1] = 0
            M[0][1][0] = eps + m_avg[1, col]
            M[0][1][1] = eps + m_avg[2, col]
            M[1][0][0] = M[0, 1, 0]
            M[1][0][1] = -M[0, 1, 1]
            M[1][1][0] = eps + m_avg[3, col]
            M[1][1][1] = 0

            # Diagonalisation function call
            V, lambda_ = processing.diagonalisation(2, M, V, lambda_)

            # Mean scattering mechanism
            if flag[Alpha] != -1: 
                m_out[flag[Alpha]][lig][col] = 0
            if flag[Beta] != -1: 
                m_out[flag[Beta]][lig][col] = 0
            if flag[Delta] != -1: 
                m_out[flag[Delta]][lig][col] = 0
            if flag[Gamma] != -1: 
                m_out[flag[Gamma]][lig][col] = 0
            if flag[Lambda] != -1: 
                m_out[flag[Lambda]][lig][col] = 0
            if flag[H] != -1: 
                m_out[flag[H]][lig][col] = 0

            for k in range(3):
                if flag[Alpha] != -1: 
                    m_out[flag[Alpha]][lig][col] += alpha[k] * p[k]
                if flag[Beta] != -1: 
                    m_out[flag[Beta]][lig][col] += beta[k] * p[k]
                if flag[Delta] != -1: 
                    m_out[flag[Delta]][lig][col] += delta[k] * p[k]
                if flag[Gamma] != -1: 
                    m_out[flag[Gamma]][lig][col] += gamma[k] * p[k]
                if flag[Lambda] != -1: 
                    m_out[flag[Lambda]][lig][col] += lambda_[k] * p[k]
                if flag[H] != -1: 
                    m_out[flag[H]][lig][col] -= p[k] * math.log(p[k] + eps)

            # Scaling
            if flag[Alpha] != -1: 
                m_out[flag[Alpha]][lig][col] *= 180. / math.pi
            if flag[Beta] != -1: 
                m_out[flag[Beta]][lig][col] *= 180. / math.pi
            if flag[Delta] != -1: 
                m_out[flag[Delta]][lig][col] *= 180. / math.pi
            if flag[Gamma] != -1: 
                m_out[flag[Gamma]][lig][col] *= 180. / math.pi
            if flag[H] != -1: 
                m_out[flag[H]][lig][col] /= math.log(3)
            if flag[A] != -1:
                m_out[flag[A], lig, col] = (p[1] - p[2]) / (p[1] + p[2] + eps)

            if flag[comb_ha] != -1:
                m_out[flag[comb_ha], lig, col] = m_out[flag[H], lig, col] * m_out[flag[A], lig, col]
            if flag[comb_h1ma] != -1:
                m_out[flag[comb_h1ma], lig, col] = m_out[flag[H], lig, col] * (1. - m_out[flag[A], lig, col])
            if flag[comb_1mha] != -1:
                m_out[flag[comb_1mha], lig, col] = (1. - m_out[flag[H], lig, col]) * m_out[flag[A], lig, col]
            if flag[comb_1mh1ma] != -1:
                m_out[flag[comb_1mh1ma], lig, col] = (1. - m_out[flag[H], lig, col]) * (1. - m_out[flag[A], lig, col])
    return m_out
# %% [codecell] process_T3_C3
@jit(nopython=True)
def process_t3_c3(lig, n_out, n_win_l, n_win_c, sub_n_col, n_polar_out, m_out, phase, n_win_lm1s2, n_win_cm1s2, valid, m_in, flag, eps, Alpha, Beta, Delta, Gamma, Lambda, H, A, comb_ha, comb_h1ma, comb_1mha, comb_1mh1ma):
    alpha = np.zeros(4)
    beta = np.zeros(4)
    delta = np.zeros(4)
    gamma = np.zeros(4)
    p = np.zeros(4)
    M = np.zeros((3, 3, 2))
    V = np.zeros((3, 3, 2))
    lambda_ = np.zeros(3)
    m_avg = np.zeros((n_polar_out, sub_n_col))
    # Assuming average_TCI() is another function to be implemented
    m_avg = util_block.average_TCI(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm1s2, n_win_cm1s2)
    M = 0
    for col in range(sub_n_col):
        for k in range(n_out):
            m_out[k][lig][col] = 0.
        if valid[n_win_lm1s2+lig][n_win_cm1s2+col] == 1.:
            
            M[0][0][0] = eps + m_avg[0][col]
            M[0][0][1] = 0.
            M[0][1][0] = eps + m_avg[1][col]
            M[0][1][1] = eps + m_avg[2][col]
            M[0][2][0] = eps + m_avg[3][col]
            M[0][2][1] = eps + m_avg[4][col]
            M[1][0][0] = M[0][1][0]
            M[1][0][1] = -M[0][1][1]
            M[1][1][0] = eps + m_avg[5][col]
            M[1][1][1] = 0.
            M[1][2][0] = eps + m_avg[6][col]
            M[1][2][1] = eps + m_avg[7][col]
            M[2][0][0] = M[0][2][0]
            M[2][0][1] = -M[0][2][1]
            M[2][1][0] = M[1][2][0]
            M[2][1][1] = -M[1][2][1]
            M[2][2][0] = eps + m_avg[8][col]
            M[2][2][1] = 0.

            V, lambda_ = processing.diagonalisation(3, M, V, lambda_)

            for k in range(3):
                if lambda_[k] < 0.: 
                    lambda_[k] = 0.
            for k in range(3):
                alpha[k] = math.acos(math.sqrt(V[0][k][0] ** 2 + V[0][k][1] ** 2))
                phase[k] = math.atan2(V[0][k][1], eps + V[0][k][0])
                beta[k] = math.atan2(math.sqrt(V[2][k][0] ** 2 + V[2][k][1] ** 2), eps + math.sqrt(V[1][k][0] ** 2 + V[1][k][1] ** 2))
                delta[k] = math.atan2(V[1][k][1], eps + V[1][k][0]) - phase[k]
                delta[k] = math.atan2(math.sin(delta[k]), math.cos(delta[k]) + eps)
                gamma[k] = math.atan2(V[2][k][1], eps + V[2][k][0]) - phase[k]
                gamma[k] = math.atan2(math.sin(gamma[k]), math.cos(gamma[k]) + eps)
                
                # Mean scattering mechanism
                if flag[Alpha] != -1: m_out[flag[Alpha]][lig][col] = 0
                if flag[Beta] != -1: m_out[flag[Beta]][lig][col] = 0
                if flag[Delta] != -1: m_out[flag[Delta]][lig][col] = 0
                if flag[Gamma] != -1: m_out[flag[Gamma]][lig][col] = 0
                if flag[Lambda] != -1: m_out[flag[Lambda]][lig][col] = 0
                if flag[H] != -1: m_out[flag[H]][lig][col] = 0

                # Scaling
                if flag[Alpha] != -1: m_out[flag[Alpha]][lig][col] *= 180. / math.pi
                if flag[Beta] != -1: m_out[flag[Beta]][lig][col] *= 180. / math.pi
                if flag[Delta] != -1: m_out[flag[Delta]][lig][col] *= 180. / math.pi
                if flag[Gamma] != -1: m_out[flag[Gamma]][lig][col] *= 180. / math.pi
                if flag[H] != -1: m_out[flag[H]][lig][col] /= math.log(3.)

                if flag[comb_ha] != -1: m_out[flag[comb_ha]][lig][col] = m_out[flag[H]][lig][col] * m_out[flag['A']][lig][col]
                if flag[comb_h1ma] != -1: m_out[flag[comb_h1ma]][lig][col] = m_out[flag[H]][lig][col] * (1. - m_out[flag['A']][lig][col])
                if flag[comb_1mha] != -1: m_out[flag[comb_1mha]][lig][col] = (1. - m_out[flag[H]][lig][col]) * m_out[flag['A']][lig][col]
                if flag[comb_1mh1ma] != -1: m_out[flag[comb_1mh1ma]][lig][col] = (1. - m_out[flag[H]][lig][col]) * (1. - m_out[flag['A']][lig][col])
    return m_out

# %% [codecell] process_T4_C4
@jit(nopython=True)
def process_t4_c4(lig, n_out, n_win_l, n_win_c, sub_n_col, n_polar_out, m_out, phase, n_win_lm1s2, n_win_cm1s2, valid, m_in, flag, eps, Alpha, Beta, Delta, Gamma, Lambda, Epsi, Nhu, H, A, comb_ha, comb_h1ma, comb_1mha, comb_1mh1ma):
    alpha = np.zeros(4)
    beta = np.zeros(4)
    delta = np.zeros(4)
    gamma = np.zeros(4)
    epsilon = np.zeros(4)
    nhu = np.zeros(4    )
    p = np.zeros(4)
    M = np.zeros((4, 4, 2))
    V = np.zeros((4, 4, 2))
    lambda_ = np.zeros((4,))
    M_avg = np.zeros((n_polar_out, sub_n_col))
    # Assuming average_TCI() is another function to be implemented
    m_avg = util_block.average_TCI(m_in, valid, n_polar_out, M_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm1s2, n_win_cm1s2)
    M = 0
    for col in range(sub_n_col):
        for k in range(n_out):
            m_out[k][lig][col] = 0.
        if valid[n_win_lm1s2+lig][n_win_cm1s2+col] == 1.:
            M[0][0][0] = eps + M_avg[0][col]
            M[0][0][1] = 0.
            M[0][1][0] = eps + M_avg[1][col]
            M[0][1][1] = eps + M_avg[2][col]
            M[0][2][0] = eps + M_avg[3][col]
            M[0][2][1] = eps + M_avg[4][col]
            M[0][3][0] = eps + M_avg[5][col]
            M[0][3][1] = eps + M_avg[6][col]
            M[1][0][0] =  M[0][1][0]
            M[1][0][1] = -M[0][1][1]
            M[1][1][0] = eps + M_avg[7][col]
            M[1][1][1] = 0.
            M[1][2][0] = eps + M_avg[8][col]
            M[1][2][1] = eps + M_avg[9][col]
            M[1][3][0] = eps + M_avg[10][col]
            M[1][3][1] = eps + M_avg[11][col]
            M[2][0][0] =  M[0][2][0]
            M[2][0][1] = -M[0][2][1]
            M[2][1][0] =  M[1][2][0]
            M[2][1][1] = -M[1][2][1]
            M[2][2][0] = eps + M_avg[12][col]
            M[2][2][1] = 0.
            M[2][3][0] = eps + M_avg[13][col]
            M[2][3][1] = eps + M_avg[14][col]
            M[3][0][0] =  M[0][3][0]
            M[3][0][1] = -M[0][3][1]
            M[3][1][0] =  M[1][3][0]
            M[3][1][1] = -M[1][3][1]
            M[3][2][0] =  M[2][3][0]
            M[3][2][1] = -M[2][3][1]
            M[3][3][0] = eps + M_avg[15][col]
            M[3][3][1] = 0.

            V, lambda_ = processing.diagonalisation(4, M, V, lambda_)

            for k in range(4):
                alpha[k] = acos(sqrt(V[0][k][0]**2 + V[0][k][1]**2))
                phase[k] = atan2(V[0][k][1], eps + V[0][k][0])
                beta[k] = atan2(sqrt(V[2][k][0]**2 + V[2][k][1]**2 + V[3][k][0]**2 + V[3][k][1]**2), 
                                eps + sqrt(V[1][k][0]**2 + V[1][k][1]**2))
                epsilon[k] = atan2(sqrt(V[3][k][0]**2 + V[3][k][1]**2), 
                                eps + sqrt(V[2][k][0]**2 + V[2][k][1]**2))
                delta[k] = atan2(V[1][k][1], eps + V[1][k][0]) - phase[k]
                delta[k] = atan2(sin(delta[k]), cos(delta[k]) + eps)
                gamma[k] = atan2(V[2][k][1], eps + V[2][k][0]) - phase[k]
                gamma[k] = atan2(sin(gamma[k]), cos(gamma[k]) + eps)
                nhu[k] = atan2(V[3][k][1], eps + V[3][k][0]) - phase[k]
                nhu[k] = atan2(sin(nhu[k]), cos(nhu[k]) + eps)
                
                # Scattering mechanism probability of occurrence
                p[k] = lambda_[k] / (eps + lambda_[0] + lambda_[1] + lambda_[2] + lambda_[3])
                if p[k] < 0.: 
                    p[k] = 0.
                if p[k] > 1.: 
                    p[k] = 1.
                flag = 0
                m_out = 0
                # Mean scattering mechanism
                if flag[Alpha] != -1:
                    m_out[flag[Alpha]][lig][col] = 0
                if flag[Beta] != -1:
                    m_out[flag[Beta]][lig][col] = 0
                if flag[Epsi] != -1:
                    m_out[flag[Epsi]][lig][col] = 0
                if flag[Delta] != -1:
                    m_out[flag[Delta]][lig][col] = 0
                if flag[Gamma] != -1:
                    m_out[flag[Gamma]][lig][col] = 0
                if flag[Nhu] != -1:
                    m_out[flag[Nhu]][lig][col] = 0
                if flag[Lambda] != -1:
                    m_out[flag[Lambda]][lig][col] = 0
                if flag[H] != -1:
                    m_out[flag[H]][lig][col] = 0

                from math import log
                for k in range(4):
                    if flag[Alpha] != -1:
                        m_out[flag[Alpha]][lig][col] += alpha[k] * p[k]
                    if flag[Beta] != -1:
                        m_out[flag[Beta]][lig][col] += beta[k] * p[k]
                    if flag[Epsi] != -1:
                        m_out[flag[Epsi]][lig][col] += epsilon[k] * p[k]
                    if flag[Delta] != -1:
                        m_out[flag[Delta]][lig][col] += delta[k] * p[k]
                    if flag[Gamma] != -1:
                        m_out[flag[Gamma]][lig][col] += gamma[k] * p[k]
                    if flag[Nhu] != -1:
                        m_out[flag[Nhu]][lig][col] += nhu[k] * p[k]
                    if flag[Lambda] != -1:
                        m_out[flag[Lambda]][lig][col] += lambda_[k] * p[k]
                    if flag[H] != -1:
                        m_out[flag[H]][lig][col] -= p[k] * log(p[k] + eps)

                # Scaling
                if flag[Alpha] != -1:
                    m_out[flag[Alpha]][lig][col] *= 180. / pi
                if flag[Beta] != -1:
                    m_out[flag[Beta]][lig][col] *= 180. / pi
                if flag[Epsi] != -1:
                    m_out[flag[Epsi]][lig][col] *= 180. / pi
                if flag[Delta] != -1:
                    m_out[flag[Delta]][lig][col] *= 180. / pi
                if flag[Gamma] != -1:
                    m_out[flag[Gamma]][lig][col] *= 180. / pi
                if flag[Nhu] != -1:
                    m_out[flag[Nhu]][lig][col] *= 180. / pi
                if flag[H] != -1:
                    m_out[flag[H]][lig][col] /= log(4.)

                if flag[A] != -1:
                    m_out[flag[A]][lig][col] = (p[1] - p[2]) / (p[1] + p[2] + eps)

                if flag[comb_ha] != -1:
                    m_out[flag[comb_ha]][lig][col] = m_out[flag[H]][lig][col] * m_out[flag[A]][lig][col]
                if flag[comb_h1ma] != -1:
                    m_out[flag[comb_h1ma]][lig][col] = m_out[flag[H]][lig][col] * (1. - m_out[flag[A]][lig][col])
                if flag[comb_1mha] != -1:
                    m_out[flag[comb_1mha]][lig][col] = (1. - m_out[flag[H]][lig][col]) * m_out[flag[A]][lig][col]
                if flag[comb_1mh1ma] != -1:
                    m_out[flag[comb_1mh1ma]][lig][col] = (1. - m_out[flag[H]][lig][col]) * (1. - m_out[flag[A]][lig][col])

    return m_out

# %% [codecell] is_pol_type_valid
def is_pol_type_valid(pol_type):
    """
    Check if the given pol_type is valid for processing.
    Returns True if the pol_type is valid, False otherwise.
    """
    valid_pol_types = ["S2", "SPP", "SPPpp1", "SPPpp2", "SPPpp3"]
    return pol_type in valid_pol_types

if __name__ == "__main__":
    main()

