"""
Polsarpro
===
Description :  Freeman Decomposition

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

Optional Parameters:
mask	mask file (valid pixels)
errf	memory error file
help	displays this message
data	displays the help concerning Data Format parameter
"""

import os
import sys
import argparse
import numpy as np
from collections import namedtuple
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool
from numba import jit
import util
import util_block
import numba
import util_convert


def main(in_dir, out_dir, pol_type, n_win_l, n_win_c, off_lig, off_col, sub_n_lig, sub_n_col, file_memerr, file_valid):
    
    """
    Main Function for the freeman.
    Parses the input arguments, reads the input files, and processes the data using freeman filtering.
    """
    print("********************Welcome in freeman_decomposition********************")

    # Definitions
    NPolType = ["S2", "C3", "T3"]
    file_name = ''

    eps = 1.E-30

    n_lig_block = np.zeros(8192, dtype=int)
    n_polar_out = 0
    m_in = []
    valid = 0
    in_datafile = []
    in_valid = 0

    # Internal variables
    ii, lig, col = 0, 0, 0
    ligDone = 0

    span, span_min, span_max = 0.0, 0.0, 0.0

    n_win_lm_1s2 = int((n_win_l - 1) / 2)
    n_win_cm_1s2 = int((n_win_c - 1) / 2)

    # # /* INPUT/OUPUT CONFIGURATIONS */
    n_lig, n_col, polar_case, polar_type = read_configuration(in_dir)  

    # # /* POLAR TYPE CONFIGURATION */
    pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = configure_polar_type(pol_type)

    # # /* INPUT/OUTPUT FILE CONFIGURATION */
    file_name_in = configure_input_output_files(pol_type_in, in_dir, out_dir)

    # # /* INPUT FILE OPENING*/
    in_datafile, in_valid, flag_valid = open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

    # /* OUTPUT FILE OPENING*/
    out_odd, out_dbl, out_vol = open_output_files(out_dir)

    # /* COPY HEADER*/
    copy_header(in_dir, out_dir)

    # /* MATRIX ALLOCATION */
    util.vc_in, util.vf_in, util.mc_in, util.mf_in, valid, m_in, m_odd, m_dbl, m_vol = allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col)

    # /* MASK VALID PIXELS (if there is no MaskFile */
    valid = set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l)

    # SPANMIN / SPANMAX DETERMINATION
    for Np in range(n_polar_in):
        in_datafile[Np].seek(0)
        
    if flag_valid == 1:
        in_valid.seek(0)

    span = 0.0
    span_min = np.inf
    span_max = -np.inf

    nb_block = 1
    for Nb in range(nb_block):
        ligDone = 0
        if nb_block > 2:
            print("%f\r" % (100 * Nb / (nb_block - 1)), end="", flush = True)
        if flag_valid == 1:
            m_in = util_block.read_block_matrix_float(in_valid, valid, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        if pol_type == "S2":
            m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
        else:
            # Case of C,T or I
            m_in = util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
            
        if pol_type_out == "T3":
            m_in = util_convert.T3_to_C3(m_in, sub_n_lig, sub_n_col + n_win_c, 0, 0)
        
        span_min, span_max = determination(lig, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
        
    if span_min < eps:
        span_min = eps
    
    # DATA PROCESSING
    for Np in range(n_polar_in):
        in_datafile[Np].seek(0)
        if flag_valid == 1:
            in_valid.seek(0)

    for Nb in range(nb_block):
        ligDone = 0
        if nb_block > 2:
            print(f"{100. * Nb / (nb_block - 1)}", end='\r', flush=True)

        if flag_valid == 1:
            m_in = util_block.read_block_matrix_float(in_valid, valid, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        if pol_type == "S2":
            m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
        else:
            # Case of C,T or I
            m_in = util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, Nb, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
            
        if pol_type_out == "T3":
            m_in = util_convert.T3_to_C3(m_in, sub_n_lig, sub_n_col + n_win_c, 0, 0)

        m_odd, m_dbl, m_vol = freeman(lig, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, span_min, span_max, m_odd, m_dbl, m_vol)

    util_block.write_block_matrix_float(out_odd, m_odd, sub_n_lig, sub_n_col, 0, 0, sub_n_col)
    util_block.write_block_matrix_float(out_dbl, m_dbl, sub_n_lig, sub_n_col, 0, 0, sub_n_col)
    util_block.write_block_matrix_float(out_vol, m_vol, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

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
    polar_type_replacements = {
        "SPP": "SPPC2"
    }

    if pol_type in polar_type_replacements:
        pol_type = polar_type_replacements[pol_type]
    return util.pol_type_config(pol_type)

# %% [codecell] configure_input_output_files
def configure_input_output_files(pol_type_in, in_dir, out_dir):
    """
    Configure the input and output files based on the provided polar types and directories.
    Return the input and output file names.
    """
    file_name_in = util.init_file_name(pol_type_in, in_dir)
    return file_name_in

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
def open_output_files(out_dir):
    """
    Open output files and return the file objects.
    """
    try:
        out_odd = open(os.path.join(out_dir, "Freeman_Odd.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, 'Freeman_Odd.bin')}")
        raise

    try:
        out_dbl = open(os.path.join(out_dir, "Freeman_Dbl.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, 'Freeman_Dbl.bin')}")
        raise

    try:
        out_vol = open(os.path.join(out_dir, "Freeman_Vol.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, 'Freeman_Vol.bin')}")
        raise

    return out_odd, out_dbl, out_vol

# %% [codecell] set_valid_pixels
def set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l):
    """
    Set the valid pixels for the freeman filter based on the provided parameters.
    """
    if not flag_valid:
        valid[:sub_n_lig + n_win_l, :sub_n_col + n_win_c] = 1.0
    return valid

# %% [codecell] allocate_matrices
def allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col):
    """
    Allocate matrices with given dimensions
    """
    vc_in = np.zeros(2 * n_col, dtype=np.float32)
    vf_in = np.zeros(n_col, dtype=np.float32)
    mc_in = np.zeros((4, 2 * n_col), dtype=np.float32)
    mf_in = np.zeros((n_polar_out, n_win_l, n_col + n_win_c), dtype=np.float32)

    valid = np.zeros((sub_n_lig + n_win_l, sub_n_col + n_win_c), dtype=np.float32)
    m_in = np.zeros((n_polar_out, sub_n_lig + n_win_l, n_col + n_win_c), dtype=np.float32)

    m_odd = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)
    m_dbl = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)
    m_vol = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)

    return vc_in, vf_in, mc_in, mf_in, valid, m_in, m_odd, m_dbl, m_vol

# %% [codecell] freeman
# @numba.njit(parallel=False)
def freeman(lig, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, span_min, span_max, m_odd, m_dbl, m_vol):
    """
    Perform freeman filtering on the input data, updating the output matrix with the filtered values.
    """
    CC11 = CC13_re = CC13_im = CC22 = CC33 = FV = FD = FS = ALP = BET = rtemp = 0.0
    m_avg = np.zeros((n_polar_out,sub_n_col))
    util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
    for col in range(sub_n_col):
        if valid[n_win_lm_1s2+lig][n_win_cm_1s2+col] == 1.:
            eps = 1.E-30
            CC11 = m_avg[util.C311][col]
            CC13_re = m_avg[util.C313_RE][col]
            CC13_im = m_avg[util.C313_IM][col]
            CC22 = m_avg[util.C322][col]
            CC33 = m_avg[util.C333][col]

            FV = 3. * CC22 / 2.
            CC11 = CC11 - FV
            CC33 = CC33 - FV
            CC13_re = CC13_re - FV / 3.

            if (CC11 <= eps) or (CC33 <= eps):
                FV = 3. * (CC11 + CC22 + CC33 + 2 * FV) / 8.
                FD = 0.
                FS = 0.
            else:
                if (CC13_re * CC13_re + CC13_im * CC13_im) > CC11 * CC33:
                    rtemp = CC13_re * CC13_re + CC13_im * CC13_im
                    CC13_re = CC13_re * np.sqrt(CC11 * CC33 / rtemp)
                    CC13_im = CC13_im * np.sqrt(CC11 * CC33 / rtemp)
                
                if CC13_re >= 0.:
                    ALP = -1.
                    FD = (CC11 * CC33 - CC13_re * CC13_re - CC13_im * CC13_im) / (CC11 + CC33 + 2 * CC13_re)
                    FS = CC33 - FD
                    BET = np.sqrt((FD + CC13_re) * (FD + CC13_re) + CC13_im * CC13_im) / FS
                else:
                    BET = 1.
                    FS = (CC11 * CC33 - CC13_re * CC13_re - CC13_im * CC13_im) / (CC11 + CC33 - 2 * CC13_re)
                    FD = CC33 - FS
                    ALP = np.sqrt((FS - CC13_re) * (FS - CC13_re) + CC13_im * CC13_im) / FD

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

    return m_odd, m_dbl, m_vol

# %% [codecell] is_pol_type_valid
def is_pol_type_valid(pol_type):
    """
    Check if the given pol_type is valid for processing.
    Returns True if the pol_type is valid, False otherwise.
    """
    valid_pol_types = ["S2", "SPP", "SPPpp1", "SPPpp2", "SPPpp3"]
    return pol_type in valid_pol_types

def copy_header(src_dir, dst_dir):
    src_path = os.path.join(src_dir, 'config.txt')
    dst_path = os.path.join(dst_dir, 'config.txt')

    if os.path.isfile(src_path):
        with open(src_path, 'r') as src_file:
            content = src_file.read()
        
        with open(dst_path, 'w') as dst_file:
            dst_file.write(content)
    else:
        print(f"Source file {src_path} does not exist.")

# @numba.njit(parallel=False)
def determination(lig, n_polar_out, sub_n_col, m_in, valid, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2):
    M_avg = np.zeros((n_polar_out,sub_n_col), dtype=float)

    M_avg = util_block.average_tci(m_in, valid, n_polar_out, M_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
    SpanMax = -np.inf
    SpanMin = np.inf
    for col in range(sub_n_col):
        if valid[n_win_lm_1s2+lig][n_win_cm_1s2+col] == 1.:
            Span = M_avg[util.C311][col]+M_avg[util.C322][col]+M_avg[util.C333][col]
            if Span >= SpanMax: 
                SpanMax = Span
            if Span <= SpanMin: 
                SpanMin = Span
    return SpanMin, SpanMax

if __name__ == "__main__":
    main("D:\\Satim\\PolSARPro\\Datasets\\T3\\", "D:\\Satim\\PolSARPro\\Datasets\\output\\", "T3", 7, 7, 0, 0, 18432, 1248, "", "")