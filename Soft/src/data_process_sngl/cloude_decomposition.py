"""
Polsarpro
===
Description :  Cloude decomposition

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
import processing
import numba

def main(in_dir, out_dir, pol_type, n_win_l, n_win_c, off_lig, off_col, sub_n_lig, sub_n_col, file_memerr, file_valid):
    """
    Main Function for the Cloude.
    Parses the input arguments, reads the input files, and processes the data using Cloude filtering.
    """
    print("********************Welcome in Cloude********************")

    pol_type_conf = ["S2C3", "S2T3", "C3", "T3"]

    # Initialising the arguments for the parser
    n_lig_block = np.zeros(8192, dtype=int)
    n_polar_out = 0
    m_out = 0
    m_in = []
    valid = 0
    in_datafile = []
    in_valid = 0

    m = np.zeros((3, 3, 2))

    #   ********************************************************************
    #   ********************************************************************/
    #   /* USAGE */
    # args = parse_arguments(pol_type_conf)
    # in_dir = args.id
    # out_dir = args.od
    # pol_type = args.iodf
    # n_win_l = args.nwr
    # n_win_c = args.nwc
    # off_lig = args.ofr
    # off_col = args.ofc
    # sub_n_lig = args.fnr
    # sub_n_col = args.fnc
    # file_memerr = args.errf
    # file_valid = args.mask

    n_win_lm_1s2 = int((n_win_l - 1) / 2)
    n_win_cm_1s2 = int((n_win_c - 1) / 2)

    # # /* INPUT/OUPUT CONFIGURATIONS */
    n_lig, n_col, polar_case, polar_type = read_configuration(in_dir)

    # # /* POLAR TYPE CONFIGURATION */
    pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out = configure_polar_type(pol_type)

    # # /* INPUT/OUTPUT FILE CONFIGURATION */
    file_name_in, file_name_out = configure_input_output_files(pol_type_in, pol_type_out, in_dir, out_dir)

    # # /* INPUT FILE OPENING*/
    in_datafile, in_valid, flag_valid = open_input_files(file_name_in, file_valid, in_datafile, n_polar_in, in_valid)

    # /* OUTPUT FILE OPENING*/
    out_datafile = open_output_files(file_name_out, n_polar_out)

    # /* COPY HEADER*/
    copy_header(in_dir, out_dir)

    # /* MATRIX ALLOCATION */
    util.vc_in, util.vf_in, util.mc_in, util.mf_in, valid, m_in, m_out = allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col)

    # /* MASK VALID PIXELS (if there is no MaskFile */
    valid = set_valid_pixels(valid, flag_valid, n_lig_block, sub_n_col, n_win_c, n_win_l)

    nb_block = 1
    for block in range(nb_block):
        ligDone = 0
        if nb_block > 2:
            print(f"{100. * block / (nb_block - 1):.2f}", end='\r', flush=True)

        if flag_valid == 1:
            m_in = util_block.read_block_matrix_float(in_valid, valid, block, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        if pol_type_in == "S2":
            m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, block, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
        else:
            # Case of C,T or I
            m_in = util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, block, nb_block, sub_n_lig, sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        m_out = cloude(n_polar_out,sub_n_lig, sub_n_col, n_win_lm_1s2, n_win_cm_1s2, n_win_c, m_in, valid, m_out, m, n_win_l)

        util_block.write_block_matrix3d_float(out_datafile, n_polar_out, m_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)


def cloude(n_polar_out,sub_n_lig, sub_n_col, n_win_lm_1s2, n_win_cm_1s2, n_win_c, m_in, valid, m_out, m, n_win_l):
    ligDone = 0
    for lig in range(sub_n_lig):
        if lig % ((sub_n_lig) // 20) == 0:
            print("{:.2f}%\r".format(100.0 * lig / (sub_n_lig - 1)), end="",)
        ligDone += 1
        k1r, k1i, k2r, k2i, k3r, k3i = 0, 0, 0, 0, 0, 0
        eps = np.float64(1e-30)
        m = np.zeros((3, 3, 2), dtype=float)
        v = np.zeros((3, 3, 2), dtype=float)
        lambda_ = np.zeros(3, dtype=float)
        m_avg = np.zeros((n_polar_out,sub_n_col))
        m_avg = util_block.average_tci(m_in, valid, n_polar_out, m_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
        for col in range(sub_n_col):
            if valid[n_win_lm_1s2 + lig][n_win_cm_1s2 + col] == 1.:
                m[0][0][0] = eps + m_avg[0][col]
                m[0][0][1] = 0.
                m[0][1][0] = eps + m_avg[1][col]
                m[0][1][1] = eps + m_avg[2][col]
                m[0][2][0] = eps + m_avg[3][col]
                m[0][2][1] = eps + m_avg[4][col]
                m[1][0][0] =  m[0][1][0]
                m[1][0][1] = -m[0][1][1]
                m[1][1][0] = eps + m_avg[5][col]
                m[1][1][1] = 0.
                m[1][2][0] = eps + m_avg[6][col]
                m[1][2][1] = eps + m_avg[7][col]
                m[2][0][0] =  m[0][2][0]
                m[2][0][1] = -m[0][2][1]
                m[2][1][0] =  m[1][2][0]
                m[2][1][1] = -m[1][2][1]
                m[2][2][0] = eps + m_avg[8][col]
                m[2][2][1] = 0.

                #EIGENVECTOR/EIGENVALUE DECOMPOSITION
                #V complex eigenvecor matrix, lambda real vector
                v, lambda_ = processing.diagonalisation(3, m, v, lambda_)
                for k in range(3):
                    if lambda_[k] < 0.:
                        lambda_[k] = 0.

                #Cloude algorithm
                k1r = np.sqrt(lambda_[0]) * v[0][0][0]
                k1i = np.sqrt(lambda_[0]) * v[0][0][1]
                k2r = np.sqrt(lambda_[0]) * v[1][0][0]
                k2i = np.sqrt(lambda_[0]) * v[1][0][1]
                k3r = np.sqrt(lambda_[0]) * v[2][0][0]
                k3i = np.sqrt(lambda_[0]) * v[2][0][1]

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
                for Np in range(n_polar_out):
                    m_out[Np][lig][col] = 0.
    return m_out

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
def configure_input_output_files(pol_type_in, pol_type_out, in_dir, out_dir):
    """
    Configure the input and output files based on the provided polar types and directories.
    Return the input and output file names.
    """
    file_name_in = util.init_file_name(pol_type_in, in_dir)
    file_name_out = util.init_file_name(pol_type_out, out_dir)
    return file_name_in, file_name_out

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
def open_output_files(file_name_out, n_polar_out):
    """
    Open output files and return the file objects.
    """
    out_datafile = []
    for n_pol in range(n_polar_out):
        try:
            out_datafile.append(open(file_name_out[n_pol], "wb"))
        except IOError:
            print("Could not open output file : ", file_name_out[n_pol])
            raise
    return out_datafile

# %% [codecell] set_valid_pixels
def set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l):
    """
    Set the valid pixels for the Cloude filter based on the provided parameters.
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
    m_out = np.zeros((n_polar_out, sub_n_lig, sub_n_col), dtype=np.float32)

    return vc_in, vf_in, mc_in, mf_in, valid, m_in, m_out

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

if __name__ == "__main__":
    main("D:\\Satim\\PolSARPro\\Datasets\\T3\\", "D:\\Satim\\PolSARPro\\Datasets\\output\\", "T3", 7, 7, 0, 0, 18432, 1248, "", "")