"""
Polsarpro
===
J.S. LEE refined fully polarimetric speckle filter

Input paraeters:

id  	input directory
od  	output directory
iodf	input-output data format
nw  	Nwin Row and Col
nlk 	Nlook
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

# %% [codecell] import
import argparse
from collections import namedtuple
import concurrent.futures
import math
import os

import numpy as np
import tqdm
from numba import jit
import numba

import util
import util_block

eps = 1.E-30

# %% [codecell] main
def main(in_dir, out_dir, pol_type, n_win, n_look, off_lig, off_col, sub_n_lig, sub_n_col, file_memerr, file_valid):
    """
    Main Function for the J. S. Lee refined fully polarimetric speckle filter.
    Parses the input arguments, reads the input files, and processes the data using J. S. Lee filtering.
    """
    print("********************Welcome in lee_refinec_filter********************")

    pol_type_conf = [
        "S2C3",
        "S2C4",
        "S2T3",
        "S2T4",
        "C2",
        "C3",
        "C4",
        "T2",
        "T3",
        "T4",
        "SPP",
        "IPP",
    ]

    # Initialising the arguments for the parser
    n_lig_block = np.zeros(8192, dtype=int)
    n_polar_out = 0
    m_out = 0
    m_in = []
    valid = 0
    in_datafile = []
    in_valid = 0
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

    n_win_m1s2 = int((n_win - 1) / 2)
    n_win_l = n_win
    n_win_c = n_win

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
    util.vc_in, util.vf_in, util.mc_in, util.mf_in, valid, m_in, m_out, mask, span, coeff, n_max = allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col, n_win)

    # /* MASK VALID PIXELS (if there is no MaskFile */
    valid = set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l)

    # Speckle variance given by the input data number of looks
    sigma2 = 1.0 / float(n_look)

    # /* Gradient window calculation parameters */
    nn_win, deplct = gradient_window_cal_params(n_win, window_parameters)

    # /* Create Mask */
    mask = make_mask(n_win)
    
    nb_block = 1
    for block in range(nb_block):
        ligDone = 0
        if nb_block > 2:
            print("{:.2f}\r".format(100.0 * block / (nb_block - 1)), end='', flush=True)

        if (pol_type_in == "S2" or pol_type_in == "SPP" or pol_type_in == "SPPpp1"
                or pol_type_in == "SPPpp2" or pol_type_in == "SPPpp3"):

            if pol_type_in == "S2":
                m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, block, nb_block,
                                    n_lig_block[block], sub_n_col, n_win, n_win, off_lig, off_col, n_col)
            else:
                m_in = util_block.read_block_SPP_noavg(in_datafile, m_in, pol_type_out, n_polar_out, block, nb_block,
                                        n_lig_block[block], sub_n_col, n_win, n_win,off_lig, off_col, n_col)
        else:
            # Case of C, T, or I
            m_in = util_block.read_block_TCI_noavg(
                in_datafile,
                m_in,
                n_polar_out,
                block,
                nb_block,
                sub_n_lig,
                sub_n_col,
                n_win_l,
                n_win_c,
                off_lig,
                off_col,
                n_col,
            )

        if flag_valid == 1:
            m_in = util_block.read_block_matrix_float(in_valid, valid, block, nb_block, n_lig_block[block], sub_n_col,
                                    n_win, n_win, off_lig, off_col, n_col)
        # Span Determination
        span = span_determination(sub_n_lig, sub_n_col, n_win, pol_type_out, span, m_in)

        coeff, n_max = make_coeff(sigma2, deplct, nn_win, n_win_m1s2, sub_n_lig, sub_n_col, span, mask, n_max, coeff)

        m_out = lee_refined(sub_n_lig, sub_n_col, n_polar_out, m_out, n_win_m1s2, valid, n_max, mask, coeff, m_in)

        util_block.write_block_matrix3d_float(out_datafile, n_polar_out, m_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)

# %% [codecell] make_coeff
@numba.njit(parallel=False)
def make_coeff(sigma2, deplct, nnwin, nwin_m1_s2, sub_n_lig, sub_ncol, span, mask, nmax, coeff):
    # Internal variables
    subwin = [[0.0] * 3 for _ in range(3)]
    Dist = [0.0] * 4
    MaxDist = 0.0
    Npoints = 0.0

    # FILTERING
    for lig in numba.prange(sub_n_lig):
        for col in range(sub_ncol):
            # 3*3 average SPAN Sub_window calculation for directional gradient determination
            for k in range(3):
                for l in range(3):
                    subwin[k][l] = 0.0
                    for kk in range(nnwin):
                        for ll in range(nnwin):
                            subwin[k][l] += span[k * deplct + kk + lig][l * deplct + ll + col] / (nnwin * nnwin)
            # Directional gradient computation
            Dist[0] = -subwin[0][0] + subwin[0][2] - subwin[1][0] + subwin[1][2] - subwin[2][0] + subwin[2][2]
            Dist[1] =  subwin[0][1] + subwin[0][2] - subwin[1][0] + subwin[1][2] - subwin[2][0] - subwin[2][1]
            Dist[2] =  subwin[0][0] + subwin[0][1] + subwin[0][2] - subwin[2][0] - subwin[2][1] - subwin[2][2]
            Dist[3] =  subwin[0][0] + subwin[0][1] + subwin[1][0] - subwin[1][2] - subwin[2][1] - subwin[2][2]

            # Choice of a directional mask according to the maximum gradient
            MaxDist = np.NINF
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

            for k in range(-nwin_m1_s2, 1 + nwin_m1_s2):
                for l in range(-nwin_m1_s2, 1 + nwin_m1_s2):
                    if mask[nmax[lig][col]][nwin_m1_s2 + k][nwin_m1_s2 + l] == 1:
                        m_span += span[nwin_m1_s2 + k + lig][nwin_m1_s2 + l + col]
                        m_span2 += span[nwin_m1_s2 + k + lig][nwin_m1_s2 + l + col] * span[nwin_m1_s2 + k + lig][nwin_m1_s2 + l + col]
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
    return coeff, nmax

# %% [codecell] parse_arguments
def parse_arguments(pol_type_conf):
    """
    Parse command line arguments and return them as an 'args' object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type=str, required=True, help="input directory")
    parser.add_argument("-od", type=str, required=True, help="output directory")
    parser.add_argument("-iodf", type=str, required=True, choices=pol_type_conf, help="")
    parser.add_argument("-nw", type=int, required=True, help="Nwin Row and Col")
    parser.add_argument("-nlk", type=int, required=True, help="Nlook")
    parser.add_argument("-ofr", type=int, required=True, help="Offset Row")
    parser.add_argument("-ofc", type=int, required=True, help="Offset Col")
    parser.add_argument("-fnr", type=int, required=True, help="Final Number of Row")
    parser.add_argument("-fnc", type=int, required=True, help="Final Number of Col")

    parser.add_argument(
        "-mask", type=int, required=False, help="Optional - mask file (valid pixels)"
    )
    parser.add_argument(
        "-errf", type=int, required=False, help="Optional - memory error file"
    )
    parser.add_argument(
        "-data",
        type=int,
        required=False,
        help="Optional - displays the help concerning Data Format parameter\pri",
    )
    if not __debug__:
        args = parser.parse_args()
    else:
        Arg = namedtuple(
            "Arg",
            [
                "id",
                "od",
                "iodf",
                "nw",
                "nlk",
                "ofr",
                "ofc",
                "fnr",
                "fnc",
                "mask",
                "errf",
            ],
        )
        args = Arg(
            id="D:\\Satim\\PolSARPro\\Datasets\\T3\\",
            od="D:\\Satim\\PolSARPro\\Datasets\\output\\",
            iodf="T3",
            nw=7,
            nlk=7,
            ofr=0,
            ofc=0,
            fnr=18432,
            fnc=1248,
            mask="",
            errf="",
        )
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
    polar_type_replacements = {
        "SPP": "SPPC2"
    }

    if pol_type in polar_type_replacements:
        pol_type = polar_type_replacements[pol_type]
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
def open_input_files(file_name_in, file_valid, in_datafile, pol_type_in, in_valid):
    """
    Open input files and return the file objects and a flag indicating if the 'valid' file is present.
    """
    flag_valid = False
    for n_pol in range(pol_type_in):
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

# %% [codecell] allocate_matrices
def allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col, n_win):
    """
    Allocate matrices with given dimensions
    """
    vc_in = np.zeros(2 * n_col, dtype=np.float32)
    vf_in = np.zeros(n_col, dtype=np.float32)
    mc_in = np.zeros((4, 2 * n_col), dtype=np.float32)
    mf_in = np.zeros((n_polar_out, n_win_l, n_col + n_win_c), dtype=np.float32)

    valid = np.zeros((sub_n_lig + n_win_l, sub_n_col + n_win_c), dtype=np.float32)

    mask = np.zeros((8, n_win, n_win), dtype=np.float32)
    span = np.zeros((sub_n_lig + n_win, n_col + n_win), dtype=np.float32)
    coeff = np.zeros((sub_n_lig, sub_n_col), dtype=np.float32)
    n_max = np.zeros((sub_n_lig + n_win, n_col + n_win), dtype=int)

    m_in = np.zeros((n_polar_out, sub_n_lig + n_win_l, n_col + n_win_c), dtype=np.float32)
    m_out = np.zeros((n_polar_out, sub_n_lig, sub_n_col), dtype=np.float32)

    return vc_in, vf_in, mc_in, mf_in, valid, m_in, m_out, mask, span, coeff, n_max

# %% [codecell] set_valid_pixels
def set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l):
    """
    Set the valid pixels for the boxcar filter based on the provided parameters.
    """
    if not flag_valid:
        valid[:sub_n_lig + n_win_l, :sub_n_col + n_win_c] = 1.0
    return valid

# %% [codecell] gradient_window_cal_params
def gradient_window_cal_params(n_win, window_parameters):
    if n_win in window_parameters:
        nn_win, deplct = window_parameters[n_win]
        return nn_win, deplct
    else:
        raise ValueError("The window width Nwin must be set to 3 to 31")

# %% [codecell] make_mask
def make_mask(n_win):
    mask = np.zeros((8, n_win, n_win), dtype=float)

    mask[0, :, (n_win - 1) // 2:] = 1
    mask[4, :, :1 + (n_win - 1) // 2] = 1

    mask[1, np.triu_indices(n_win)] = 1
    mask[5, np.tril_indices(n_win)] = 1

    mask[2, :1 + (n_win - 1) // 2, :] = 1
    mask[6, (n_win - 1) // 2:, :] = 1

    mask[3, np.tril_indices(n_win, k=-1)] = 1
    mask[7, np.triu_indices(n_win, k=1)] = 1

    return mask

# %% [codecell] span_determination
@numba.njit(parallel=False)
def span_determination(sub_n_lig, sub_n_col, n_win, pol_type_out, span, m_in):
    for lig in numba.prange(sub_n_lig + n_win):
        for col in range(sub_n_col + n_win):
            if pol_type_out in ["C2", "C2pp1", "C2pp2", "C2pp3", "T2", "T2pp1", "T2pp2", "T2pp3"]:
                span[lig][col] = m_in[0][lig][col] + m_in[3][lig][col]
            elif pol_type_out in ["C3", "T3"]:
                span[lig][col] = m_in[0][lig][col] + m_in[5][lig][col] + m_in[8][lig][col]
            elif pol_type_out in ["C4", "T4"]:
                span[lig][col] = m_in[0][lig][col] + m_in[7][lig][col] + m_in[12][lig][col] + m_in[15][lig][col]
    return span

# %% [codecell] filtering
@numba.njit(parallel=False)
def lee_refined(sub_n_lig, sub_n_col, n_polar_out, m_out, n_win_m1s2, valid, n_max, mask, coeff, m_in):
    ligDone = 0
    # for lig in range(sub_n_lig):
    #     ligDone += 1
    #     for col in range(sub_n_col):
    #         for Np in range(n_polar_out):
    #             m_out[Np][lig][col] = 0.0

    #         if valid[n_win_m1s2+lig][n_win_m1s2+col] == 1.0:
    #             for Np in range(n_polar_out):
    #                 mean = 0.0
    #                 Npoints = 0.0

    #                 for k in range(-n_win_m1s2, 1 + n_win_m1s2):
    #                     for l in range(-n_win_m1s2, 1 + n_win_m1s2):
    #                         if mask[n_max[lig][col]][n_win_m1s2 + k][n_win_m1s2 + l] == 1:
    #                             mean += m_in[Np][n_win_m1s2+lig+k][n_win_m1s2+col+l]
    #                             Npoints += 1.0
                    
    #                 if Npoints != 0:
    #                     mean /= Npoints
                    
    #                 # Filtering f(x)=E(x)+k*(x-E(x))
    #                 a = mean + coeff[lig][col] * (m_in[Np][n_win_m1s2+lig][n_win_m1s2+col] - mean)
    #                 m_out[Np][lig][col] = mean + coeff[lig][col] * (m_in[Np][n_win_m1s2+lig][n_win_m1s2+col] - mean)
    # return m_out
    for lig in numba.prange(sub_n_lig):
        ligDone += 1

        for col in range(sub_n_col):
            for Np in range(n_polar_out):
                m_out[Np][lig][col] = 0.0

            if valid[n_win_m1s2 + lig][n_win_m1s2 + col] == 1.0:
                for Np in range(n_polar_out):
                    mean = 0.0
                    Npoints = 0.0

                    for k in range(-n_win_m1s2, 1 + n_win_m1s2):
                        for l in range(-n_win_m1s2, 1 + n_win_m1s2):
                            if mask[n_max[lig][col]][n_win_m1s2 + k][n_win_m1s2 + l] == 1:
                                mean += m_in[Np][n_win_m1s2 + lig + k][n_win_m1s2 + col + l]
                                Npoints += 1.0

                    mean /= Npoints

                    # Filtering f(x)=E(x)+k*(x-E(x))
                    a = (m_in[Np][n_win_m1s2 + lig][n_win_m1s2 + col] - mean)
                    b = coeff[lig][col]
                    c = mean
                    m_out[Np][lig][col] = mean + coeff[lig][col] * (m_in[Np][n_win_m1s2 + lig][n_win_m1s2 + col] - mean)
    return m_out

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