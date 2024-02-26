"""
Polsarpro
===
BoxCar fully polarimetric speckle filter

Input parameters:
id = input directory
od = output directory
iodf = input-output data format
nwr = Nwin Row
nwc = Nwin Col
ofr = Offset Row
ofc = Offset Col
fnr = Final Number of Row
fnc = Final Number of Col

Optional Parameters:
maskp = Optional - mask file (valid pixels)
errfp = Optional - memory error file
datap = Optional - displays the help concerning Data Format parameter
"""
# %% [codecell] import
import os
import sys
import argparse
import numba
import numpy as np
from collections import namedtuple
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool
from numba import jit
import util
import util_block
from joblib import Parallel, delayed


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
                "nwr",
                "nwc",
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
            nwr=7,
            nwc=7,
            ofr=0,
            ofc=0,
            fnr=18432,
            fnc=1248,
            mask="",
            errf="",
        )
    return args

# %% [codecell] main
def main(in_dir, out_dir, pol_type, n_win_l, n_win_c, off_lig, off_col, sub_n_lig, sub_n_col, file_memerr, file_valid):
    """
    Main Function for the BoxCar fully polarimetric speckle filter.
    Parses the input arguments, reads the input files, and processes the data using boxcar filtering.
    """
    print("********************Welcome in boxcar_filter********************")

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
    valid = set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l)

    # /********************************************************************
    # ********************************************************************/
    # /* DATA PROCESSING */
    nb_block = 1
    for block in range(nb_block):
        if nb_block > 2:
            print("%f\r" % (100 * block / (nb_block - 1)), end="", flush = True)

        if is_pol_type_valid(pol_type_in):
            if pol_type_in == "S2":
                m_in = util_block.read_block_S2_noavg(
                    in_datafile,
                    m_in,
                    pol_type_out,
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
            else:
                m_in = util_block.read_block_SPP_noavg(
                    in_datafile,
                    m_in,
                    pol_type_out,
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
        else:
            #   /* Case of C,T or I */
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

        if flag_valid == True:
            valid = util_block.read_block_matrix_float(
                in_valid,
                valid,
                block,
                nb_block,
                n_lig_block[block],
                sub_n_col,
                n_win_l,
                n_win_c,
                off_lig,
                off_col,
                n_col,
            )

        m_out = boxcar_filter(n_win_c, sub_n_lig, sub_n_col, n_polar_out, m_out, m_in, valid, n_win_lm_1s2, n_win_cm_1s2)

        util_block.write_block_matrix3d_float(
            out_datafile, n_polar_out, m_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col
        )

@numba.njit(parallel=False)
def boxcar_filter(n_win_c, sub_n_lig, sub_n_col, n_polar_out, m_out, m_in, valid, n_win_lm_1s2, n_win_cm_1s2):
    lig_done = 0
    for lig in numba.prange(sub_n_lig):
        mean = np.zeros(n_polar_out)
        n_valid = 0.0
        lig_done += 1
        lig_prev = 0
        for col in range(sub_n_col):
            for n_pol in range(n_polar_out):
                m_out[n_pol][lig][col] = 0.0

            if col == 0:
                n_valid = 0.0
                for k in range(-n_win_lm_1s2, 1 + n_win_lm_1s2):
                    for l in range(-n_win_cm_1s2, 1 + n_win_cm_1s2):
                        for n_pol in range(n_polar_out):
                            mean[n_pol] = (
                                    mean[n_pol]
                                    + m_in[n_pol][n_win_lm_1s2 + lig + k][
                                        n_win_cm_1s2 + col + l
                                    ]
                                    * valid[n_win_lm_1s2 + lig + k][
                                        n_win_cm_1s2 + col + l
                                    ]
                                )
                        n_valid = (
                                n_valid
                                + valid[n_win_lm_1s2 + lig + k][n_win_cm_1s2 + col + l]
                            )
            else:
                for k in range(-n_win_lm_1s2, 1 + n_win_lm_1s2):
                    idx_y = n_win_lm_1s2 + lig + k
                    for n_pol in range(n_polar_out):
                        mean[n_pol] = (
                                mean[n_pol]
                                - m_in[n_pol][idx_y][col - 1] * valid[idx_y][col - 1]
                            )
                        mean[n_pol] = (
                                mean[n_pol]
                                + m_in[n_pol][idx_y][n_win_c - 1 + col]
                                * valid[idx_y][n_win_c - 1 + col]
                            )
                    n_valid = (
                            n_valid
                            - valid[idx_y][col - 1]
                            + valid[idx_y][n_win_c - 1 + col]
                        )
            if n_valid != 0:
                for n_pol in range(n_polar_out):
                    if lig_prev == lig:
                        m_out[n_pol][lig][col] = mean[n_pol] / n_valid
                    else:
                        m_out[n_pol][lig][col] = mean[n_pol] / n_valid
                        lig_prev = lig
    return m_out
        #git

        # parallel_execution(n_lig_block, block, n_polar_out, sub_n_col, m_out, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, n_win_c)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     M_out_futures = [executor.submit(boxcar, lig, n_polar_out, sub_n_col, n_win_lm_1s2, n_win_cm_1s2, n_win_c, m_in, valid, m_out) for lig in range(sub_n_lig)]
        #     for future in tqdm(concurrent.futures.as_completed(M_out_futures), total=len(M_out_futures), desc="Processing"):
        #         pass

        # util_block.write_block_matrix3d_float(out_datafile, n_polar_out, m_out, sub_n_lig, sub_n_col, 0, 0, sub_n_col)



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
    Set the valid pixels for the boxcar filter based on the provided parameters.
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

# %% [codecell] boxcar
@jit(nopython=True)
def boxcar(lig, n_polar_out, sub_n_col, n_win_lm_1s2, n_win_cm_1s2, n_win_c, m_in, valid, m_out):
    """
    Perform boxcar filtering on the input data, updating the output matrix with the filtered values.
    """
    mean = np.zeros(n_polar_out)
    n_valid = 0.0

    for col in range(sub_n_col):
        for n_pol in range(n_polar_out):
            m_out[n_pol][lig][col] = 0.0

        if col == 0:
            n_valid = 0.0
            for k in range(-n_win_lm_1s2, 1 + n_win_lm_1s2):
                for l in range(-n_win_cm_1s2, 1 + n_win_cm_1s2):
                    for n_pol in range(n_polar_out):
                        mean[n_pol] = (
                            mean[n_pol]
                            + m_in[n_pol][n_win_lm_1s2 + lig + k][
                                n_win_cm_1s2 + col + l
                            ]
                            * valid[n_win_lm_1s2 + lig + k][
                                n_win_cm_1s2 + col + l
                            ]
                        )
                    n_valid = (
                        n_valid
                        + valid[n_win_lm_1s2 + lig + k][n_win_cm_1s2 + col + l]
                    )
        else:
            for k in range(-n_win_lm_1s2, 1 + n_win_lm_1s2):
                idx_y = n_win_lm_1s2 + lig + k
                for n_pol in range(n_polar_out):
                    mean[n_pol] = (
                        mean[n_pol]
                        - m_in[n_pol][idx_y][col - 1] * valid[idx_y][col - 1]
                    )
                    mean[n_pol] = (
                        mean[n_pol]
                        + m_in[n_pol][idx_y][n_win_c - 1 + col]
                        * valid[idx_y][n_win_c - 1 + col]
                    )
                n_valid = (
                    n_valid
                    - valid[idx_y][col - 1]
                    + valid[idx_y][n_win_c - 1 + col]
                )
        if n_valid != 0:
            for n_pol in range(n_polar_out):
                m_out[n_pol][lig][col] = mean[n_pol] / n_valid
    return m_out

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


def process_lig(lig, NligBlock, Nb, NpolarOut, Sub_Ncol, M_out, NwinLM1S2, NwinCM1S2, M_in, Valid, NwinC):
    mean = [0.] * NpolarOut
    Nvalid = 0.

    for col in range(Sub_Ncol):
        for Np in range(NpolarOut):
            M_out[Np][lig][col] = 0.

        if col == 0:
            Nvalid = 0.
            for k in range(-NwinLM1S2, 1 + NwinLM1S2):
                for l in range(-NwinCM1S2, 1 + NwinCM1S2):
                    for Np in range(NpolarOut):
                        mean[Np] += M_in[Np][NwinLM1S2+lig+k][NwinCM1S2+col+l] * Valid[NwinLM1S2+lig+k][NwinCM1S2+col+l]
                    Nvalid += Valid[NwinLM1S2+lig+k][NwinCM1S2+col+l]
        else:
            for k in range(-NwinLM1S2, 1 + NwinLM1S2):
                idxY = NwinLM1S2 + lig + k
                for Np in range(NpolarOut):
                    mean[Np] -= M_in[Np][idxY][col-1] * Valid[idxY][col-1]
                    mean[Np] += M_in[Np][idxY][NwinC-1+col] * Valid[idxY][NwinC-1+col]
                Nvalid = Nvalid - Valid[idxY][col-1] + Valid[idxY][NwinC-1+col]

        if Nvalid != 0.:
            for Np in range(NpolarOut):
                M_out[Np][lig][col] = mean[Np] / Nvalid

    del mean
    return M_out

# Parallel execution using joblib
def parallel_execution(NligBlock, block, NpolarOut, Sub_Ncol, M_out, NwinLM1S2, NwinCM1S2, M_in, Valid, NwinC):
    ligDone = [0]
    results = Parallel(n_jobs=-1, backend='threading')(delayed(process_lig)(lig, NligBlock, block, NpolarOut, Sub_Ncol, M_out, NwinLM1S2, NwinCM1S2, M_in, Valid, NwinC) for lig in range(NligBlock[block]))
    
    # Update M_out with the results
    for i, result in enumerate(results):
        for j in range(NpolarOut):
            for k in range(len(result[j])):
                for l in range(len(result[j][k])):
                    M_out[j][k][l] = result[j][k][l]
        ligDone[0] += 1
        if i == 0:
            print(f"{ligDone[0]}/{NligBlock[block]}")
    
    return M_out

if __name__ == "__main__":
    main("D:\\Satim\\PolSARPro\\Datasets\\T3\\", "D:\\Satim\\PolSARPro\\Datasets\\output\\", "T3", 7, 7, 0, 0, 18432, 1248, "", "")