"""
Polsarpro
===
Description :  Yamaguchi 4 components Decomposition

Input parameters:
id  	input directory
od  	output directory
iodf	input-output data format
mod 	decomposition mode (Y4O, Y4R, S4R)
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
import util
import util_block

def main(in_dir, out_dir, pol_type, yam_mode, n_win_l, n_win_c, off_lig, off_col, sub_n_lig, sub_n_col, file_memerr, file_valid):
    
    """
    Main Function for the yamaguchi_4_components.
    Parses the input arguments, reads the input files, and processes the data using yamaguchi_4_components filtering.
    """
    print("********************Welcome in yamaguchi_4_components_decomposition********************")

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
    out_odd, out_dbl, out_vol, out_hlx = open_output_files(out_dir)

    # /* COPY HEADER*/
    copy_header(in_dir, out_dir)

    # /* MATRIX ALLOCATION */
    util.vc_in, util.vf_in, util.mc_in, util.mf_in, valid, m_in, m_odd, m_dbl, m_vol, m_hlx = allocate_matrices(n_col, n_polar_out, n_win_l, n_win_c, sub_n_lig, sub_n_col)

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
            print(f"{100. * Nb / (nb_block - 1)}", end='\r', flush=True)
        if flag_valid == 1:
            m_in = util_block.read_block_matrix_float(in_valid, valid, Nb, nb_block, n_lig_block[Nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)

        if pol_type == "S2":
            m_in = util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, Nb, nb_block, n_lig_block[Nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
        else:
            # Case of C,T or I
            m_in = util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, Nb, nb_block, n_lig_block[Nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, n_col)
            
        if pol_type_out == "T3":
            m_in = util_block.T3_to_C3(m_in, n_lig_block[Nb], sub_n_col + n_win_c, 0, 0)
        
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_lig = {executor.submit(determination, lig): lig for lig in range(n_lig_block[Nb])}
            for future in tqdm(concurrent.futures.as_completed(future_to_lig), total=len(future_to_lig), desc="Processing"):
                try:
                    span_min, span_max = future.result()
                    if span_min < SpanMin_global:
                        SpanMin_global = span_min
                    if span_max > SpanMax_global:
                        SpanMax_global = span_max
                except Exception as exc:
                    print(f"An exception occurred: {exc}")

    if SpanMin_global < eps:
        SpanMin_global = eps
    
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
            util_block.read_block_matrix_float(in_valid, valid, Nb, nb_block, n_lig_block[Nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, Ncol)

        if pol_type == "S2":
            util_block.read_block_S2_noavg(in_datafile, m_in, pol_type_out, n_polar_out, Nb, nb_block, n_lig_block[Nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, Ncol)
        else:
            # Case of C,T or I
            util_block.read_block_TCI_noavg(in_datafile, m_in, n_polar_out, Nb, nb_block, n_lig_block[Nb], sub_n_col, n_win_l, n_win_c, off_lig, off_col, Ncol)
            
        if pol_type_out == "C3":
            util_block.c3_to_t3(m_in, n_lig_block[Nb], sub_n_col + n_win_c, 0, 0)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            n_win_lm_1s2, n_win_cm_1s2, M_in, Valid = 0
            M_out_futures = [executor.submit(yamaguchi_4_components, lig, lig, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, span_min, span_max, m_odd, m_dbl, m_vol, m_hlx, yam_mode, eps) for lig in range(n_lig_block[Nb])]
            for future in tqdm(concurrent.futures.as_completed(M_out_futures), total=len(M_out_futures), desc="Processing"):
                m_odd, m_dbl, m_vol, m_hlx = future.result()

        util_block.write_block_matrix_float(out_odd, m_odd, n_lig_block[Nb], sub_n_col, 0, 0, sub_n_col)
        util_block.write_block_matrix_float(out_dbl, m_dbl, n_lig_block[Nb], sub_n_col, 0, 0, sub_n_col)
        util_block.write_block_matrix_float(out_vol, m_vol, n_lig_block[Nb], sub_n_col, 0, 0, sub_n_col)
        util_block.write_block_matrix_float(out_hlx, m_hlx, n_lig_block[Nb], sub_n_col, 0, 0, sub_n_col)

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
        "S2": "S2T3"
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
def open_output_files(out_dir, yam_mode):
    """
    Open output files and return the file objects.
    """
    try:
        out_odd = open(os.path.join(out_dir, f"Yamaguchi4_{yam_mode}_Odd.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, f'Yamaguchi4_{yam_mode}_Odd.bin')}")
        raise

    try:
        out_dbl = open(os.path.join(out_dir, f"Yamaguchi4_{yam_mode}_Dbl.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, f'Yamaguchi4_{yam_mode}_Dbl.bin')}")
        raise

    try:
        out_vol = open(os.path.join(out_dir, f"Yamaguchi4_{yam_mode}_Vol.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, f'Yamaguchi4_{yam_mode}_Vol.bin')}")
        raise

    try:
        out_hlx = open(os.path.join(out_dir, f"Yamaguchi4_{yam_mode}_Hlx.bin"), "wb")
    except IOError:
        print(f"Could not open output file : {os.path.join(out_dir, f'Yamaguchi4_{yam_mode}_Hlx.bin')}")
        raise

    return out_odd, out_dbl, out_vol, out_hlx

# %% [codecell] set_valid_pixels
def set_valid_pixels(valid, flag_valid, sub_n_lig, sub_n_col, n_win_c, n_win_l):
    """
    Set the valid pixels for the yamaguchi_4_components filter based on the provided parameters.
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

    m_odd = np.zeros((sub_n_lig, sub_n_col), dtype=float)
    m_dbl = np.zeros((sub_n_lig, sub_n_col), dtype=float)
    m_vol = np.zeros((sub_n_lig, sub_n_col), dtype=float)
    m_hlx = np.zeros((sub_n_lig, sub_n_col), dtype=float)

    return vc_in, vf_in, mc_in, mf_in, valid, m_in, m_odd, m_dbl, m_vol, m_hlx

# %% [codecell] yamaguchi_4_components
@jit(nopython=True)
def yamaguchi_4_components(lig, n_polar_out, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2, m_in, valid, span_min, span_max, m_odd, m_dbl, m_vol, m_hlx, yam_mode, eps):
    """
    Perform yamaguchi_4_components filtering on the input data, updating the output matrix with the filtered values.
    """
    n_win_cm_1s2 = 0
    TT = np.zeros(n_polar_out)
    M_avg = np.zeros((n_polar_out,sub_n_col))
    util_block.average_TCI(m_in, valid, n_polar_out, M_avg, lig, sub_n_col, n_win_l, n_win_c, n_win_lm_1s2, n_win_cm_1s2)
    for col in range(sub_n_col):
        if valid[n_win_lm_1s2+lig][n_win_cm_1s2+col] == 1.:
            for Np in range(n_polar_out):
                TT[Np] = M_avg[Np][col]
            teta = 0.
            if yam_mode in ["Y4R", "S4R"]:
                teta = 0.5 * math.atan(2*TT[util.T323_RE]/(TT[util.T322]-TT[util.T333]))
                unitary_rotation(TT,teta)
            Pc = 2. * abs(TT[util.T323_IM])
            HV_type = 1 # Surface scattering
            if yam_mode == "S4R":
                C1 = TT[util.T311] - TT[util.T322] + (7./8.)*TT[util.T333] + (Pc/16.)
                if C1 > 0:
                    HV_type = 1 # Surface scattering
                else:
                    HV_type = 2 # Double bounce scattering
            if HV_type == 1:
                ratio = 10.*math.log10((TT[util.T311] + TT[util.T322]-2.*TT[util.T312_RE])/(TT[util.T311] + TT[util.T322]+2.*TT[util.T312_RE]))
                if -2. < ratio <= 2.:
                    Pv = 2.*(2.*TT[util.T333] - Pc)
                else:
                    Pv = (15./8.)*(2.*TT[util.T333] - Pc)
            if HV_type == 2:
                Pv = (15./16.)*(2.*TT[util.T333] - Pc)
            TP = TT[util.T311] + TT[util.T322] + TT[util.T333]
            if Pv < 0.:
                #Freeman - Yamaguchi 3-components algorithm*#
                HHHH = (TT[util.T311] + 2 * TT[util.T312_re] + TT[util.T322]) / 2
                HHVVre = (TT[util.T311] - TT[util.T322]) / 2
                HHVVim = -TT[util.T312_im]
                HVHV = TT[util.T333] / 2
                VVVV = (TT[util.T311] - 2 * TT[util.T312_re] + TT[util.T322]) / 2

                ratio = 10 * np.log10(VVVV/HHHH)
                if ratio <= -2:
                    FV = 15 * (HVHV / 4)
                    HHHH -= 8 * (FV/15)
                    VVVV -= 3 * (FV/15)
                    HHVVre -= 2 * (FV/15)
                elif ratio > 2:
                    FV = 15 * (HVHV / 4)
                    HHHH -= 3 * (FV/15)
                    VVVV -= 8 * (FV/15)
                    HHVVre -= 2 * (FV/15)
                elif -2 < ratio <= 2:
                    FV = 8 * (HVHV / 2)
                    HHHH -= 3 * (FV/8)
                    VVVV -= 3 * (FV/8)
                    HHVVre -= 1 * (FV/8)

                if HHHH <= eps or VVVV <= eps:
                    FD = 0
                    FS = 0
                    if -2 < ratio <= 2:
                        FV = (HHHH + 3 * (FV/8)) + HVHV + (VVVV + 3 * (FV/8))
                    elif ratio <= -2:
                        FV = (HHHH + 8 * (FV/15)) + HVHV + (VVVV + 3 * (FV/15))
                    elif ratio > 2:
                        FV = (HHHH + 3 * (FV/15)) + HVHV + (VVVV + 8 * (FV/15))

                else:
                    if (HHVVre ** 2 + HHVVim ** 2) > HHHH * VVVV:
                        rtemp = HHVVre ** 2 + HHVVim ** 2
                        HHVVre *= np.sqrt((HHHH * VVVV) / rtemp)
                        HHVVim *= np.sqrt((HHHH * VVVV )/ rtemp)

                    if HHVVre >= 0:
                        ALPre = -1
                        ALPim = 0
                        FD = (HHHH * VVVV - HHVVre ** 2 - HHVVim ** 2) / (HHHH + VVVV + 2 * HHVVre)
                        FS = VVVV - FD
                        BETre = (FD + HHVVre) / FS
                        BETim = HHVVim / FS
                    elif HHVVre < 0:
                        BETre = 1
                        BETim = 0
                        FS = (HHHH * VVVV - HHVVre ** 2 - HHVVim ** 2) / (HHHH + VVVV - 2 * HHVVre)
                        FD = VVVV - FS
                        ALPre = (HHVVre - FS) / FD
                        ALPim = HHVVim / FD

                m_odd[lig][col] = FS * (1 + BETre ** 2 + BETim ** 2)
                m_dbl[lig][col] = FD * (1 + ALPre ** 2 + ALPim ** 2)
                m_vol[lig][col] = FV
                m_hlx[lig][col] = 0

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
                # Yamaguchi 4-Components algorithm
                # Surface scattering
                if HV_type == 1:
                    S = TT['T311'] - (Pv / 2.)
                    D = TP - Pv - Pc - S
                    Cre = TT['T312_re'] + TT['T313_re']
                    Cim = TT['T312_im'] + TT['T313_im']
                    if ratio <= -2.: Cre -= (Pv / 6.)
                    if ratio > 2.: Cre += (Pv / 6.)

                    if (Pv + Pc) > TP:
                        Ps, Pd = 0., 0.
                        Pv = TP - Pc
                    else:
                        CO = 2.*TT['T311'] + Pc - TP
                        if CO > 0.:
                            Ps = S + (Cre*Cre + Cim*Cim)/S
                            Pd = D - (Cre*Cre + Cim*Cim)/S
                        else:
                            Pd = D + (Cre*Cre + Cim*Cim)/D
                            Ps = S - (Cre*Cre + Cim*Cim)/D

                    if Ps < 0.:
                        if Pd < 0.:
                            Ps, Pd = 0., 0.
                            Pv = TP - Pc
                        else:
                            Ps = 0.
                            Pd = TP - Pv - Pc
                    elif Pd < 0.:
                        Pd = 0.
                        Ps = TP - Pv - Pc

                # Double bounce scattering
                elif HV_type == 2:
                    S = TT['T311']
                    D = TP - Pv - Pc - S
                    Cre = TT['T312_re'] + TT['T313_re']
                    Cim = TT['T312_im'] + TT['T313_im']

                    Pd = D + (Cre*Cre + Cim*Cim)/D
                    Ps = S - (Cre*Cre + Cim*Cim)/D

                    if Ps < 0.:
                        if Pd < 0.:
                            Ps, Pd = 0., 0.
                            Pv = TP - Pc
                        else:
                            Ps = 0.
                            Pd = TP - Pv - Pc
                    elif Pd < 0.:
                        Pd = 0.
                        Ps = TP - Pv - Pc

                Ps = max(Ps, span_min)
                Pd = max(Pd, span_min)
                Pv = max(Pv, span_min)
                Pc = max(Pc, span_min)

                Ps = min(Ps, span_max)
                Pd = min(Pd, span_max)
                Pv = min(Pv, span_max)
                Pc = min(Pc, span_max)

                m_odd[lig][col] = Ps
                m_dbl[lig][col] = Pd
                m_vol[lig][col] = Pv
                m_hlx[lig][col] = Pc
        else:
            m_odd[lig][col] = 0.
            m_dbl[lig][col] = 0.
            m_vol[lig][col] = 0.
            m_hlx[lig][col] = 0.

    return m_odd, m_dbl, m_vol, m_hlx

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

def yamaguchi_4_components_algorithm(CC11, CC13_re, CC13_im, CC22, CC33, eps):
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

    return FV, FD, FS, ALP, BET

def unitary_rotation(TT, teta):
    T11 = TT[util.T311]
    T12_re, T12_im = TT[util.T312_RE], TT[util.T312_IM]
    T13_re, T13_im = TT[util.T313_RE], TT[util.T313_IM]
    T22 = TT[util.T322]
    T23_re, T23_im = TT[util.T323_RE], TT[util.T323_IM]
    T33 = TT[util.T333]

    TT[util.T311] = T11
    TT[util.T312_RE] = T12_re*np.cos(teta) + T13_re*np.sin(teta)
    TT[util.T312_IM] = T12_im*np.cos(teta) + T13_im*np.sin(teta)
    TT[util.T313_RE] = -T12_re*np.sin(teta) + T13_re*np.cos(teta)
    TT[util.T313_IM] = -T12_im*np.sin(teta) + T13_im*np.cos(teta)
    TT[util.T322] = T22*np.cos(teta)**2 + 2.*T23_re*np.cos(teta)*np.sin(teta) + T33*np.sin(teta)**2
    TT[util.T323_RE] = -T22*np.cos(teta)*np.sin(teta) + T23_re*np.cos(teta)**2 - T23_re*np.sin(teta)**2 + T33*np.cos(teta)*np.sin(teta)
    TT[util.T323_IM] = T23_im*np.cos(teta)**2 + T23_im*np.sin(teta)**2
    TT[util.T333] = T22*np.sin(teta)**2 + T33*np.cos(teta)**2 - 2.*T23_re*np.cos(teta)*np.sin(teta)

if __name__ == "__main__":
    main("D:\\Satim\\PolSARPro\\Datasets\\T3\\", "D:\\Satim\\PolSARPro\\Datasets\\output\\", "T3", 7, 7, 0, 0, 18432, 1248, "", "")

