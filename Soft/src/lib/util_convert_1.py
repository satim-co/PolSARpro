"""
Polsarpro
===
util_block
"""

# %% [codecell] import
from concurrent.futures import ThreadPoolExecutor
import math
import os
import sys
import struct

import numpy as np
import util

def S2_to_T6(S_in1, S_in2, M_in, Nlig, Ncol, NwinLig, NwinCol):
    hh = 0
    hv = 1
    vh = 2
    vv = 3

    with ThreadPoolExecutor() as executor:
        executor.map(process_s2_to_t6, range(Nlig + NwinLig), 
                     [Ncol]*Nlig, [NwinCol]*Nlig, 
                     [hh]*Nlig, [hv]*Nlig, [vh]*Nlig, [vv]*Nlig, 
                     [S_in1]*Nlig, [S_in2]*Nlig, [M_in]*Nlig)

    return 1

def process_s2_to_t6(lig, Ncol, NwinCol, hh, hv, vh, vv, S_in1, S_in2, M_in):
    for col in range(Ncol + NwinCol):
        k1r = (S_in1[hh][lig][2*col] + S_in1[vv][lig][2*col]) / math.sqrt(2.)
        k1i = (S_in1[hh][lig][2*col+1] + S_in1[vv][lig][2*col+1]) / math.sqrt(2.)
        k2r = (S_in1[hh][lig][2*col] - S_in1[vv][lig][2*col]) / math.sqrt(2.)
        k2i = (S_in1[hh][lig][2*col+1] - S_in1[vv][lig][2*col+1]) / math.sqrt(2.)
        k3r = (S_in1[hv][lig][2*col] + S_in1[vh][lig][2*col]) / math.sqrt(2.)
        k3i = (S_in1[hv][lig][2*col+1] + S_in1[vh][lig][2*col+1]) / math.sqrt(2.)

        k4r = (S_in2[hh][lig][2*col] + S_in2[vv][lig][2*col]) / math.sqrt(2.)
        k4i = (S_in2[hh][lig][2*col+1] + S_in2[vv][lig][2*col+1]) / math.sqrt(2.)
        k5r = (S_in2[hh][lig][2*col] - S_in2[vv][lig][2*col]) / math.sqrt(2.)
        k5i = (S_in2[hh][lig][2*col+1] - S_in2[vv][lig][2*col+1]) / math.sqrt(2.)
        k6r = (S_in2[hv][lig][2*col] + S_in2[vh][lig][2*col]) / math.sqrt(2.)
        k6i = (S_in2[hv][lig][2*col+1] + S_in2[vh][lig][2*col+1]) / math.sqrt(2.)

        M_in[0][lig][col] = k1r * k1r + k1i * k1i

def T3_to_C3(M_in, Nlig, Ncol, NwinLig, NwinCol):
    for lig in range(Nlig + NwinLig):
        for col in range(Ncol + NwinCol):
            T11 = M_in[util.T311, lig, col]
            T12_re = M_in[util.T312_RE, lig, col]
            T12_im = M_in[util.T312_IM, lig, col]
            T13_re = M_in[util.T313_RE, lig, col]
            T13_im = M_in[util.T313_IM, lig, col]
            T22 = M_in[util.T322, lig, col]
            T23_re = M_in[util.T323_RE, lig, col]
            T23_im = M_in[util.T323_IM, lig, col]
            T33 = M_in[util.T333, lig, col]

            M_in[util.C311, lig, col] = (T11 + 2 * T12_re + T22) / 2
            M_in[util.C312_RE, lig, col] = (T13_re + T23_re) / np.sqrt(2)
            M_in[util.C312_IM, lig, col] = (T13_im + T23_im) / np.sqrt(2)
            M_in[util.C313_RE, lig, col] = (T11 - T22) / 2
            M_in[util.C313_IM, lig, col] = -T12_im
            M_in[util.C322, lig, col] = T33
            M_in[util.C323_RE, lig, col] = (T13_re - T23_re) / np.sqrt(2)
            M_in[util.C323_IM, lig, col] = (-T13_im + T23_im) / np.sqrt(2)
            M_in[util.C333, lig, col] = (T11 - 2 * T12_re + T22) / 2

    return M_in
