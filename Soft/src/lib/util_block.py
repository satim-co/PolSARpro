"""
Polsarpro
===
util_block
"""

# %% [codecell] import
import math
import os
import sys
import struct
import numpy as np
import logging
import numba
from . import util
from . import matrix


# %% [codecell] read_block_S2_noavg
def read_block_S2_noavg(
    datafile,
    m_out,
    pol_type,
    nn_polar,
    nn_block,
    nn_bblock,
    sub_nnlig,
    sub_nncol,
    nn_win_lig,
    nn_win_col,
    ooff_lig,
    ooff_col,
    nn_col,
):
    """Read S2 Sinclair matrix without applying a spatial
    averaging"""
    nn_polar_in = 4
    hh, hv, vh, vv = 0, 1, 2, 3
    nn_win_lig_m1s2 = (nn_win_lig - 1) // 2
    nn_win_col_m1_s2 = (nn_win_col - 1) // 2
    nn_win_lig_m1 = nn_win_lig - 1

    if pol_type in ["S2", "SPPpp1", "SPPpp2", "SPPpp3"]:
        if nn_block == 0:
            # OFFSET LINES READING
            for lig in range(ooff_lig):
                for n in range(nn_polar_in):
                    util.mc_in[n][0] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
            m_out = np.zeros((nn_polar, nn_win_lig_m1s2, 2 * (sub_nncol + nn_win_col)))
        else:
            # FSEEK NNwinL LINES
            for n in range(nn_polar_in):
                datafile[n].seek(
                    -1, nn_win_lig_m1 * 2 * nn_col * np.dtype(np.float32).itemsize
                )
            m_out = np.zeros((nn_polar, nn_win_lig_m1s2, sub_nncol + nn_win_col))
            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(nn_win_lig_m1s2):
                if pol_type == "S2":
                    for n in range(nn_polar_in):
                        util.mc_in[n] = np.fromfile(
                            datafile[n], dtype=np.float32, count=2 * nn_col
                        )
                elif pol_type == "SPPpp1":
                    util.mc_in[0] = np.fromfile(
                        datafile[hh], dtype=np.float32, count=2 * nn_col
                    )
                    util.mc_in[1] = np.fromfile(
                        datafile[vh], dtype=np.float32, count=2 * nn_col
                    )
                elif pol_type == "SPPpp2":
                    util.mc_in[0] = np.fromfile(
                        datafile[vv], dtype=np.float32, count=2 * nn_col
                    )
                    util.mc_in[1] = np.fromfile(
                        datafile[hv], dtype=np.float32, count=2 * nn_col
                    )
                elif pol_type == "SPPpp3":
                    util.mc_in[0] = np.fromfile(
                        datafile[hh], dtype=np.float32, count=2 * nn_col
                    )
                    util.mc_in[1] = np.fromfile(
                        datafile[vv], dtype=np.float32, count=2 * nn_col
                    )

                m_out = np.zeros((nn_polar, nn_win_lig_m1s2, sub_nncol + nn_win_col))

                for col in range(2 * sub_nncol):
                    for n in range(nn_polar):
                        m_out[n, lig, col + 2 * nn_win_col_m1_s2] = util.mc_in[n][
                            col + 2 * ooff_col
                        ]

        # * READING NLIG LINES */
        for lig in range(sub_nnlig + nn_win_lig_m1s2):
            if nn_bblock == 1:
                if lig % int((sub_nnlig + nn_win_lig_m1s2) / 20) == 0:
                    print(f"{100 * lig / (sub_nnlig + nn_win_lig_m1s2 - 1)}%", end="\r")
                    sys.stdout.flush()

            # /* 1 line reading with zero padding */
            if lig < sub_nnlig:
                if pol_type == "S2":
                    for n in range(nn_polar_in):
                        util.mc_in[n][0] = np.fromfile(
                            datafile[0], dtype=np.float32, count=2 * nn_col
                        )
                elif pol_type == "SPPpp1":
                    util.mc_in[0] = np.fromfile(
                        datafile[hh], dtype=np.float32, count=2 * nn_col
                    )
                    util.mc_in[1] = np.fromfile(
                        datafile[vh], dtype=np.float32, count=2 * nn_col
                    )
                elif pol_type == "SPPpp2":
                    util.mc_in[0] = np.fromfile(
                        datafile[vv], dtype=np.float32, count=2 * nn_col
                    )
                    util.mc_in[1] = np.fromfile(
                        datafile[hv], dtype=np.float32, count=2 * nn_col
                    )
                elif pol_type == "SPPpp3":
                    util.mc_in[0] = np.fromfile(
                        datafile[hh], dtype=np.float32, count=2 * nn_col
                    )
                    util.mc_in[1] = np.fromfile(
                        datafile[vv], dtype=np.float32, count=2 * nn_col
                    )
            else:
                if nn_block == (nn_bblock - 1):
                    if pol_type == "S2":
                        for n in range(nn_polar_in):
                            for col in range(2 * nn_col):
                                util.mc_in[n][col] = 0.0
                    else:
                        for n in range(2):
                            for col in range(2 * nn_col):
                                util.mc_in[n][col] = 0.0
                else:
                    if pol_type == "S2":
                        for n in range(nn_polar_in):
                            util.mc_in[n] = np.fromfile(
                                datafile[n], dtype=np.float32, count=2 * nn_col
                            )
                    else:
                        if pol_type == "SPPpp1":
                            util.mc_in[0] = np.fromfile(
                                datafile[hh], dtype=np.float32, count=2 * nn_col
                            )
                            util.mc_in[1] = np.fromfile(
                                datafile[vh], dtype=np.float32, count=2 * nn_col
                            )
                        if pol_type == "SPPpp2":
                            util.mc_in[0] = np.fromfile(
                                datafile[vv], dtype=np.float32, count=2 * nn_col
                            )
                            util.mc_in[1] = np.fromfile(
                                datafile[hv], dtype=np.float32, count=2 * nn_col
                            )
                        if pol_type == "SPPpp3":
                            util.mc_in[0] = np.fromfile(
                                datafile[hh], dtype=np.float32, count=2 * nn_col
                            )
                            util.mc_in[1] = np.fromfile(
                                datafile[vv], dtype=np.float32, count=2 * nn_col
                            )
            for n in range(nn_polar):
                for col in range(sub_nncol + nn_win_col):
                    m_out[n][nn_win_lig_m1s2 + lig][col] = 0.0

            # /* Row-wise shift */
            for col in range(2 * sub_nncol):
                for n in range(nn_polar):
                    m_out[n][nn_win_lig_m1s2 + lig][
                        col + 2 * nn_win_col_m1_s2
                    ] = util.mc_in[n][col + 2 * ooff_col]
    else:
        if nn_block == 0:
            # OFFSET LINES READING
            for lig in range(ooff_lig):
                for n in range(nn_polar_in):
                    util.mc_in[n] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
            # Set the Tmp matrix to 0
            for lig in range(nn_win_lig_m1s2):
                for col in range(sub_nncol + nn_win_col):
                    for n in range(nn_polar):
                        m_out[n][lig][col] = 0.0
        else:
            for n in range(nn_polar_in):
                datafile[n].seek(
                    -1, nn_win_lig_m1 * 2 * nn_col * np.dtype(np.float32).itemsize
                )

            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(nn_win_lig_m1s2):
                for n in range(nn_polar_in):
                    util.mc_in[n] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )

                for n in range(nn_polar):
                    for col in range(sub_nncol + nn_win_col):
                        m_out[n][lig][col] = 0.0

                for col in range(ooff_col, sub_nncol + ooff_col):
                    if pol_type == "IPPpp4":
                        k1r = util.mc_in[hh][2 * col]
                        k1i = util.mc_in[hh][2 * col + 1]
                        k2r = (
                            util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                        ) / math.sqrt(2)
                        k2i = (
                            util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                        ) / math.sqrt(2)
                        k3r = util.mc_in[vv][2 * col]
                        k3i = util.mc_in[vv][2 * col + 1]
                        m_out[0][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[1][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )
                        m_out[2][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k3r + k3i * k3i
                        )

                    if pol_type == "IPPpp5":
                        k1r = util.mc_in[hh][2 * col]
                        k1i = util.mc_in[hh][2 * col + 1]
                        k2r = util.mc_in[vh][2 * col]
                        k2i = util.mc_in[vh][2 * col + 1]
                        m_out[0][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[1][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )

                    if pol_type == "IPPpp6":
                        k1r = util.mc_in[vv][2 * col]
                        k1i = util.mc_in[vv][2 * col + 1]
                        k2r = util.mc_in[hv][2 * col]
                        k2i = util.mc_in[hv][2 * col + 1]
                        m_out[0][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[1][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )

                    if pol_type == "IPPpp7":
                        k1r = util.mc_in[hh][2 * col]
                        k1i = util.mc_in[hh][2 * col + 1]
                        k2r = util.mc_in[vv][2 * col]
                        k2i = util.mc_in[vv][2 * col + 1]
                        m_out[0][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[1][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )

                    if pol_type == "T3":
                        k1r = (
                            util.mc_in[hh][2 * col] + util.mc_in[vv][2 * col]
                        ) / math.sqrt(2)
                        k1i = (
                            util.mc_in[hh][2 * col + 1] + util.mc_in[vv][2 * col + 1]
                        ) / math.sqrt(2)
                        k2r = (
                            util.mc_in[hh][2 * col] - util.mc_in[vv][2 * col]
                        ) / math.sqrt(2)
                        k2i = (
                            util.mc_in[hh][2 * col + 1] - util.mc_in[vv][2 * col + 1]
                        ) / math.sqrt(2)
                        k3r = (
                            util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                        ) / math.sqrt(2)
                        k3i = (
                            util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                        ) / math.sqrt(2)

                        m_out[util.T311][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[util.T312_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k2r + k1i * k2i
                        )
                        m_out[util.T312_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k2r - k1r * k2i
                        )
                        m_out[util.T313_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k3r + k1i * k3i
                        )
                        m_out[util.T313_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k3r - k1r * k3i
                        )
                        m_out[util.T322][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )
                        m_out[util.T323_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k3r + k2i * k3i
                        )
                        m_out[util.T323_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2i * k3r - k2r * k3i
                        )
                        m_out[util.T333][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k3r + k3i * k3i
                        )

                    if pol_type == "T4":
                        k1r = (
                            util.mc_in[hh][2 * col] + util.mc_in[vv][2 * col]
                        ) / math.sqrt(2)
                        k1i = (
                            util.mc_in[hh][2 * col + 1] + util.mc_in[vv][2 * col + 1]
                        ) / math.sqrt(2)
                        k2r = (
                            util.mc_in[hh][2 * col] - util.mc_in[vv][2 * col]
                        ) / math.sqrt(2)
                        k2i = (
                            util.mc_in[hh][2 * col + 1] - util.mc_in[vv][2 * col + 1]
                        ) / math.sqrt(2)
                        k3r = (
                            util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                        ) / math.sqrt(2)
                        k3i = (
                            util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                        ) / math.sqrt(2)
                        k4r = (
                            util.mc_in[vh][2 * col + 1] - util.mc_in[hv][2 * col + 1]
                        ) / math.sqrt(2)
                        k4i = (
                            util.mc_in[hv][2 * col] - util.mc_in[vh][2 * col]
                        ) / math.sqrt(2)

                        m_out[util.T411][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[util.T412_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k2r + k1i * k2i
                        )
                        m_out[util.T412_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k2r - k1r * k2i
                        )
                        m_out[util.T413_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k3r + k1i * k3i
                        )
                        m_out[util.T413_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k3r - k1r * k3i
                        )
                        m_out[util.T414_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k4r + k1i * k4i
                        )
                        m_out[util.T414_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k4r - k1r * k4i
                        )
                        m_out[util.T422][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )
                        m_out[util.T423_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k3r + k2i * k3i
                        )
                        m_out[util.T423_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2i * k3r - k2r * k3i
                        )
                        m_out[util.T424_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k4r + k2i * k4i
                        )
                        m_out[util.T424_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2i * k4r - k2r * k4i
                        )
                        m_out[util.T433][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k3r + k3i * k3i
                        )
                        m_out[util.T434_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k4r + k3i * k4i
                        )
                        m_out[util.T434_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3i * k4r - k3r * k4i
                        )
                        m_out[util.T444][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k4r * k4r + k4i * k4i
                        )

                    if pol_type == "C3":
                        k1r = util.mc_in[hh][2 * col]
                        k1i = util.mc_in[hh][2 * col + 1]
                        k2r = (
                            util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                        ) / math.sqrt(2.0)
                        k2i = (
                            util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                        ) / math.sqrt(2.0)
                        k3r = util.mc_in[vv][2 * col]
                        k3i = util.mc_in[vv][2 * col + 1]

                        m_out[util.C311][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[util.C312_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k2r + k1i * k2i
                        )
                        m_out[util.C312_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k2r - k1r * k2i
                        )
                        m_out[util.C313_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k3r + k1i * k3i
                        )
                        m_out[util.C313_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k3r - k1r * k3i
                        )
                        m_out[util.C322][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )
                        m_out[util.C323_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k3r + k2i * k3i
                        )
                        m_out[util.C323_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2i * k3r - k2r * k3i
                        )
                        m_out[util.C333][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k3r + k3i * k3i
                        )

                    if pol_type == "C4":
                        k1r = util.mc_in[hh][2 * col]
                        k1i = util.mc_in[hh][2 * col + 1]
                        k2r = util.mc_in[hv][2 * col]
                        k2i = util.mc_in[hv][2 * col + 1]
                        k3r = util.mc_in[vh][2 * col]
                        k3i = util.mc_in[vh][2 * col + 1]
                        k4r = util.mc_in[vv][2 * col]
                        k4i = util.mc_in[vv][2 * col + 1]

                        m_out[util.C411][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[util.C412_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k2r + k1i * k2i
                        )
                        m_out[util.C412_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k2r - k1r * k2i
                        )
                        m_out[util.C413_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k3r + k1i * k3i
                        )
                        m_out[util.C413_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k3r - k1r * k3i
                        )
                        m_out[util.C414_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1r * k4r + k1i * k4i
                        )
                        m_out[util.C414_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k1i * k4r - k1r * k4i
                        )
                        m_out[util.C422][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k2r + k2i * k2i
                        )
                        m_out[util.C423_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k3r + k2i * k3i
                        )
                        m_out[util.C423_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2i * k3r - k2r * k3i
                        )
                        m_out[util.C424_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2r * k4r + k2i * k4i
                        )
                        m_out[util.C424_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k2i * k4r - k2r * k4i
                        )
                        m_out[util.C433][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k3r + k3i * k3i
                        )
                        m_out[util.C434_RE][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3r * k4r + k3i * k4i
                        )
                        m_out[util.C434_IM][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k3i * k4r - k3r * k4i
                        )
                        m_out[util.C444][lig][col - ooff_col + nn_win_col_m1_s2] = (
                            k4r * k4r + k4i * k4i
                        )

        # /* READING NLIG LINES */
        for lig in range(0, sub_nnlig + nn_win_lig_m1s2):
            if nn_bblock == 1:
                if lig % int((sub_nnlig + nn_win_lig_m1s2) / 20) == 0:
                    print(
                        "{:.2f}%\r".format(
                            100 * lig / (sub_nnlig + nn_win_lig_m1s2 - 1)
                        ),
                        end="",
                    )

            #   /* 1 line reading with zero padding */
            if lig < sub_nnlig:
                for n in range(nn_polar_in):
                    util.mc_in[n][:] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
            else:
                if nn_block == (nn_bblock - 1):
                    for n in range(nn_polar_in):
                        util.mc_in[n][:] = np.zeros(2 * nn_col, dtype=np.float32)
                else:
                    for n in range(nn_polar_in):
                        util.mc_in[n][:] = np.fromfile(
                            datafile[n], dtype=np.float32, count=2 * nn_col
                        )
            for n in range(nn_polar):
                for col in range(sub_nncol + nn_win_col):
                    m_out[n][nn_win_lig_m1s2 + lig][col] = 0.0

            # /* Row-wise shift */
            for col in range(ooff_col, sub_nncol + ooff_col):
                if pol_type == "IPPpp4":
                    k1r = util.mc_in[hh][2 * col]
                    k1i = util.mc_in[hh][2 * col + 1]
                    k2r = (
                        util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                    ) / math.sqrt(2.0)
                    k2i = (
                        util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k3r = util.mc_in[vv][2 * col]
                    k3i = util.mc_in[vv][2 * col + 1]
                    m_out[0][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[1][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)
                    m_out[2][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k3r + k3i * k3i)

                if pol_type == "IPPpp5":
                    k1r = util.mc_in[hh][2 * col]
                    k1i = util.mc_in[hh][2 * col + 1]
                    k2r = util.mc_in[vh][2 * col]
                    k2i = util.mc_in[vh][2 * col + 1]
                    m_out[0][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[1][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)

                if pol_type == "IPPpp6":
                    k1r = util.mc_in[vv][2 * col]
                    k1i = util.mc_in[vv][2 * col + 1]
                    k2r = util.mc_in[hv][2 * col]
                    k2i = util.mc_in[hv][2 * col + 1]
                    m_out[0][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[1][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)

                if pol_type == "IPPpp7":
                    k1r = util.mc_in[hh][2 * col]
                    k1i = util.mc_in[hh][2 * col + 1]
                    k2r = util.mc_in[vv][2 * col]
                    k2i = util.mc_in[vv][2 * col + 1]
                    m_out[0][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[1][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)

                if pol_type == "T3":
                    k1r = (
                        util.mc_in[hh][2 * col] + util.mc_in[vv][2 * col]
                    ) / math.sqrt(2.0)
                    k1i = (
                        util.mc_in[hh][2 * col + 1] + util.mc_in[vv][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k2r = (
                        util.mc_in[hh][2 * col] - util.mc_in[vv][2 * col]
                    ) / math.sqrt(2.0)
                    k2i = (
                        util.mc_in[hh][2 * col + 1] - util.mc_in[vv][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k3r = (
                        util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                    ) / math.sqrt(2.0)
                    k3i = (
                        util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                    ) / math.sqrt(2.0)

                    m_out[util.T311][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[util.T312_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k2r + k1i * k2i)
                    m_out[util.T312_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k2r - k1r * k2i)
                    m_out[util.T313_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k3r + k1i * k3i)
                    m_out[util.T313_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k3r - k1r * k3i)
                    m_out[util.T322][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)
                    m_out[util.T323_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k3r + k2i * k3i)
                    m_out[util.T323_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2i * k3r - k2r * k3i)
                    m_out[util.T333][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k3r + k3i * k3i)

                if pol_type == "T4":
                    k1r = (
                        util.mc_in[hh][2 * col] + util.mc_in[vv][2 * col]
                    ) / math.sqrt(2.0)
                    k1i = (
                        util.mc_in[hh][2 * col + 1] + util.mc_in[vv][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k2r = (
                        util.mc_in[hh][2 * col] - util.mc_in[vv][2 * col]
                    ) / math.sqrt(2.0)
                    k2i = (
                        util.mc_in[hh][2 * col + 1] - util.mc_in[vv][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k3r = (
                        util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                    ) / math.sqrt(2.0)
                    k3i = (
                        util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k4r = (
                        util.mc_in[vh][2 * col + 1] - util.mc_in[hv][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k4i = (
                        util.mc_in[hv][2 * col] - util.mc_in[vh][2 * col]
                    ) / math.sqrt(2.0)

                    m_out[util.T411][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[util.T412_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k2r + k1i * k2i)
                    m_out[util.T412_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k2r - k1r * k2i)
                    m_out[util.T413_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k3r + k1i * k3i)
                    m_out[util.T413_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k3r - k1r * k3i)
                    m_out[util.T414_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k4r + k1i * k4i)
                    m_out[util.T414_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k4r - k1r * k4i)
                    m_out[util.T422][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)
                    m_out[util.T423_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k3r + k2i * k3i)
                    m_out[util.T423_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2i * k3r - k2r * k3i)
                    m_out[util.T424_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k4r + k2i * k4i)
                    m_out[util.T424_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2i * k4r - k2r * k4i)
                    m_out[util.T433][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k3r + k3i * k3i)
                    m_out[util.T434_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k4r + k3i * k4i)
                    m_out[util.T434_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3i * k4r - k3r * k4i)
                    m_out[util.T444][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k4r * k4r + k4i * k4i)

                if pol_type == "C3":
                    k1r = util.mc_in[hh][2 * col]
                    k1i = util.mc_in[hh][2 * col + 1]
                    k2r = (
                        util.mc_in[hv][2 * col] + util.mc_in[vh][2 * col]
                    ) / math.sqrt(2.0)
                    k2i = (
                        util.mc_in[hv][2 * col + 1] + util.mc_in[vh][2 * col + 1]
                    ) / math.sqrt(2.0)
                    k3r = util.mc_in[vv][2 * col]
                    k3i = util.mc_in[vv][2 * col + 1]

                    m_out[util.C311][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[util.C312_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k2r + k1i * k2i)
                    m_out[util.C312_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k2r - k1r * k2i)
                    m_out[util.C313_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k3r + k1i * k3i)
                    m_out[util.C313_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k3r - k1r * k3i)
                    m_out[util.C322][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)
                    m_out[util.C323_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k3r + k2i * k3i)
                    m_out[util.C323_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2i * k3r - k2r * k3i)
                    m_out[util.C333][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k3r + k3i * k3i)

                if pol_type == "C4":
                    k1r = util.mc_in[hh][2 * col]
                    k1i = util.mc_in[hh][2 * col + 1]
                    k2r = util.mc_in[hv][2 * col]
                    k2i = util.mc_in[hv][2 * col + 1]
                    k3r = util.mc_in[vh][2 * col]
                    k3i = util.mc_in[vh][2 * col + 1]
                    k4r = util.mc_in[vv][2 * col]
                    k4i = util.mc_in[vv][2 * col + 1]

                    m_out[util.C411][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[util.C412_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k2r + k1i * k2i)
                    m_out[util.C412_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k2r - k1r * k2i)
                    m_out[util.C413_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k3r + k1i * k3i)
                    m_out[util.C413_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k3r - k1r * k3i)
                    m_out[util.C414_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1r * k4r + k1i * k4i)
                    m_out[util.C414_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k1i * k4r - k1r * k4i)
                    m_out[util.C422][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k2r + k2i * k2i)
                    m_out[util.C423_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k3r + k2i * k3i)
                    m_out[util.C423_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2i * k3r - k2r * k3i)
                    m_out[util.C424_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2r * k4r + k2i * k4i)
                    m_out[util.C424_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k2i * k4r - k2r * k4i)
                    m_out[util.C433][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k3r + k3i * k3i)
                    m_out[util.C434_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3r * k4r + k3i * k4i)
                    m_out[util.C434_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k3i * k4r - k3r * k4i)
                    m_out[util.C444][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1_s2
                    ] = (k4r * k4r + k4i * k4i)
    return m_out


# %% [codecell] read_block_SPP_noavg
def read_block_SPP_noavg(
    datafile,
    m_out,
    pol_type,
    nn_polar,
    nn_block,
    nn_bblock,
    sub_nnlig,
    sub_nncol,
    nn_win_lig,
    nn_win_col,
    ooff_lig,
    ooff_col,
    nn_col,
):
    PolT = "SPP"
    chx1, chx2, nn_polar_in = 0, 1, 2
    nn_win_lig_m1s2 = (nn_win_lig - 1) / 2
    nn_win_col_m1s2 = (nn_win_col - 1) / 2
    nn_win_lig_m1 = nn_win_lig - 1
    k1r, k1i, k2r, k2i = 0, 0, 0, 0

    if pol_type in ["SPP", "SPPpp1", "SPPpp2", "SPPpp3"]:
        PolT = "SPP"
    elif pol_type in ["IPPpp5", "IPPpp6", "IPPpp7"]:
        PolT = "IPP"
    elif pol_type in ["C2", "C2pp1", "C2pp2", "C2pp3"]:
        PolT = "C2"
    elif pol_type in ["T2", "T2pp1", "T2pp2", "T2pp3"]:
        PolT = "T2"

    if PolT == "SPP":
        if nn_block == 0:
            # OFFSET LINES READING
            for lig in range(ooff_lig):
                for n in range(nn_polar_in):
                    util.mc_in[n][0] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
            m_out = np.zeros((nn_polar, nn_win_lig_m1s2, 2 * (sub_nncol + nn_win_col)))
        else:
            # FSEEK NNwinL LINES
            for n in range(nn_polar_in):
                datafile[n].seek(
                    -1 * nn_win_lig_m1 * 2 * nn_col * np.dtype(np.float32).itemsize, 2
                )

            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(nn_win_lig_m1s2):
                for n in range(nn_polar_in):
                    util.mc_in[n] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
                    m_out[n][lig][:] = 0
                    m_out[n][lig][
                        2 * nn_win_col_m1s2 : 2 * nn_win_col_m1s2 + 2 * sub_nncol
                    ] = util.mc_in[n][2 * ooff_col : 2 * ooff_col + 2 * sub_nncol]

        # READING NLIG LINES
        for lig in range(sub_nnlig + nn_win_lig_m1s2):
            if nn_bblock == 1:
                if lig % int((sub_nnlig + nn_win_lig_m1s2) / 20) == 0:
                    print(
                        "{:.2f}%\r".format(
                            100 * lig / (sub_nnlig + nn_win_lig_m1s2 - 1)
                        ),
                        end="",
                    )

            for n in range(nn_polar_in):
                if lig < sub_nnlig:
                    util.mc_in[n] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
                    m_out[n][nn_win_lig_m1s2 + lig][:] = 0
                    m_out[n][nn_win_lig_m1s2 + lig][
                        2 * nn_win_col_m1s2 : 2 * nn_win_col_m1s2 + 2 * sub_nncol
                    ] = util.mc_in[n][2 * ooff_col : 2 * ooff_col + 2 * sub_nncol]
                else:
                    if nn_block == (nn_bblock - 1):
                        m_out[n][nn_win_lig_m1s2 + lig][:] = 0
                    else:
                        util.mc_in[n] = np.fromfile(
                            datafile[n], dtype=np.float32, count=2 * nn_col
                        )
                        m_out[n][nn_win_lig_m1s2 + lig][:] = 0
                        m_out[n][nn_win_lig_m1s2 + lig][
                            2 * nn_win_col_m1s2 : 2 * nn_win_col_m1s2 + 2 * sub_nncol
                        ] = util.mc_in[n][2 * ooff_col : 2 * ooff_col + 2 * sub_nncol]
    else:
        if nn_block == 0:
            # OFFSET LINES READING
            for n in range(nn_polar_in):
                util.mc_in[n] = np.fromfile(
                    datafile[n], dtype=np.float32, count=2 * nn_col
                )

            # Set the Tmp matrix to 0
            m_out = np.zeros(nn_polar, nn_win_lig_m1s2, nn_col + nn_win_col)
        else:
            # FSEEK NNwinL LINES
            for n in range(nn_polar_in):
                datafile[n].seek(
                    -1 * nn_win_lig_m1 * 2 * nn_col * np.dtype(np.float32).itemsize,
                    os.SEEK_CUR,
                )

            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(nn_win_lig_m1s2):
                for n in range(nn_polar_in):
                    util.mc_in[n] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )

                for n in range(nn_polar):
                    m_out[n][lig][: sub_nncol + nn_win_col] = 0.0

                for col in range(ooff_col, sub_nncol + ooff_col):
                    if PolT == "C2":
                        k1r = util.mc_in[chx1][2 * col]
                        k1i = util.mc_in[chx1][2 * col + 1]
                        k2r = util.mc_in[chx2][2 * col]
                        k2i = util.mc_in[chx2][2 * col + 1]
                        m_out[util.C211][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[util.C212_RE][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1r * k2r + k1i * k2i
                        )
                        m_out[util.C212_IM][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1i * k2r - k1r * k2i
                        )
                        m_out[util.C222][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k2r * k2r + k2i * k2i
                        )
                    if PolT == "T2":
                        k1r = (
                            util.mc_in[chx1][2 * col] + util.mc_in[chx2][2 * col]
                        ) / math.sqrt(2.0)
                        k1i = (
                            util.mc_in[chx1][2 * col + 1]
                            + util.mc_in[chx2][2 * col + 1]
                        ) / math.sqrt(2.0)
                        k2r = (
                            util.mc_in[chx1][2 * col] - util.mc_in[chx2][2 * col]
                        ) / math.sqrt(2.0)
                        k2i = (
                            util.mc_in[chx1][2 * col + 1]
                            - util.mc_in[chx2][2 * col + 1]
                        ) / math.sqrt(2.0)

                        m_out[util.T211][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[util.T212_RE][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1r * k2r + k1i * k2i
                        )
                        m_out[util.T212_IM][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1i * k2r - k1r * k2i
                        )
                        m_out[util.T222][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k2r * k2r + k2i * k2i
                        )
                    if PolT == "IPP":
                        k1r = util.mc_in[chx1][2 * col]
                        k1i = util.mc_in[chx1][2 * col + 1]
                        k2r = util.mc_in[chx2][2 * col]
                        k2i = util.mc_in[chx2][2 * col + 1]

                        m_out[chx1][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k1r * k1r + k1i * k1i
                        )
                        m_out[chx2][lig][col - ooff_col + nn_win_col_m1s2] = (
                            k2r * k2r + k2i * k2i
                        )

        # READING AND AVERAGING NLIG LINES
        for lig in range(sub_nnlig + nn_win_lig_m1s2):
            if nn_bblock == 1:
                if lig % int((sub_nnlig + nn_win_lig_m1s2) / 20) == 0:
                    print(
                        "{}%\r".format(100.0 * lig / (sub_nnlig + nn_win_lig_m1s2 - 1))
                    )

            # 1 line reading with zero padding
            if lig < sub_nnlig:
                for n in range(nn_polar_in):
                    util.mc_in[n] = np.fromfile(
                        datafile[n], dtype=np.float32, count=2 * nn_col
                    )
            else:
                if nn_block == (nn_bblock - 1):
                    for n in range(nn_polar_in):
                        util.mc_in[n] = np.zeros(2 * nn_col)
                else:
                    for n in range(nn_polar_in):
                        util.mc_in[n] = np.fromfile(
                            datafile[n], dtype=np.float32, count=2 * nn_col
                        )
            m_out = np.zeros(
                nn_polar, sub_nnlig + nn_win_lig_m1s2, sub_nncol + nn_win_col
            )

            for col in range(ooff_col, sub_nncol + ooff_col):
                if PolT == "C2":
                    k1r = util.mc_in[chx1][2 * col]
                    k1i = util.mc_in[chx1][2 * col + 1]
                    k2r = util.mc_in[chx2][2 * col]
                    k2i = util.mc_in[chx2][2 * col + 1]

                    m_out[util.C211][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[util.C212_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1r * k2r + k1i * k2i)
                    m_out[util.C212_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1i * k2r - k1r * k2i)
                    m_out[util.C222][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k2r * k2r + k2i * k2i)
                if PolT == "T2":
                    k1r = (
                        util.mc_in[chx1][2 * col] + util.mc_in[chx2][2 * col]
                    ) / np.sqrt(2)
                    k1i = (
                        util.mc_in[chx1][2 * col + 1] + util.mc_in[chx2][2 * col + 1]
                    ) / np.sqrt(2)
                    k2r = (
                        util.mc_in[chx1][2 * col] - util.mc_in[chx2][2 * col]
                    ) / np.sqrt(2)
                    k2i = (
                        util.mc_in[chx1][2 * col + 1] - util.mc_in[chx2][2 * col + 1]
                    ) / np.sqrt(2)

                    m_out[util.T211][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[util.T212_RE][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1r * k2r + k1i * k2i)
                    m_out[util.T212_IM][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1i * k2r - k1r * k2i)
                    m_out[util.T222][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k2r * k2r + k2i * k2i)
                if PolT == "IPP":
                    k1r = util.mc_in[chx1][2 * col]
                    k1i = util.mc_in[chx1][2 * col + 1]
                    k2r = util.mc_in[chx2][2 * col]
                    k2i = util.mc_in[chx2][2 * col + 1]

                    m_out[chx1][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k1r * k1r + k1i * k1i)
                    m_out[chx2][nn_win_lig_m1s2 + lig][
                        col - ooff_col + nn_win_col_m1s2
                    ] = (k2r * k2r + k2i * k2i)

    return m_out


@numba.njit
def read_block_spp_noavg(datafile, M_out, PolType, NNpolar, NNblock, NNbBlock, Sub_NNlig, Sub_NNcol, NNwinLig, NNwinCol, OOff_lig, OOff_col, NNcol, _MC_in):
    '''
    Routine  : read_block_SPP_noavg
    Authors  : Eric POTTIER
    Creation : 08/2010
    Update  :
    *--------------------------------------------------------------------
    Description : Read SPP Partial Sinclair matrix
            without applying a spatial averaging
    ********************************************************************/
    '''
    NNpolarIn = 2
    Chx1 = 0
    Chx2 = 1

    NNwinLigM1S2 = (NNwinLig - 1) / 2
    NNwinColM1S2 = (NNwinCol - 1) / 2
    NNwinLigM1 = (NNwinLig - 1)

    PolT = 'SPP'
    if PolType in ['SPP', 'SPPpp1', 'SPPpp2', 'SPPpp3']:
        PolT = "SPP"
    elif PolType in ['IPPpp5', 'IPPpp6', 'IPPpp7']:
        PolT = "IPP"
    elif PolType in ['C2', 'C2pp1', 'C2pp2', 'C2pp3']:
        PolT = "C2"
    elif PolType in ['T2', 'T2pp1', 'T2pp2', 'T2pp3']:
        PolT = "T2"

    if PolT == 'SPP':
        if NNblock == 0:
            # OFFSET LINES READING
            for lig in range(OOff_lig):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
            # Set the Tmp matrix to 0
            for lig in range(NNwinLigM1S2):
                for col in range(2 * (Sub_NNcol + NNwinCol)):
                    for Np in range(NNpolarIn):
                        M_out[Np][lig][col] = 0.

        else:
            # FSEEK NNwinL LINES
            for Np in range(NNpolarIn):
                util.my_fseek(datafile[Np], -1, NNwinLigM1, 2 * NNcol * np.dtype(np.float32).itemsize)
            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(NNwinLigM1S2):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
                    for col in range(Sub_NNcol + NNwinCol):
                        M_out[Np][lig][col] = 0.
                    for col in range(2 * Sub_NNcol):
                        M_out[Np][lig][col + 2 * NNwinColM1S2] = _MC_in[Np][col + 2 * OOff_col]
        # READING NLIG LINES
        for lig in range(Sub_NNlig + NNwinLigM1S2):
            if NNbBlock == 1:
                if lig % (int)((Sub_NNlig + NNwinLigM1S2) / 20) == 0:
                    print("{:.2f}%\r".format(100. * lig / (Sub_NNlig + NNwinLigM1S2 - 1)), end="", flush=True)

            # 1 line reading with zero padding
            for Np in range(NNpolarIn):
                if lig < Sub_NNlig:
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
                    for col in range(Sub_NNcol + NNwinCol):
                        M_out[Np][NNwinLigM1S2 + lig][col] = 0.
                    for col in range(2 * Sub_NNcol):
                        M_out[Np][NNwinLigM1S2 + lig][col + 2 * NNwinColM1S2] = _MC_in[Np][col + 2 * OOff_col]
                else:
                    if NNblock == (NNbBlock - 1):
                        for col in range(Sub_NNcol + NNwinCol):
                            M_out[Np][NNwinLigM1S2 + lig][col] = 0.
                    else:
                        _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
                        for col in range(Sub_NNcol + NNwinCol):
                            M_out[Np][NNwinLigM1S2 + lig][col] = 0.
                        for col in range(2 * Sub_NNcol):
                            M_out[Np][NNwinLigM1S2 + lig][col + 2 * NNwinColM1S2] = _MC_in[Np][col + 2 * OOff_col]
    else:
        if NNblock == 0:
            # OFFSET LINES READING
            for lig in range(OOff_lig):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
            # Set the Tmp matrix to 0
            for lig in range(NNwinLigM1S2):
                for col in range(NNcol + NNwinCol):
                    for Np in range(NNpolar):
                        M_out[Np][lig][col] = 0.
        else:
            # FSEEK NNwinL LINES
            for Np in range(NNpolarIn):
                util.my_fseek(datafile[Np], -1, NNwinLigM1, 2 * NNcol * np.dtype(np.float32).itemsize)
            # FIRST (NNwin+1)/2 LINES READING */
            for lig in range(NNwinLigM1S2):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
            # Set the Tmp matrix to 0

                for Np in range(NNpolar):
                    for col in range(Sub_NNcol + NNwinCol):
                        M_out[Np][lig][col] = 0.

                for col in range(OOff_col, Sub_NNcol + OOff_col):
                    if PolT == 'C2':
                        k1r = _MC_in[Chx1][2 * col]
                        k1i = _MC_in[Chx1][2 * col + 1]
                        k2r = _MC_in[Chx2][2 * col]
                        k2i = _MC_in[Chx2][2 * col + 1]
                        M_out[util.C211][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[util.C212_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                        M_out[util.C212_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                        M_out[util.C222][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                    if PolT == 'T2':
                        k1r = (_MC_in[Chx1][2 * col] + _MC_in[Chx2][2 * col]) / math.sqrt(2.)
                        k1i = (_MC_in[Chx1][2 * col + 1] + _MC_in[Chx2][2 * col + 1]) / math.sqrt(2.)
                        k2r = (_MC_in[Chx1][2 * col] - _MC_in[Chx2][2 * col]) / math.sqrt(2.)
                        k2i = (_MC_in[Chx1][2 * col + 1] - _MC_in[Chx2][2 * col + 1]) / math.sqrt(2.)
                        M_out[util.T211][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[util.T212_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                        M_out[util.T212_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                        M_out[util.T222][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                    if PolT == 'IPP':
                        k1r = _MC_in[Chx1][2 * col]
                        k1i = _MC_in[Chx1][2 * col + 1]
                        k2r = _MC_in[Chx2][2 * col]
                        k2i = _MC_in[Chx2][2 * col + 1]
                        M_out[Chx1][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[Chx2][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i

        # READING AND AVERAGING NLIG LINES
        for lig in range(Sub_NNlig + NNwinLigM1S2):
            if NNbBlock == 1:
                if lig % (int)((Sub_NNlig + NNwinLigM1S2) / 20) == 0:
                    print("{:.2f}%\r".format(100. * lig / (Sub_NNlig + NNwinLigM1S2 - 1)), end="", flush=True)

            # 1 line reading with zero padding
            if lig < Sub_NNlig:
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
            else:
                if NNblock == (NNbBlock - 1):
                    for Np in range(NNpolarIn):
                        for col in range(2 * NNcol):
                            _MC_in[Np][col] = 0.
                else:
                    for Np in range(NNpolarIn):
                        _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)

            for Np in range(NNpolar):
                for col in range(Sub_NNcol + NNwinCol):
                    M_out[Np][NNwinLigM1S2 + lig][col] = 0.

            # Row-wise shift
            for col in range(OOff_col, Sub_NNcol + OOff_col):
                if PolT == 'C2':
                    k1r = _MC_in[Chx1][2 * col]
                    k1i = _MC_in[Chx1][2 * col + 1]
                    k2r = _MC_in[Chx2][2 * col]
                    k2i = _MC_in[Chx2][2 * col + 1]
                    M_out[util.C211][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                    M_out[util.C212_RE][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                    M_out[util.C212_IM][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                    M_out[util.C222][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                if PolT == 'T2':
                    k1r = (_MC_in[Chx1][2 * col] + _MC_in[Chx2][2 * col]) / math.sqrt(2.)
                    k1i = (_MC_in[Chx1][2 * col + 1] + _MC_in[Chx2][2 * col + 1]) / math.sqrt(2.)
                    k2r = (_MC_in[Chx1][2 * col] - _MC_in[Chx2][2 * col]) / math.sqrt(2.)
                    k2i = (_MC_in[Chx1][2 * col + 1] - _MC_in[Chx2][2 * col + 1]) / math.sqrt(2.)
                    M_out[util.T211][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                    M_out[util.T212_RE][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                    M_out[util.T212_IM][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                    M_out[util.T222][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                if PolT == 'IPP':
                    k1r = _MC_in[Chx1][2 * col]
                    k1i = _MC_in[Chx1][2 * col + 1]
                    k2r = _MC_in[Chx2][2 * col]
                    k2i = _MC_in[Chx2][2 * col + 1]
                    M_out[Chx1][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                    M_out[Chx2][NNwinLigM1S2 + lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
    # else SPP
    return 1


@numba.njit(parallel=False)
def read_block_tci_noavg_reading_nlig_lines(_M_out, _NNpolar, _NNblock, _NNbBlock, _Sub_NNlig, _Sub_NNcol, _NNwinCol, _OOff_col, _NNwinLigM1S2, _NNwinColM1S2, _NNcol, _vf_in):
    lig_range = _Sub_NNlig + _NNwinLigM1S2
    for lig in range(lig_range):
        if _NNbBlock == 1:
            if lig % (int)((lig_range) / 20) == 0:
                # print("{:.2f}%\r".format(100. * lig / (_Sub_NNlig+_NNwinLigM1S2 - 1)), end="", flush = True)
                util.printf_line(lig, lig_range)

        # 1 line reading with zero padding
        for Np in range(_NNpolar):
            if lig < _Sub_NNlig:
                # _vf_in = np.fromfile(datafile[Np], dtype=np.float32, count=_NNcol)
                # print(f'{Np=} {_vf_in=}')
                # print(f'f{Np=} {__vf_in[Np]}')
                for col in range(_Sub_NNcol + _NNwinCol):
                    _M_out[Np][_NNwinLigM1S2 + lig][col] = 0.
                for col in range(_Sub_NNcol):
                    _M_out[Np][_NNwinLigM1S2 + lig][col + _NNwinColM1S2] = _vf_in[lig][Np][col + _OOff_col]
            else:
                if _NNblock == (_NNbBlock - 1):
                    for col in range(_Sub_NNcol + _NNwinCol):
                        _M_out[Np][_NNwinLigM1S2 + lig][col] = 0.
                else:
                    # _vf_in = np.fromfile(datafile[Np], dtype=np.float32, count=_NNcol)
                    # print(f'{Np=} {_vf_in=}')
                    # print('f{Np=} {__vf_in[Np]=}')
                    for col in range(_Sub_NNcol + _NNwinCol):
                        _M_out[Np][_NNwinLigM1S2 + lig][col] = 0.
                    for col in range(_Sub_NNcol):
                        _M_out[Np][_NNwinLigM1S2 + lig][col + _NNwinColM1S2] = _vf_in[lig][Np][col + _OOff_col]


@numba.njit(parallel=False)
def read_block_tci_noavg_set_tmp_matrix_to_zero(NNwinLigM1S2, M_out, NNpolar, Sub_NNcol, NNwinCol):
    for lig in range(NNwinLigM1S2):
        for col in range(Sub_NNcol + NNwinCol):
            for Np in range(NNpolar):
                M_out[Np][lig][col] = 0.


@numba.njit(parallel=False)
def read_block_tci_noavg_set_m_out(M_out, Sub_NNcol, NNwinCol, OOff_col, _VF_in, Np, lig, NNwinColM1S2):
    for col in range(Sub_NNcol + NNwinCol):
        M_out[Np][lig][col] = 0.
    for col in range(Sub_NNcol):
        M_out[Np][lig][col + NNwinColM1S2] = _VF_in[col + OOff_col]


@util.enter_exit_func_decorator
def read_block_tci_noavg(datafile, M_out, NNpolar, NNblock, NNbBlock, Sub_NNlig, Sub_NNcol, NNwinLig, NNwinCol, OOff_lig, OOff_col, NNcol, _VF_in, _vf_in_readingLines=None):
    """
    Description : Read T Coherency, C Covariance or I Intensity matrix without applying a spatial averaging
    """

    NNwinLigM1S2 = (NNwinLig - 1) // 2
    NNwinColM1S2 = (NNwinCol - 1) // 2
    NNwinLigM1 = (NNwinLig - 1)
    # logging.info(f'{OOff_lig=} {NNpolar=} {NNwinLigM1S2=}')

    if NNblock == 0:
        # OFFSET LINES READING
        # logging.info('OFFSET LINES READING 1')
        for lig in range(OOff_lig):
            for Np in range(NNpolar):
                _VF_in = np.fromfile(datafile[Np], dtype=np.float32, count=NNcol)

        # logging.info('OFFSET LINES READING 2')
        # Set the Tmp matrix to 0
        read_block_tci_noavg_set_tmp_matrix_to_zero(NNwinLigM1S2, M_out, NNpolar, Sub_NNcol, NNwinCol)
        # for lig in range(NNwinLigM1S2):
        #     for col in range(Sub_NNcol + NNwinCol):
        #         for Np in range(NNpolar):
        #             M_out[Np][lig][col] = 0.
    else:
        # FSEEK NNwinL LINES
        # logging.info('FSEEK NNwinL LINES')
        for Np in range(NNpolar):
            util.my_fseek(datafile[Np], -1, NNwinLigM1, NNcol * np.dtype(np.float32).itemsize)
        # FIRST (NNwin+1)/2 LINES READING
        # logging.info('FIRST (NNwin+1)/2 LINES READING')
        for lig in range(NNwinLigM1S2):
            for Np in range(NNpolar):
                _VF_in = np.fromfile(datafile[Np], dtype=np.float32, count=NNcol)
                read_block_tci_noavg_set_m_out(M_out, Sub_NNcol, NNwinCol, OOff_col, _VF_in, Np, lig, NNwinColM1S2)
                # for col in range(Sub_NNcol + NNwinCol):
                #     M_out[Np][lig][col] = 0.
                # for col in range(Sub_NNcol):
                #     M_out[Np][lig][col + NNwinColM1S2] = _VF_in[col + OOff_col]

    # READING NLIG LINES
    # logging.info('READING NLIG LINES 1')
    _vf_in = None
    if _vf_in_readingLines is None:
        logging.info(f'READING NLIG LINES 1 {NNcol=} {NNpolar=} {Sub_NNlig=} {NNwinLigM1S2=} from util_block')
        _vf_in = [numba.typed.List([np.fromfile(datafile[Np], dtype=np.float32, count=NNcol) for Np in range(NNpolar)]) for lig in range(Sub_NNlig + NNwinLigM1S2)]
    else:
        _vf_in = _vf_in_readingLines
    # logging.info('READING NLIG LINES 2')
    read_block_tci_noavg_reading_nlig_lines(M_out, NNpolar, NNblock, NNbBlock, Sub_NNlig, Sub_NNcol, NNwinCol, OOff_col, NNwinLigM1S2, NNwinColM1S2, NNcol, numba.typed.List(_vf_in))

#KS #@util.enter_exit_func_decorator
#KS def ks_read_block_tci_noavg(datafile, M_out, NNpolar, NNblock, NNbBlock, Sub_NNlig, Sub_NNcol, NNwinLig, NNwinCol, OOff_lig, OOff_col, NNcol, _VF_in):
#KS     """
#KS     Description : Read T Coherency, C Covariance or I Intensity matrix without applying a spatial averaging
#KS     """
#KS 
#KS     NNwinLigM1S2 = (NNwinLig - 1) // 2
#KS     NNwinColM1S2 = (NNwinCol - 1) // 2
#KS     NNwinLigM1 = (NNwinLig - 1)
#KS 
#KS     if NNblock == 0:
#KS       # OFFSET LINES READING
#KS       for lig in range(OOff_lig):
#KS           for Np in range(NNpolar):
#KS               _VF_in = np.fromfile(datafile[Np], dtype=np.float32, count=NNcol)
#KS 
#KS       # Set the Tmp matrix to 0
#KS       for lig in range(NNwinLigM1S2):
#KS           for col in range(Sub_NNcol + NNwinCol):
#KS               for Np in range(NNpolar):
#KS                   M_out[Np][lig][col] = 0.
#KS     else:
#KS         # FSEEK NNwinL LINES
#KS         for Np in range(NNpolar):
#KS               util.my_fseek(datafile[Np], -1, NNwinLigM1, NNcol * np.dtype(np.float32).itemsize);
#KS         # FIRST (NNwin+1)/2 LINES READING
#KS         for lig in range(NNwinLigM1S2):
#KS             for Np in range(NNpolar):
#KS                 _VF_in = np.fromfile(datafile[Np], dtype=np.float32, count=NNcol)
#KS                 print(f'f{Np=} {_VF_in}')
#KS                 for col in range(Sub_NNcol + NNwinCol):
#KS                     M_out[Np][lig][col] = 0.
#KS                 for col in range(Sub_NNcol):
#KS                     M_out[Np][lig][col + NNwinColM1S2] = _VF_in[col + OOff_col]
#KS 
#KS     # READING NLIG LINES
#KS     for lig in range(Sub_NNlig+NNwinLigM1S2):
#KS         if NNbBlock == 1:
#KS             if lig%(int)((Sub_NNlig+NNwinLigM1S2)/20) == 0:
#KS                 print("{:.2f}%\r".format(100. * lig / (Sub_NNlig+NNwinLigM1S2 - 1)), end="", flush = True)
#KS 
#KS         # 1 line reading with zero padding
#KS         for Np in range(NNpolar):
#KS             if lig < Sub_NNlig:
#KS                 _VF_in = np.fromfile(datafile[Np], dtype=np.float32, count=NNcol)
#KS                 print(f'f{Np=} {_VF_in}')
#KS                 for col in range(Sub_NNcol + NNwinCol):
#KS                     M_out[Np][NNwinLigM1S2+lig][col] = 0.
#KS                 for col in range(Sub_NNcol):
#KS                     M_out[Np][NNwinLigM1S2+lig][col + NNwinColM1S2] = _VF_in[col + OOff_col]
#KS             else:
#KS                 if NNblock == (NNbBlock - 1):
#KS                     for col in range(Sub_NNcol + NNwinCol):
#KS                         M_out[Np][NNwinLigM1S2+lig][col] = 0.
#KS                 else:
#KS                     _VF_in = np.fromfile(datafile[Np], dtype=np.float32, count=NNcol)
#KS                     print(f'f{Np=} {_VF_in}')
#KS                     for col in range(Sub_NNcol + NNwinCol):
#KS                          M_out[Np][NNwinLigM1S2+lig][col] = 0.;
#KS                     for col in range(Sub_NNcol):
#KS                         M_out[Np][NNwinLigM1S2+lig][col + NNwinColM1S2] = _VF_in[col + OOff_col]


@util.enter_exit_func_decorator
def read_block_s2_noavg(datafile, M_out, PolType, NNpolar, NNblock, NNbBlock, Sub_NNlig, Sub_NNcol, NNwinLig, NNwinCol, OOff_lig, OOff_col, NNcol, _MC_in):
    """
    Description : Read S2 Sinclair matrix without applying a spatial averaging
    """
    NNpolarIn = 4
    hh = 0
    hv = 1
    vh = 2
    vv = 3

    NNwinLigM1S2 = (NNwinLig - 1) // 2
    NNwinColM1S2 = (NNwinCol - 1) // 2
    NNwinLigM1 = (NNwinLig - 1)

    if PolType == "S2" or PolType == "SPPpp1" or PolType == "SPPpp2" or PolType == "SPPpp3":
        if NNblock == 0:
            # OFFSET LINES READING
            for lig in range(OOff_lig):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)

            # Set the Tmp matrix to 0
            for lig in range(NNwinLigM1S2):
                for col in range(2 * (Sub_NNcol + NNwinCol)):
                    for Np in range(NNpolar):
                        M_out[Np][lig][col] = 0.
        else:
            # FSEEK NNwinL LINES
            for Np in range(NNpolarIn):
                util.my_fseek(datafile[Np], -1, NNwinLigM1, 2 * NNcol * np.dtype(np.float32).itemsize)
            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(NNwinLigM1S2):
                if PolType == "S2":
                    for Np in range(NNpolarIn):
                        _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
                else:
                    if PolType == "SPPpp1":
                        _MC_in[0][0] = np.fromfile(datafile[hh], dtype=np.float32, count=2 * NNcol)
                        _MC_in[1][0] = np.fromfile(datafile[vh], dtype=np.float32, count=2 * NNcol)
                    if PolType == "SPPpp2":
                        _MC_in[0][0] = np.fromfile(datafile[vv], dtype=np.float32, count=2 * NNcol)
                        _MC_in[1][0] = np.fromfile(datafile[hv], dtype=np.float32, count=2 * NNcol)
                    if PolType == "SPPpp3":
                        _MC_in[0][0] = np.fromfile(datafile[hh], dtype=np.float32, count=2 * NNcol)
                        _MC_in[1][0] = np.fromfile(datafile[vv], dtype=np.float32, count=2 * NNcol)
                for Np in range(NNpolar):
                    for col in range(Sub_NNcol + NNwinCol):
                        M_out[Np][lig][col] = 0.

                for col in range(2 * Sub_NNcol):
                    for Np in range(NNpolar):
                        M_out[Np][lig][col + 2 * NNwinColM1S2] = _MC_in[Np][col + 2 * OOff_col]

        # READING NLIG LINES
        for lig in range(Sub_NNlig + NNwinLigM1S2):
            if NNbBlock == 1:
                if lig%(int)((Sub_NNlig + NNwinLigM1S2)/20) == 0:
                    print("%f\r", 100. * lig / (Sub_NNlig + NNwinLigM1S2 - 1), end="", flush = True)

            # 1 line reading with zero padding
            if lig < Sub_NNlig:
                if PolType == "S2":
                    for Np in range(NNpolarIn):
                        _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
                else:
                    if PolType == "SPPpp1":
                        _MC_in[0][0] = np.fromfile(datafile[hh], dtype=np.float32, count=2 * NNcol)
                        _MC_in[1][0] = np.fromfile(datafile[vh], dtype=np.float32, count=2 * NNcol)
                    if PolType == "SPPpp2":
                        _MC_in[0][0] = np.fromfile(datafile[vv], dtype=np.float32, count=2 * NNcol)
                        _MC_in[1][0] = np.fromfile(datafile[hv], dtype=np.float32, count=2 * NNcol)
                    if PolType == "SPPpp3":
                        _MC_in[0][0] = np.fromfile(datafile[hh], dtype=np.float32, count=2 * NNcol)
                        _MC_in[1][0] = np.fromfile(datafile[vv], dtype=np.float32, count=2 * NNcol)
            else:
                if NNblock == (NNbBlock - 1):
                    if PolType == "S2":
                        for Np in range(NNpolarIn):
                            for col in range(2 * NNcol):
                                _MC_in[Np][col] = 0.
                    else:
                        for Np in range(2):
                            for col in range(2 * NNcol):
                                _MC_in[Np][col] = 0.
                else:
                    if PolType == "S2":
                        for Np in range(NNpolarIn):
                            _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
                    else:
                        if PolType == "SPPpp1":
                            _MC_in[0][0] = np.fromfile(datafile[hh], dtype=np.float32, count=2 * NNcol)
                            _MC_in[1][0] = np.fromfile(datafile[vh], dtype=np.float32, count=2 * NNcol)
                        if PolType == "SPPpp2":
                            _MC_in[0][0] = np.fromfile(datafile[vv], dtype=np.float32, count=2 * NNcol)
                            _MC_in[1][0] = np.fromfile(datafile[hv], dtype=np.float32, count=2 * NNcol)
                        if PolType == "SPPpp3":
                            _MC_in[0][0] = np.fromfile(datafile[hh], dtype=np.float32, count=2 * NNcol)
                            _MC_in[1][0] = np.fromfile(datafile[vv], dtype=np.float32, count=2 * NNcol)

            for Np in range(NNpolar):
                for col in range(Sub_NNcol + NNwinCol):
                    M_out[Np][NNwinLigM1S2 + lig][col] = 0.

            # Row-wise shift
            for col in range(2 * Sub_NNcol):
                for Np in range(NNpolar):
                    M_out[Np][NNwinLigM1S2 + lig][col + 2 * NNwinColM1S2] = _MC_in[Np][col + 2 * OOff_col]
    else:
        if NNblock == 0:
            # OFFSET LINES READING */
            for lig in range(OOff_lig):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
            # Set the Tmp matrix to 0
            for lig in range(NNwinLigM1S2):
                for col in range(Sub_NNcol + NNwinCol):
                    for Np in range(NNpolar):
                        M_out[Np][lig][col] = 0.
        else:
            # FSEEK NNwinL LINES
            for Np in range(NNpolarIn):
                util.y_fseek(datafile[Np], -1, NNwinLigM1, 2 * NNcol * np.dtype(np.float32).itemsize)
            # FIRST (NNwin+1)/2 LINES READING
            for lig in range(NNwinLigM1S2):
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)

                for Np in range(NNpolar):
                    for col in range(Sub_NNcol + NNwinCol):
                        M_out[Np][lig][col] = 0.

                for col in range(OOff_col, Sub_NNcol + OOff_col):
                    if PolType == "IPPpp4":
                        k1r = _MC_in[hh][2 * col]
                        k1i = _MC_in[hh][2 * col + 1]
                        k2r = (_MC_in[hv][2 * col] + _MC_in[vh][2*col]) / math.sqrt(2.)
                        k2i = (_MC_in[hv][2 * col + 1] + _MC_in[vh][2*col + 1]) / sqrt(2.)
                        k3r = _MC_in[vv][2 * col]
                        k3i = _MC_in[vv][2*col + 1]
                        M_out[0][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[1][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                        M_out[2][lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i
                    if PolType == "IPPpp5":
                        k1r = _MC_in[hh][2*col]; k1i = _MC_in[hh][2*col+1]
                        k2r = _MC_in[vh][2*col]; k2i = _MC_in[vh][2*col+1]
                        M_out[0][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[1][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                    if PolType == "IPPpp6":
                        k1r = _MC_in[vv][2*col]; k1i = _MC_in[vv][2*col+1]
                        k2r = _MC_in[hv][2*col]; k2i = _MC_in[hv][2*col+1]
                        M_out[0][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[1][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                    if PolType == "IPPpp7":
                        k1r = _MC_in[hh][2*col]; k1i = _MC_in[hh][2*col+1]
                        k2r = _MC_in[vv][2*col]; k2i = _MC_in[vv][2*col+1]
                        M_out[0][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[1][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                    if PolType == "T3":
                        k1r = (_MC_in[hh][2*col] + _MC_in[vv][2*col]) / sqrt(2.)
                        k1i = (_MC_in[hh][2*col+1] + _MC_in[vv][2*col+1]) / sqrt(2.)
                        k2r = (_MC_in[hh][2*col] - _MC_in[vv][2*col]) / sqrt(2.)
                        k2i = (_MC_in[hh][2*col+1] - _MC_in[vv][2*col+1]) / sqrt(2.)
                        k3r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.)
                        k3i = (_MC_in[hv][2*col+1] + _MC_in[vh][2*col+1]) / sqrt(2.)

                        M_out[T311][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[T312_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                        M_out[T312_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                        M_out[T313_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i
                        M_out[T313_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i
                        M_out[T322][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                        M_out[T323_re][lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i
                        M_out[T323_im][lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i
                        M_out[T333][lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i
                    if PolType == "T4":
                        k1r = (_MC_in[hh][2*col] + _MC_in[vv][2*col]) / sqrt(2.)
                        k1i = (_MC_in[hh][2*col+1] + _MC_in[vv][2*col+1]) / sqrt(2.)
                        k2r = (_MC_in[hh][2*col] - _MC_in[vv][2*col]) / sqrt(2.)
                        k2i = (_MC_in[hh][2*col+1] - _MC_in[vv][2*col+1]) / sqrt(2.)
                        k3r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.)
                        k3i = (_MC_in[hv][2*col+1] + _MC_in[vh][2*col+1]) / sqrt(2.)
                        k4r = (_MC_in[vh][2*col+1] - _MC_in[hv][2*col+1]) / sqrt(2.)
                        k4i = (_MC_in[hv][2*col] - _MC_in[vh][2*col]) / sqrt(2.)
  
                        M_out[T411][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[T412_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                        M_out[T412_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                        M_out[T413_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i
                        M_out[T413_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i
                        M_out[T414_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k4r + k1i * k4i
                        M_out[T414_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k4r - k1r * k4i
                        M_out[T422][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                        M_out[T423_re][lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i
                        M_out[T423_im][lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i
                        M_out[T424_re][lig][col - OOff_col + NNwinColM1S2] = k2r * k4r + k2i * k4i
                        M_out[T424_im][lig][col - OOff_col + NNwinColM1S2] = k2i * k4r - k2r * k4i
                        M_out[T433][lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i
                        M_out[T434_re][lig][col - OOff_col + NNwinColM1S2] = k3r * k4r + k3i * k4i
                        M_out[T434_im][lig][col - OOff_col + NNwinColM1S2] = k3i * k4r - k3r * k4i
                        M_out[T444][lig][col - OOff_col + NNwinColM1S2] = k4r * k4r + k4i * k4i
                    if PolType == "C3":
                        k1r = _MC_in[hh][2*col]
                        k1i = _MC_in[hh][2*col + 1]
                        k2r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.)
                        k2i = (_MC_in[hv][2*col + 1] + _MC_in[vh][2*col + 1]) / sqrt(2.)
                        k3r = _MC_in[vv][2*col]
                        k3i = _MC_in[vv][2*col + 1]

                        M_out[C311][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[C312_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                        M_out[C312_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                        M_out[C313_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i
                        M_out[C313_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i
                        M_out[C322][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                        M_out[C323_re][lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i
                        M_out[C323_im][lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i
                        M_out[C333][lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i
                    if PolType == "C4":
                        k1r = _MC_in[hh][2*col]
                        k1i = _MC_in[hh][2*col+1]
                        k2r = _MC_in[hv][2*col]
                        k2i  = _MC_in[hv][2*col+1]
                        k3r = _MC_in[vh][2*col]
                        k3i = _MC_in[vh][2*col+1]
                        k4r = _MC_in[vv][2*col]
                        k4i = _MC_in[vv][2*col+1]

                        M_out[C411][lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i
                        M_out[C412_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i
                        M_out[C412_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i
                        M_out[C413_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i
                        M_out[C413_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i
                        M_out[C414_re][lig][col - OOff_col + NNwinColM1S2] = k1r * k4r + k1i * k4i
                        M_out[C414_im][lig][col - OOff_col + NNwinColM1S2] = k1i * k4r - k1r * k4i
                        M_out[C422][lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i
                        M_out[C423_re][lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i
                        M_out[C423_im][lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i
                        M_out[C424_re][lig][col - OOff_col + NNwinColM1S2] = k2r * k4r + k2i * k4i
                        M_out[C424_im][lig][col - OOff_col + NNwinColM1S2] = k2i * k4r - k2r * k4i
                        M_out[C433][lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i
                        M_out[C434_re][lig][col - OOff_col + NNwinColM1S2] = k3r * k4r + k3i * k4i
                        M_out[C434_im][lig][col - OOff_col + NNwinColM1S2] = k3i * k4r - k3r * k4i
                        M_out[C444][lig][col - OOff_col + NNwinColM1S2] = k4r * k4r + k4i * k4i

        # READING NLIG LINES
        for lig in range( Sub_NNlig + NNwinLigM1S2):
            if NNbBlock == 1:
                if lig%(int)((Sub_NNlig+NNwinLigM1S2)/20) == 0:
                    print("%f\r", 100. * lig / (Sub_NNlig+NNwinLigM1S2 - 1), end="", flush = True)

            # 1 line reading with zero padding
            if lig < Sub_NNlig:
                for Np in range(NNpolarIn):
                    _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)
            else:
                if NNblock == (NNbBlock - 1):
                    for Np in range(NNpolarIn):
                        for col in range(2 * NNcol):
                            _MC_in[Np][col] = 0.
                else:
                    for Np in range(NNpolarIn):
                        _MC_in[Np][0] = np.fromfile(datafile[Np], dtype=np.float32, count=2 * NNcol)

            for Np in range( NNpolar):
                for col in range(Sub_NNcol + NNwinCol):
                    M_out[Np][NNwinLigM1S2+lig][col] = 0.

            # Row-wise shift
            for col in range(Sub_NNcol + OOff_col):
                if PolType == "IPPpp4":
                    k1r = _MC_in[hh][2*col]; k1i = _MC_in[hh][2*col + 1];
                    k2r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.);
                    k2i = (_MC_in[hv][2*col + 1] + _MC_in[vh][2*col + 1]) / sqrt(2.);
                    k3r = _MC_in[vv][2*col]; k3i = _MC_in[vv][2*col + 1];
                    M_out[0][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[1][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                    M_out[2][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i;
                if PolType == "IPPpp5":
                    k1r = _MC_in[hh][2*col]; k1i = _MC_in[hh][2*col+1];
                    k2r = _MC_in[vh][2*col]; k2i = _MC_in[vh][2*col+1];
                    M_out[0][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[1][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                if PolType == "IPPpp6":
                    k1r = _MC_in[vv][2*col]; k1i = _MC_in[vv][2*col+1];
                    k2r = _MC_in[hv][2*col]; k2i = _MC_in[hv][2*col+1];
                    M_out[0][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[1][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                if PolType == "IPPpp7":
                    k1r = _MC_in[hh][2*col]; k1i = _MC_in[hh][2*col+1];
                    k2r = _MC_in[vv][2*col]; k2i = _MC_in[vv][2*col+1];
                    M_out[0][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[1][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                if PolType == "T3":
                    k1r = (_MC_in[hh][2*col] + _MC_in[vv][2*col]) / sqrt(2.);
                    k1i = (_MC_in[hh][2*col+1] + _MC_in[vv][2*col+1]) / sqrt(2.);
                    k2r = (_MC_in[hh][2*col] - _MC_in[vv][2*col]) / sqrt(2.);
                    k2i = (_MC_in[hh][2*col+1] - _MC_in[vv][2*col+1]) / sqrt(2.);
                    k3r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.);
                    k3i = (_MC_in[hv][2*col+1] + _MC_in[vh][2*col+1]) / sqrt(2.);

                    M_out[T311][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[T312_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i;
                    M_out[T312_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i;
                    M_out[T313_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i;
                    M_out[T313_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i;
                    M_out[T322][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                    M_out[T323_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i;
                    M_out[T323_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i;
                    M_out[T333][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i;
                if PolType == "T4":
                    k1r = (_MC_in[hh][2*col] + _MC_in[vv][2*col]) / sqrt(2.);
                    k1i = (_MC_in[hh][2*col+1] + _MC_in[vv][2*col+1]) / sqrt(2.);
                    k2r = (_MC_in[hh][2*col] - _MC_in[vv][2*col]) / sqrt(2.);
                    k2i = (_MC_in[hh][2*col+1] - _MC_in[vv][2*col+1]) / sqrt(2.);
                    k3r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.);
                    k3i = (_MC_in[hv][2*col+1] + _MC_in[vh][2*col+1]) / sqrt(2.);
                    k4r = (_MC_in[vh][2*col+1] - _MC_in[hv][2*col+1]) / sqrt(2.);
                    k4i = (_MC_in[hv][2*col] - _MC_in[vh][2*col]) / sqrt(2.);

                    M_out[T411][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[T412_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i;
                    M_out[T412_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i;
                    M_out[T413_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i;
                    M_out[T413_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i;
                    M_out[T414_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k4r + k1i * k4i;
                    M_out[T414_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k4r - k1r * k4i;
                    M_out[T422][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                    M_out[T423_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i;
                    M_out[T423_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i;
                    M_out[T424_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k4r + k2i * k4i;
                    M_out[T424_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2i * k4r - k2r * k4i;
                    M_out[T433][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i;
                    M_out[T434_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k4r + k3i * k4i;
                    M_out[T434_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3i * k4r - k3r * k4i;
                    M_out[T444][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k4r * k4r + k4i * k4i;
                if PolType == "C3":
                    k1r = _MC_in[hh][2*col];
                    k1i = _MC_in[hh][2*col + 1];
                    k2r = (_MC_in[hv][2*col] + _MC_in[vh][2*col]) / sqrt(2.);
                    k2i = (_MC_in[hv][2*col + 1] + _MC_in[vh][2*col + 1]) / sqrt(2.);
                    k3r = _MC_in[vv][2*col];
                    k3i = _MC_in[vv][2*col + 1];

                    M_out[C311][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[C312_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i;
                    M_out[C312_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i;
                    M_out[C313_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i;
                    M_out[C313_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i;
                    M_out[C322][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                    M_out[C323_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i;
                    M_out[C323_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i;
                    M_out[C333][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i;
                if PolType == "C4":
                    k1r = _MC_in[hh][2*col];
                    k1i = _MC_in[hh][2*col+1];
                    k2r = _MC_in[hv][2*col];
                    k2i = _MC_in[hv][2*col+1];
                    k3r = _MC_in[vh][2*col];
                    k3i = _MC_in[vh][2*col+1];
                    k4r = _MC_in[vv][2*col];
                    k4i = _MC_in[vv][2*col+1];

                    M_out[C411][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k1r + k1i * k1i;
                    M_out[C412_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k2r + k1i * k2i;
                    M_out[C412_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k2r - k1r * k2i;
                    M_out[C413_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k3r + k1i * k3i;
                    M_out[C413_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k3r - k1r * k3i;
                    M_out[C414_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1r * k4r + k1i * k4i;
                    M_out[C414_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k1i * k4r - k1r * k4i;
                    M_out[C422][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k2r + k2i * k2i;
                    M_out[C423_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k3r + k2i * k3i;
                    M_out[C423_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2i * k3r - k2r * k3i;
                    M_out[C424_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2r * k4r + k2i * k4i;
                    M_out[C424_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k2i * k4r - k2r * k4i;
                    M_out[C433][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k3r + k3i * k3i;
                    M_out[C434_re][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3r * k4r + k3i * k4i;
                    M_out[C434_im][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k3i * k4r - k3r * k4i;
                    M_out[C444][NNwinLigM1S2+lig][col - OOff_col + NNwinColM1S2] = k4r * k4r + k4i * k4i;



@numba.njit(parallel=False)
def read_block_matrix_float_inner_1(_m_in, _nnwinligm1s2, _sub_nncol, _nnwincol):
    for lig in range(_nnwinligm1s2):
        for col in range(_sub_nncol + _nnwincol):
            _m_in[lig][col] = 0.


@numba.njit(parallel=False)
def read_block_matrix_float_inner_2(M_in, VF_in, Sub_NNcol, NNwinCol, NNwinLigM1S2, lig, NNwinColM1S2, OOff_col):
    for col in range(Sub_NNcol + NNwinCol):
        M_in[NNwinLigM1S2 + lig][col] = 0.
    for col in range(Sub_NNcol):
        M_in[NNwinLigM1S2 + lig][col + NNwinColM1S2] = VF_in[col + OOff_col]


@util.enter_exit_func_decorator
def read_block_matrix_float(in_file, M_in, NNblock, NNbBlock, Sub_NNlig, Sub_NNcol, NNwinLig, NNwinCol, OOff_lig, OOff_col, NNcol, VF_in):
    """
    Description : read a block of a binary (float) file
    """
    NNwinLigM1S2 = (NNwinLig - 1) // 2
    # logging.info(f'{NNwinLigM1S2=}')
    NNwinColM1S2 = (NNwinCol - 1) // 2
    # logging.info(f'{NNwinColM1S2=}')
    NNwinLigM1 = (NNwinLig - 1)
    # logging.info(f'{NNwinLigM1=}')
    if NNblock == 0:
        # OFFSET LINES READING
        # logging.info('OFFSET LINES READING')
        for lig in range(OOff_lig):
            # fread(VF_in[0], sizeof(float), NNcol, in_file);
            VF_in = np.fromfile(in_file, dtype=np.float32, count=NNcol)
        # Set the Tmp matrix to 0
        # logging.info('Set the Tmp matrix to 0')
        read_block_matrix_float_inner_1(M_in, NNwinLigM1S2, Sub_NNcol, NNwinCol)
        # 2023-12-19
        # for lig in range(NNwinLigM1S2):
        #     for col in range(Sub_NNcol + NNwinCol):
        #         M_in[lig][col] = 0.

    else:
        # FSEEK NNwinL LINES
        # logging.info('FSEEK NNwinL LINES')
        # my_fseek(in_file, -1, NNwinLigM1, NNcol * sizeof(float));
        util.my_fseek(in_file, -1, NNwinLigM1, NNcol * np.dtype(np.float32).itemsize)
        # FIRST (NNwin+1)/2 LINES READING
        # logging.info(f'FIRST ({NNwin=}+1)/2 LINES READING')
        for lig in range(NNwinLigM1S2):
            VF_in = np.fromfile(in_file, dtype=np.float32, count=NNcol)
            for col in range(Sub_NNcol + NNwinCol):
                M_in[lig][col] = 0.
            for col in range(Sub_NNcol):
                M_in[lig][col + NNwinColM1S2] = VF_in[col + OOff_col]

    # READING NLIG LINES
    for lig in range(Sub_NNlig + NNwinLigM1S2):
        if NNbBlock <= 2:
            util.printf_line(lig, Sub_NNlig + NNwinLigM1S2)

        # 1 line reading with zero padding
        # logging.info(f'{lig=} {Sub_NNlig=}')
        if lig < Sub_NNlig:
            VF_in = np.fromfile(in_file, dtype=np.float32, count=NNcol)
            read_block_matrix_float_inner_2(M_in, VF_in, Sub_NNcol, NNwinCol, NNwinLigM1S2, lig, NNwinColM1S2, OOff_col)
            # 2023-12-19
            # for col in range(Sub_NNcol + NNwinCol):
            #     M_in[NNwinLigM1S2 + lig][col] = 0.
            # for col in range(Sub_NNcol):
            #     M_in[NNwinLigM1S2 + lig][col + NNwinColM1S2] = VF_in[col + OOff_col]
        else:
            # logging.info('lig >= Sub_NNlig:')
            if NNblock == (NNbBlock - 1):
                for col in range(Sub_NNcol + NNwinCol):
                    M_in[NNwinLigM1S2 + lig][col] = 0.
            else:
                VF_in = np.fromfile(in_file, dtype=np.float32, count=NNcol)
                read_block_matrix_float_inner_2(M_in, VF_in, Sub_NNcol, NNwinCol, NNwinLigM1S2, lig, NNwinColM1S2, OOff_col)
                # 2023-12-19
                # for col in range(Sub_NNcol + NNwinCol):
                #     M_in[NNwinLigM1S2 + lig][col] = 0.
                # for col in range(Sub_NNcol):
                #     M_in[NNwinLigM1S2 + lig][col + NNwinColM1S2] = VF_in[col + OOff_col]


# %% [codecell] write_block_matrix3d_float
def write_block_matrix3d_float(
    datafile, nn_polar, m_out, sub_nnlig, sub_nncol, ooff_lig, ooff_col, nn_col
):
    for n in range(nn_polar):
        for lig in range(sub_nnlig):
            m_out[n][ooff_lig + lig][ooff_col : ooff_col + sub_nncol].tofile(
                datafile[n]
            )

def write_block_matrix_matrix3d_float(out_file, M_out, NNp, Sub_NNlig, Sub_NNcol, OOffLig, OOffCol, NNcol):
    eps = 1.E-30
    for lig in range(Sub_NNlig):
        data = M_out[NNp][OOffLig + lig][OOffCol:OOffCol+Sub_NNcol]
        data = np.where(np.isfinite(data), data, eps)
        out_file.write(data.tobytes())

@util.enter_exit_func_decorator
def write_block_matrix_float(out_file, m_out, Sub_NNlig, Sub_NNcol, OOffLig, OOffCol, NNcol):
    """
    Description : write a block of binary (float) file
    """
    for lig in range(Sub_NNlig):
        data = m_out[OOffLig + lig][OOffCol : Sub_NNcol]
        out_file.write(data.tobytes())
        # data.tofile(out_file)


@numba.njit(parallel=False, fastmath=True)
def average_tci(m_in, valid, NNpolar, m_avg, lig, sub_nn_col, NNwinLig, NNwinCol, NNwinLigM1S2, NNwinColM1S2):
    """Description : Apply a spatial averaging on one line of a TCI matrix"""
    mean = matrix.vector_float(NNpolar)
    NNvalid = 0.
    col_minus_1 = 0
    nn_win_col_minus_1_plus_col = 0
    NNwinLigM1S2_pls_lig = NNwinLigM1S2 + lig
    for col in range(sub_nn_col):
        col_minus_1 = col - 1
        nn_win_col_minus_1_plus_col = NNwinCol - 1 + col
        if col == 0:
            NNvalid = 0.
            for k in range(-NNwinLigM1S2, 1 + NNwinLigM1S2):
                for l in range(-NNwinColM1S2, 1 + NNwinColM1S2):
                    idxY = NNwinLigM1S2_pls_lig + k
                    idXy = NNwinColM1S2 + col + l
                    for Np in range(NNpolar):
                        mean[Np] = mean[Np] + m_in[Np][idxY][idXy] * valid[idxY][idXy]
                    NNvalid = NNvalid + valid[idxY][idXy]
        else:
            for k in range(-NNwinLigM1S2, 1 + NNwinLigM1S2):
                idxY = NNwinLigM1S2_pls_lig + k
                for Np in range(NNpolar):
                    mean[Np] = mean[Np] - m_in[Np][idxY][col_minus_1] * valid[idxY][col_minus_1]
                    mean[Np] = mean[Np] + m_in[Np][idxY][nn_win_col_minus_1_plus_col] * valid[idxY][nn_win_col_minus_1_plus_col]
                NNvalid = NNvalid - valid[idxY][col_minus_1] + valid[idxY][nn_win_col_minus_1_plus_col]
        if NNvalid != 0.:
            for Np in range(NNpolar):
                m_avg[Np][col] = mean[Np] / NNvalid
