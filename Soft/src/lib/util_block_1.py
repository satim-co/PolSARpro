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
import util

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

# %% [codecell] read_block_TCI_noavg
def read_block_TCI_noavg(
    datafile,
    m_out,
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
    nn_win_lig_m1s2 = int((nn_win_lig - 1) / 2)
    nn_win_col_m1s2 = int((nn_win_col - 1) / 2)
    nn_win_lig_m1 = nn_win_lig - 1

    if nn_block == 0:
        # OFFSET LINES READING
        for lig in range(ooff_lig):
            for n in range(nn_polar):
                util.vf_in = np.fromfile(datafile[n], dtype=np.float32, count=nn_col)
        # Set the Tmp matrix to 0
        for lig in range(nn_win_lig_m1s2):
            for col in range(sub_nncol + nn_win_col):
                for Np in range(nn_polar):
                    m_out[Np][lig][col] = 0.

    else:
        # FSEEK NNwinL LINES
        for n in range(nn_polar):
            util.my_fseek(datafile[n], -1, nn_win_lig_m1, nn_col * 4)
        for lig in range(nn_win_lig_m1s2):
            for n in range(nn_polar):
                util.vf_in = np.fromfile(datafile[n], dtype=np.float32, count=nn_col)
                for col in range(sub_nncol + nn_win_col):
                    m_out[n][lig][col] = 0.0
                for col in range(sub_nncol):
                    m_out[n][lig][col + nn_win_col_m1s2] = util.vf_in[col + ooff_col]
    # /* READING NLIG LINES */
    for lig in range(sub_nnlig + nn_win_lig_m1s2):
        if nn_bblock == 1 and lig % ((sub_nnlig + nn_win_lig_m1s2) // 20) == 0:
            print(
                "{:.2f}%\r".format(100.0 * lig / (sub_nnlig + nn_win_lig_m1s2 - 1)),
                end="",
            )

        # 1 line reading with zero padding
        for n in range(nn_polar):
            if lig < sub_nnlig:
                util.vf_in = np.fromfile(datafile[n], dtype=np.float32, count=nn_col)
                for col in range(sub_nncol + nn_win_col):
                    m_out[n][nn_win_lig_m1s2 + lig][col] = 0.0
                for col in range(sub_nncol):
                    m_out[n][nn_win_lig_m1s2 + lig][
                        col + nn_win_col_m1s2
                    ] = util.vf_in[col + ooff_col]
            else:
                if nn_block == nn_bblock - 1:
                    for col in range(sub_nncol + nn_win_col):
                        m_out[n][nn_win_lig_m1s2 + lig][col] = 0.0
                else:
                    util.vf_in = np.fromfile(
                        datafile[n], dtype=np.float32, count=nn_col
                    )
                    for col in range(sub_nncol + nn_win_col):
                        m_out[n][nn_win_lig_m1s2 + lig][col] = 0.0
                    for col in range(sub_nncol):
                        m_out[n][nn_win_lig_m1s2 + lig][
                            col + nn_win_col_m1s2
                        ] = util.vf_in[col + ooff_col]
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

# %% [codecell] read_block_matrix_float
def read_block_matrix_float(
    in_file,
    m_in,
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
    nn_win_lig_m1s2 = (nn_win_lig - 1) // 2
    nn_win_col_m1s2 = (nn_win_col - 1) // 2
    nn_win_lig_m1 = nn_win_lig - 1

    if nn_block == 0:
        # OFFSET LINES READING
        for lig in range(ooff_lig):
            in_file.read(nn_col * 4)  # size of float is 4 bytes

        # Set the Tmp matrix to 0
        m_in = np.zeros((nn_win_lig_m1s2, sub_nncol + nn_win_col))
    else:
        # FSEEK NNwinL LINES
        in_file.seek(-1 * nn_win_lig_m1 * nn_col * 4, 1)
        # FIRST (NNwin+1)/2 LINES READING
        m_in = np.zeros((nn_win_lig_m1s2, sub_nncol + nn_win_col))
        for lig in range(nn_win_lig_m1s2):
            m_in[lig, nn_win_col_m1s2 : nn_win_col_m1s2 + sub_nncol] = util.vf_in[
                ooff_col : ooff_col + sub_nncol
            ]

    #   /* READING NLIG LINES */
    for lig in range(sub_nnlig + nn_win_lig_m1s2):
        if nn_bblock <= 2:
            print(f"Reading line {lig} of {sub_nnlig + nn_win_lig_m1s2}")

        # /* 1 line reading with zero padding */
        if lig < sub_nnlig:
            m_in[
                nn_win_lig_m1s2 + lig, nn_win_col_m1s2 : nn_win_col_m1s2 + sub_nncol
            ] = util.vf_in[ooff_col : ooff_col + sub_nncol]
        else:
            m_in[
                nn_win_lig_m1s2 + lig, nn_win_col_m1s2 : nn_win_col_m1s2 + sub_nncol
            ] = util.vf_in[ooff_col : ooff_col + sub_nncol]
    return m_in

def average_tci(M_in, Valid, NNpolar, M_avg, lig, Sub_NNcol, NNwinLig, NNwinCol, NNwinLigM1S2, NNwinColM1S2):
    mean = np.zeros(NNpolar)
    NNvalid = 0
    for col in range(Sub_NNcol):
        if col == 0:
            NNvalid = 0
            for k in range(-NNwinLigM1S2, 1 + NNwinLigM1S2):
                for l in range(-NNwinColM1S2, 1 + NNwinColM1S2):
                    idxY = NNwinLigM1S2 + lig + k
                    idXy = NNwinColM1S2 + col + l
                    for Np in range(NNpolar):
                        mean[Np] += M_in[Np][idxY][idXy] * Valid[idxY][idXy]
                    NNvalid += Valid[idxY][idXy]
        else:
            for k in range(-NNwinLigM1S2, 1 + NNwinLigM1S2):
                idxY = NNwinLigM1S2 + lig + k
                for Np in range(NNpolar):
                    mean[Np] -= M_in[Np][idxY][col-1] * Valid[idxY][col-1]
                    mean[Np] += M_in[Np][idxY][NNwinCol-1+col] * Valid[idxY][NNwinCol-1+col]
                NNvalid = NNvalid - Valid[idxY][col-1] + Valid[idxY][NNwinCol-1+col]

        if NNvalid != 0:
            for Np in range(NNpolar):
                M_avg[Np][col] = mean[Np] / NNvalid
    return M_avg

# %% [codecell] write_block_matrix3d_float
def write_block_matrix3d_float(datafile, nn_polar, m_out, sub_nnlig, sub_nncol, ooff_lig, ooff_col, NNcol):
    for n in range(nn_polar):
        for lig in range(sub_nnlig):
            m_out[n][ooff_lig + lig][ooff_col : ooff_col + sub_nncol].tofile(
                datafile[n]
            )

def write_block_matrix_float(out_file, M_out, Sub_NNlig, Sub_NNcol, OOffLig, OOffCol, NNcol):
    for lig in range(Sub_NNlig):
        # Extract the relevant row slice
        row_slice = M_out[OOffLig + lig][OOffCol:OOffCol + Sub_NNcol]

        # Convert the slice to bytes and write to the file
        row_bytes = bytearray()
        for value in row_slice:
            row_bytes.extend(value.to_bytes(4, byteorder='little', signed=False))
        
        out_file.write(row_bytes)