"""
Polsarpro
===
util
"""
# %% [codecell] import
import os
import numpy as np

# /* S2 matrix */
S11 = 0
S12 = 1
S21 = 2
S22 = 3

# /* IPP Full */
I411 = 0
I412 = 1
I421 = 2
I422 = 3

# /* IPP pp4 */
I311 = 0
I312 = 1
I322 = 2

# /* IPP pp5-pp6-pp7 */
I211 = 0
I212 = 1

# /* C2 matrix */
C211 = 0
C212_RE = 1
C212_IM = 2
C222 = 3

# /* C3 matrix */
C311 = 0
C312_RE = 1
C312_IM = 2
C313_RE = 3
C313_IM = 4
C322 = 5
C323_RE = 6
C323_IM = 7
C333 = 8

# /* C4 matrix */
C411 = 0
C412_RE = 1
C412_IM = 2
C413_RE = 3
C413_IM = 4
C414_RE = 5
C414_IM = 6
C422 = 7
C423_RE = 8
C423_IM = 9
C424_RE = 10
C424_IM = 11
C433 = 12
C434_RE = 13
C434_IM = 14
C444 = 15

# /* T2 matrix */
T211 = 0
T212_RE = 1
T212_IM = 2
T222 = 3

# /* T3 matrix */
T311 = 0
T312_RE = 1
T312_IM = 2
T313_RE = 3
T313_IM = 4
T322 = 5
T323_RE = 6
T323_IM = 7
T333 = 8

# /* T4 matrix */
T411 = 0
T412_RE = 1
T412_IM = 2
T413_RE = 3
T413_IM = 4
T414_RE = 5
T414_IM = 6
T422 = 7
T423_RE = 8
T423_IM = 9
T424_RE = 10
T424_IM = 11
T433 = 12
T434_RE = 13
T434_IM = 14
T444 = 15

# /* C2 or T2 matrix */
X211 = 0
X212_RE = 1
X212_IM = 2
X222 = 3
X212 = 4

# /* C3 or T3 matrix */
X311 = 0
X312_RE = 1
X312_IM = 2
X313_RE = 3
X313_IM = 4
X322 = 5
X323_RE = 6
X323_IM = 7
X333 = 8
X312 = 9
X313 = 10
X323 = 11

# /* C4 or T4 matrix */
X411 = 0
X412_RE = 1
X412_IM = 2
X413_RE = 3
X413_IM = 4
X414_RE = 5
X414_IM = 6
X422 = 7
X423_RE = 8
X423_IM = 9
X424_RE = 10
X424_IM = 11
X433 = 12
X434_RE = 13
X434_IM = 14
X444 = 15
X412 = 16
X413 = 17
X414 = 18
X423 = 19
X424 = 20
X434 = 21

# /* T6 matrix */
T611 = 0
T612_RE = 1
T612_IM = 2
T613_RE = 3
T613_IM = 4
T614_RE = 5
T614_IM = 6
T615_RE = 7
T615_IM = 8
T616_RE = 9
T616_IM = 10
T622 = 11
T623_RE = 12
T623_IM = 13
T624_RE = 14
T624_IM = 15
T625_RE = 16
T625_IM = 17
T626_RE = 18
T626_IM = 19
T633 = 20
T634_RE = 21
T634_IM = 22
T635_RE = 23
T635_IM = 24
T636_RE = 25
T636_IM = 26
T644 = 27
T645_RE = 28
T645_IM = 29
T646_RE = 30
T646_IM = 31
T655 = 32
T656_RE = 33
T656_IM = 34
T666 = 35

vc_in = None
vf_in = None
mc_in = None
mf_in = None

# %% [codecell] pol_type_config
def pol_type_config(pol_type):
    """Check the polarimetric format configuration"""
    config = False
    pol_type_config = [
        "C2",
        "C2T2",
        "C3",
        "C3T3",
        "C4",
        "C4T4",
        "C4C3",
        "C4T3",
        "T2",
        "T2C2",
        "T3",
        "T3C3",
        "T4",
        "T4C4",
        "T4C3",
        "T4T3",
        "T6",
        "S2SPPpp1",
        "S2SPPpp2",
        "S2SPPpp3",
        "S2IPPpp4",
        "S2IPPpp5",
        "S2IPPpp6",
        "S2IPPpp7",
        "S2IPPfull",
        "S2",
        "S2C3",
        "S2C4",
        "S2T3",
        "S2T4",
        "S2T6",
        "SPP",
        "SPPC2",
        "SPPT2",
        "SPPT4",
        "SPPIPP",
        "IPP",
        "Ixx",
        "S2C2pp1",
        "S2C2pp2",
        "S2C2pp3",
        "S2SPPlhv",
        "S2SPPrhv",
        "S2SPPpi4",
        "S2C2lhv",
        "S2C2rhv",
        "S2C2pi4",
        "C2IPPpp5",
        "C2IPPpp6",
        "C2IPPpp7",
        "C3C2pp1",
        "C3C2pp2",
        "C3C2pp3",
        "C3C2lhv",
        "C3C2rhv",
        "C3C2pi4",
        "C3IPPpp4",
        "C3IPPpp5",
        "C3IPPpp6",
        "C3IPPpp7",
        "C4C2pp1",
        "C4C2pp2",
        "C4C2pp3",
        "C4C2lhv",
        "C4C2rhv",
        "C4C2pi4",
        "C4IPPpp4",
        "C4IPPpp5",
        "C4IPPpp6",
        "C4IPPpp7",
        "C4IPPfull",
        "T3C2pp1",
        "T3C2pp2",
        "T3C2pp3",
        "T3C2lhv",
        "T3C2rhv",
        "T3C2pi4",
        "T3IPPpp4",
        "T3IPPpp5",
        "T3IPPpp6",
        "T3IPPpp7",
        "T4C2pp1",
        "T4C2pp2",
        "T4C2pp3",
        "T4C2lhv",
        "T4C2rhv",
        "T4C2pi4",
        "T4IPPpp4",
        "T4IPPpp5",
        "T4IPPpp6",
        "T4IPPpp7",
        "T4IPPfull",
    ]

    if pol_type in pol_type_config:
        config = True

    # for ii in range(87):
    #     if PolTypeConfig[ii] == PolType:
    #         break

    if not config:
        raise ValueError("A processing error occurred!\nWrong Input / Output Polarimetric Data Format\nUsageHelpDataFormat\n")

    pol_type_tmp = pol_type

    if pol_type_tmp == "C2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type_in = "C2"
        pol_type_out = "C2"

    elif pol_type_tmp == "C2T2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "C2"
        pol_type_in = "C2"
        pol_type_out = "T2"

    elif pol_type_tmp == "C2IPPpp5":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "C2"
        pol_type_in = "C2"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "C2IPPpp6":
        n_polar_in = 4
        n_polar_out = 2
        pol_type_tmp = "C2"
        pol_type_in = "C2"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "C2IPPpp7":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "C2"
        pol_type_in = "C2"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "C3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type_in = "C3"
        pol_type_out = "C3"

    elif pol_type_tmp == "C3T3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "T3"

    elif pol_type_tmp == "C3C2pp1":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "C3C2pp2":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "C3C2pp3":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "C3C2lhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "C3C2rhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "C3C2pi4":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "C3IPPpp4":
        n_polar_in = 9
        n_polar_out = 3
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "C3IPPpp5":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "C3IPPpp6":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "C3IPPpp7":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "C3"
        pol_type_in = "C3"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "C4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type_in = "C4"
        pol_type_out = "C4"

    elif pol_type_tmp == "C4T4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "T4"

    elif pol_type_tmp == "C4C3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C3"

    elif pol_type_tmp == "C4T3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "T3"

    elif pol_type_tmp == "C4C2pp1":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "C4C2pp2":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "C4C2pp3":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "C4C2lhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "C4C2rhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2rhv"
    
    elif pol_type_tmp == "C4C2pi4":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "C4IPPpp4":
        n_polar_in = 16
        n_polar_out = 3
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "C4IPPpp5":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "C4IPPpp6":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "C4IPPpp7":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "C4IPPfull":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "C4"
        pol_type_in = "C4"
        pol_type_out = "IPPfull"

    elif pol_type_tmp == "T2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type_in = "T2"
        pol_type_out = "T2"

    elif pol_type_tmp == "T2C2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "T2"
        pol_type_in = "T2"
        pol_type_out = "C2"

    elif pol_type_tmp == "T3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type_in = "T3"
        pol_type_out = "T3"

    elif pol_type_tmp == "T3C3":
        n_polar_in = 9
        n_polar_out = 9
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C3"

    elif pol_type_tmp == "T3C2pp1":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "T3C2pp2":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "T3C2pp3":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "T3C2lhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "T3C2rhv":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "T3C2pi4":
        n_polar_in = 9
        n_polar_out = 4
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "T3IPPpp4":
        n_polar_in = 9
        n_polar_out = 3
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "T3IPPpp5":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "T3IPPpp6":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "T3IPPpp7":
        n_polar_in = 9
        n_polar_out = 2
        pol_type = "T3"
        pol_type_in = "T3"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "T4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type_in = "T4"
        pol_type_out = "T4"

    elif pol_type_tmp == "T4C4":
        n_polar_in = 16
        n_polar_out = 16
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C4"

    elif pol_type_tmp == "T4C3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C3"

    elif pol_type_tmp == "T4T3":
        n_polar_in = 16
        n_polar_out = 9
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "T3"

    elif pol_type_tmp == "T4C2pp1":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "T4C2pp2":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "T4C2pp3":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "T4C2lhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "T4C2rhv":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "T4C2pi4":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "T4IPPpp4":
        n_polar_in = 16
        n_polar_out = 3
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "T4IPPpp5":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "T4IPPpp6":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "T4IPPpp7":
        n_polar_in = 16
        n_polar_out = 2
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "T4IPPfull":
        n_polar_in = 16
        n_polar_out = 4
        pol_type = "T4"
        pol_type_in = "T4"
        pol_type_out = "IPPfull"

    elif pol_type_tmp == "T6":
        n_polar_in = 36
        n_polar_out = 36
        pol_type_in = "T6"
        pol_type_out = "T6"

    elif pol_type_tmp == "S2SPPpp1":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpp1"

    elif pol_type_tmp == "S2SPPpp2":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpp2"

    elif pol_type_tmp == "S2SPPpp3":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpp3"

    elif pol_type_tmp == "S2C2pp1":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pp1"

    elif pol_type_tmp == "S2C2pp2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pp2"

    elif pol_type_tmp == "S2C2pp3":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pp3"

    elif pol_type_tmp == "S2SPPlhv":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPlhv"

    elif pol_type_tmp == "S2SPPrhv":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPrhv"

    elif pol_type_tmp == "S2SPPpi4":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "SPPpi4"

    elif pol_type_tmp == "S2C2lhv":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2lhv"

    elif pol_type_tmp == "S2C2rhv":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2rhv"

    elif pol_type_tmp == "S2C2pi4":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C2pi4"

    elif pol_type_tmp == "S2IPPpp4":
        n_polar_in = 4
        n_polar_out = 3
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp4"

    elif pol_type_tmp == "S2IPPpp5":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp5"

    elif pol_type_tmp == "S2IPPpp6":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp6"

    elif pol_type_tmp == "S2IPPpp7":
        n_polar_in = 4
        n_polar_out = 2
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPpp7"

    elif pol_type_tmp == "S2IPPfull":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "IPPfull"

    elif pol_type_tmp == "S2":
        n_polar_in = 4
        n_polar_out = 4
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "S2"

    elif pol_type_tmp == "S2C3":
        n_polar_in = 4
        n_polar_out = 9
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C3"

    elif pol_type_tmp == "S2C4":
        n_polar_in = 4
        n_polar_out = 16
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "C4"

    elif pol_type_tmp == "S2T3":
        n_polar_in = 4
        n_polar_out = 9
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "T3"

    elif pol_type_tmp == "S2T4":
        n_polar_in = 4
        n_polar_out = 16
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "T4"

    elif pol_type_tmp == "S2T6":
        n_polar_in = 4
        n_polar_out = 36
        pol_type = "S2"
        pol_type_in = "S2"
        pol_type_out = "T6"

    # elif pol_type_tmp == "SPP":
    #     n_polar_in = 2
    #     n_polar_out = 2
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "SPP" + polar_type

    # elif pol_type_tmp == "SPPC2":
    #     n_polar_in = 2
    #     n_polar_out = 4
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "C2" + polar_type

    # elif pol_type_tmp == "SPPT2":
    #     n_polar_in = 2
    #     n_polar_out = 4
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "T2" + polar_type

    # elif pol_type_tmp == "SPPT4":
    #     n_polar_in = 2
    #     n_polar_out = 16
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "T4"

    # elif pol_type_tmp == "SPPIPP":
    #     n_polar_in = 2
    #     n_polar_out = 2
    #     pol_type = "SPP"
    #     pol_type_in = "SPP" + polar_type
    #     pol_type_out = "IPP"
    #     if polar_type == "pp1":
    #         pol_type_out += "pp5"
    #     elif polar_type == "pp2":
    #         pol_type_out += "pp6"
    #     elif polar_type == "pp3":
    #         pol_type_out += "pp7"

    # elif pol_type_tmp == "IPP":
    #     if polar_type == "full":
    #         n_polar_in = 4
    #         n_polar_out = 4
    #     elif polar_type in ["pp4", "pp5", "pp6", "pp7"]:
    #         n_polar_in = 2
    #         n_polar_out = 2
    #     pol_type_in = "IPP" + polar_type
    #     pol_type_out = "IPP" + polar_type

    elif pol_type_tmp == "Ixx":
        n_polar_in = 1
        n_polar_out = 1
        pol_type_in = "Ixx"
        pol_type_out = "Ixx"

    return pol_type, n_polar_in, pol_type_in, n_polar_out, pol_type_out

# %% [codecell] init_file_name
def init_file_name(pol_type, file_dir):
    """Initialisation of the binary file names"""
    fime_name = []
    file_name_c2 = ["C11.bin", "C12_real.bin", "C12_imag.bin", "C22.bin"]

    file_name_c3 = [
        "C11.bin",
        "C12_real.bin",
        "C12_imag.bin",
        "C13_real.bin",
        "C13_imag.bin",
        "C22.bin",
        "C23_real.bin",
        "C23_imag.bin",
        "C33.bin",
    ]

    file_name_c4 = [
        "C11.bin",
        "C12_real.bin",
        "C12_imag.bin",
        "C13_real.bin",
        "C13_imag.bin",
        "C14_real.bin",
        "C14_imag.bin",
        "C22.bin",
        "C23_real.bin",
        "C23_imag.bin",
        "C24_real.bin",
        "C24_imag.bin",
        "C33.bin",
        "C34_real.bin",
        "C34_imag.bin",
        "C44.bin",
    ]

    file_name_t3 = [
        "T11.bin",
        "T12_real.bin",
        "T12_imag.bin",
        "T13_real.bin",
        "T13_imag.bin",
        "T22.bin",
        "T23_real.bin",
        "T23_imag.bin",
        "T33.bin",
    ]

    if pol_type == "T3":
        for i in range(9):
            fime_name.append(f"{file_dir}{file_name_t3[i]}")

    return fime_name

# %% [codecell] read_config
def read_config(file_dir):
    """Read a configuration file"""
    nlig = 0
    ccol = 0
    polar_case = ""
    polar_type = ""

    if os.path.exists(os.path.join(file_dir, "config.txt")):
        with open(file_dir + "//config.txt", "r") as file:
            for line in file:
                if "Nrow" in line:
                    nlig = int(file.readline().strip())
                elif "Ncol" in line:
                    ccol = int(file.readline().strip())
                elif "PolarCase" in line:
                    polar_case = file.readline().strip()
                elif "PolarType" in line:
                    polar_type = file.readline().strip()
    return nlig, ccol, polar_case, polar_type

# %% [codecell] my_fseek
def my_fseek(in_file, fseek_sign, fseek_arg1, fseek_arg2):
    """Function fseek with a pointer size higher than LONG_MAX"""
    in_file.seek(fseek_sign * fseek_arg1 * fseek_arg2, 1)
