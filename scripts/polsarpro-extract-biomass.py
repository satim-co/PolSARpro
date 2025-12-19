# -*- coding: utf-8 -*-

"""
/********************************************************************
PolSARpro v6.0 is free software; you can redistribute it and/or 
modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation; either version 2 (1991) of
the License, or any later version. This program is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. 

See the GNU General Public License (Version 2, 1991) for more details

*********************************************************************

File   : PolSARproBiomassExtractData.py
Project  : ESA_POLSARPRO
Authors  : Eric POTTIER
Version  : 1.0
Creation : 12/2024
Update  :
*--------------------------------------------------------------------
INSTITUT D'ELECTRONIQUE et de TELECOMMUNICATIONS de RENNES (I.E.T.R)
UMR CNRS 6164

Waves and Signal department
SHINE Team 

UNIVERSITY OF RENNES I
Bât. 11D - Campus de Beaulieu
263 Avenue Général Leclerc
35042 RENNES Cedex
Tel :(+33) 2 23 23 57 63
Fax :(+33) 2 23 23 69 63
e-mail: eric.pottier@univ-rennes1.fr

*--------------------------------------------------------------------

Description :  Convert and extract BIOMASS full resolution 
               polarimetric binary data files (format SLC) from 
               the two Cloud Optimized GeoTIFF (format COG) files

********************************************************************/
"""

import sys, json, os
import numpy as np
import matplotlib
import rasterio
from pathlib import Path

def extractFullResolutionCOG(biomassImagePath):

    # biomassPath = biomassImagePath.lower()
    # biomassPath = biomassPath[-80:]
    # biomassPath = biomassPath[:70]

    str_tiff_abs = f"**/measurement/*_abs.tiff"
    str_tiff_phase = f"**/measurement/*_phase.tiff"

    biomassPath = Path(biomassImagePath)
    biomassDataAbsPath = list(biomassPath.glob(str_tiff_abs))[0]
    # biomassDataAbsPath = os.path.join(biomassImagePath,"measurement",biomassPath + "_i_abs.tiff")
    biomassDataPhasePath = list(biomassPath.glob(str_tiff_phase))[0] 
    # biomassDataPhasePath = os.path.join(biomassImagePath,"measurement",biomassPath + "_i_phase.tiff")
    biomassDataOutputPolSARproPath = os.path.join(biomassImagePath,"polsarpro")

    try:
        # Open the COG Abs file using rasterio
        with rasterio.open(biomassDataAbsPath) as dataAbsSet:
            # Check if the dataAbs file is a valid COG
            if not dataAbsSet.meta.get('driver') == 'GTiff':
                raise ValueError("The file provided is not a GeoTIFF.")

            # Open the COG Phase file using rasterio
            with rasterio.open(biomassDataPhasePath) as dataPhaseSet:
                # Check if the dataPhase file is a valid COG
                if not dataPhaseSet.meta.get('driver') == 'GTiff':
                    raise ValueError("The file provided is not a GeoTIFF.")

                os.makedirs(biomassDataOutputPolSARproPath, exist_ok=True)

                # Read the full-resolution data
                for idx in range(1, dataAbsSet.count + 1):
                    bandAbs = dataAbsSet.read(idx, out_shape=(1, dataAbsSet.height, dataAbsSet.width))
                    bandPhase = dataPhaseSet.read(idx, out_shape=(1, dataPhaseSet.height, dataPhaseSet.width))
                    bandReal = bandAbs * np.cos(bandPhase)
                    bandImag = bandAbs * np.sin(bandPhase)
                    bandCmplx = bandReal + 1j * bandImag
                    if (idx == 1):
                        biomassDataOutputPolSARproComplexFile = os.path.join(biomassDataOutputPolSARproPath,"channelHHfullResolution.bin")
                        print("... decoding channelHH")
                    if (idx == 2):
                        biomassDataOutputPolSARproComplexFile = os.path.join(biomassDataOutputPolSARproPath,"channelHVfullResolution.bin")
                        print("... decoding channelHV")
                    if (idx == 3):
                        biomassDataOutputPolSARproComplexFile = os.path.join(biomassDataOutputPolSARproPath,"channelVHfullResolution.bin")
                        print("... decoding channelVH")
                    if (idx == 4):
                        biomassDataOutputPolSARproComplexFile = os.path.join(biomassDataOutputPolSARproPath,"channelVVfullResolution.bin")
                        print("... decoding channelVV")
                    bandCmplx.tofile(biomassDataOutputPolSARproComplexFile)

    except Exception as e:
        print(f"An error occurred: {e}")

# use
if __name__ == "__main__":
    ###
    # The user can run this (c) Python code either :
    # - from the command line (terminal) with :
    # $> python PolSARproBiomassExtractData.py "c:\Full path to the BIOMASS folder to be processed"
    ###
    biomassImagePath = sys.argv[1]
    ###
    # - from a Jupyter Notebook in a cell with :
    # biomassImagePath = "c:\Full path to the BIOMASS folder to be processed"
    ###
    
    extractFullResolutionCOG(biomassImagePath)