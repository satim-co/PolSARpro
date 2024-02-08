'''
PolSARpro v5.0 is free software; you can redistribute it and/or 
modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation; either version 2 (1991) of
the License, or any later version. This program is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. 

See the GNU General Public License (Version 2, 1991) for more details

*********************************************************************

File   : graphics.c
Project  : ESA_POLSARPRO
Authors  : Eric POTTIER, Laurent FERRO-FAMIL
Translate to python: Krzysztof Smaza
Version  : 1.0
Creation : 09/2003
Update  : 12/2006 (Stephane MERIC)

*--------------------------------------------------------------------
INSTITUT D'ELECTRONIQUE et de TELECOMMUNICATIONS de RENNES (I.E.T.R)
UMR CNRS 6164
Groupe Image et Teledetection
Equipe SAPHIR
(SAr Polarimetrie Holographie Interferometrie Radargrammetrie)
UNIVERSITE DE RENNES I
Pôle Micro-Ondes Radar
Bât. 11D - Campus de Beaulieu
263 Avenue Général Leclerc
35042 RENNES Cedex
Tel :(+33) 2 23 23 57 63
Fax :(+33) 2 23 23 69 63
e-mail : 
eric.pottier@univ-rennes1.fr, laurent.ferro-famil@univ-rennes1.fr
'''


import math
import numpy
from . import matrix


def header(nlig, ncol, Max, Min, fbmp):
    '''
    Routine  : header
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *--------------------------------------------------------------------
    Description :  Creates and writes a bitmap file header
    *--------------------------------------------------------------------
    Inputs arguments :
    nlig   : BMP image number of lines
    ncol   : BMP image number of rows
    Max    : Coded Maximum Value
    Min    : Coded Minimum Value
    fbmp  : BMP file pointer
    '''
    # Bitmap File Header
    extracol = (int)(math.fmod(4 - (int)(math.fmod(ncol, 4)), 4))
    fbmp.write(numpy.int16(19778))
    fbmp.write(numpy.int32((ncol + extracol) * nlig + 1078))
    fbmp.write(numpy.int32(0))
    fbmp.write(numpy.int32(1078))
    fbmp.write(numpy.int32(40))
    fbmp.write(numpy.int32(ncol))
    fbmp.write(numpy.int32(nlig))
    fbmp.write(numpy.int16(1))
    fbmp.write(numpy.int16(8))
    fbmp.write(numpy.int32(0))
    fbmp.write(numpy.int32(ncol * nlig))
    fbmp.write(numpy.int32(0))
    fbmp.write(numpy.int32(0))
    fbmp.write(numpy.int32(256))
    fbmp.write(numpy.int32(0))


def write_bmp_hdr(nlig, ncol, Max, Min, nbytes, filebmp):
    '''
    Routine  : write_bmp_hdr
    Authors  : Eric POTTIER
    Creation : 07/2011
    Update   :
    *--------------------------------------------------------------------
    Description :  Write the hdr file of a BMP file
    *--------------------------------------------------------------------
    Inputs arguments :
    nlig    : matrix number of lines
    ncol    : matrix number of rows
    Max    : Maximum value
    Min    : Minimum value
    nbytes : number of bytes (8 or 24)
    filebmp   : BMP file
    '''

    # Bitmap HDR file opening
    filehdr = f'{filebmp}.hdr'
    f_bmp = None
    try:
        f_bmp = open(filehdr, 'wb')
    except IOError:
        print("Could not open the bitmap file: ", filehdr)
        raise
    f_bmp.write('ENVI\n'.encode('ascii'))
    f_bmp.write('description = {\n'.encode('ascii'))
    f_bmp.write('PolSARpro File Imported to ENVI }\n'.encode('ascii'))
    f_bmp.write(f'samples = {ncol}\n'.encode('ascii'))
    f_bmp.write(f'lines   = {nlig}\n'.encode('ascii'))
    f_bmp.write(f'max val = {Max}\n'.encode('ascii'))
    f_bmp.write(f'min val = {Min}\n'.encode('ascii'))
    f_bmp.write(f'color   = {nbytes} bytes\n'.encode('ascii'))


def bmp_wishart(mat, n_lig, n_col, nom, color_map):
    '''
    Routine  : bmp_wishart
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *--------------------------------------------------------------------
    Description :  Creates a bitmap file from a matrix resulting from the wishart
    H / A / Alpha classification
    *--------------------------------------------------------------------
    Inputs arguments :
    mat  : matrix containg float values
    nlig  : matrix number of lines
    ncol  : matrixnumber of rows
    *name : BMP file name (without the .bmp extension)
    '''

    red = numpy.zeros(256, dtype=numpy.int32)
    green = numpy.zeros(256, dtype=numpy.int32)
    blue = numpy.zeros(256, dtype=numpy.int32)

    extracol = (int)(math.fmod(4 - (int)(math.fmod(n_col, 4)), 4))
    n_col_bmp = n_col + extracol
    bufimg = matrix.vector_char(n_lig * n_col_bmp)
    bufcolor = matrix.vector_char(1024)

    nom = f'{nom}.bmp'
    f_bmp = None
    try:
        f_bmp = open(nom, "wb")
    except IOError:
        print("Could not open file: ", nom)
        raise

    my_min = 1.
    my_max = -20.
    for lig in range(n_lig):
        for col in range(n_col):
            if mat[lig][col] > my_max:
                my_max = mat[lig][col]

    header(n_lig, n_col, my_max, my_min, f_bmp)
    write_bmp_hdr(n_lig, n_col, my_max, my_min, 8, nom)

    # Definition of the Colormap
    f_color_map = None
    try:
        f_color_map = open(color_map, "r")
    except IOError:
        print("Could not open the bitmap file: ", color_map)
        raise

    # Colormap Definition
    f_color_map.readline()
    f_color_map.readline()
    n_color = int(f_color_map.readline().strip())
    for k in range(n_color):
        line = f_color_map.readline().strip()
        red[k] = line.split(sep=' ')[0]
        green[k] = line.split(sep=' ')[1]
        blue[k] = line.split(sep=' ')[2]
    f_color_map.close()

    # Bitmap colormap writing
    for col in range(256):
        bufcolor[4 * col] = numpy.byte(blue[col])
        bufcolor[4 * col + 1] = numpy.byte(green[col])
        bufcolor[4 * col + 2] = numpy.byte(red[col])
        bufcolor[4 * col + 3] = numpy.byte(0)
    bufcolor.tofile(f_bmp)

    # Image writing
    for lig in range(n_lig):
        for col in range(n_col):
            l = (int)(mat[n_lig - lig - 1][col])
            bufimg[lig * n_col_bmp + col] = numpy.byte(l)
    bufimg.tofile(f_bmp)
    f_bmp.close()


def bmp_training_set(mat, n_lig, n_col, nom, color_map16):
    '''
    Routine  : bmp_training_set
    Authors  : Eric POTTIER, Laurent FERRO-FAMIL
    Creation : 01/2002
    Update  :
    *--------------------------------------------------------------------
    Description :  Creates a bitmap file of the training areas
    *--------------------------------------------------------------------
    Inputs arguments :
    mat  : matrix containg float values
    nlig  : matrix number of lines
    ncol  : matrixnumber of rows
    *name : BMP file name (without the .bmp extension)
    Returned values  :
    void
    '''

    red = numpy.zeros(256, dtype=numpy.int32)
    green = numpy.zeros(256, dtype=numpy.int32)
    blue = numpy.zeros(256, dtype=numpy.int32)

    extracol = (int)(math.fmod(4 - (int)(math.fmod(n_col, 4)), 4))
    n_col_bmp = n_col + extracol
    bufimg = matrix.vector_char(n_lig * n_col_bmp)
    bufcolor = matrix.vector_char(1024)

    nom = f'{nom}.bmp'
    f_bmp = None
    try:
        f_bmp = open(nom, 'wb')
    except IOError:
        print("Could not open the bitmap file: ", nom)
        raise

    my_min = 1
    my_max = -20
    for lig in range(n_lig):
        for col in range(n_col):
            if mat[lig][col] > my_max:
                my_max = mat[lig][col]

    header(n_lig, n_col, my_max, my_min, f_bmp)
    write_bmp_hdr(n_lig, n_col, my_max, my_min, 8, nom)

    # Definition of the Colormap
    f_color_map = None
    try:
        f_color_map = open(color_map16, 'r')
    except IOError:
        print('Could not open the bitmap file: ', color_map16)
        raise

    # Colormap Definition
    f_color_map.readline()
    f_color_map.readline()
    n_color = int(f_color_map.readline().strip())
    for k in range(n_color):
        line = f_color_map.readline().strip()
        red[k] = line.split(sep=' ')[0]
        green[k] = line.split(sep=' ')[1]
        blue[k] = line.split(sep=' ')[2]
    f_color_map.close()

    # Bitmap colormap writing
    for k in range(n_color):
        bufcolor[4 * k] = numpy.byte(1)
        bufcolor[4 * k + 1] = numpy.byte(1)
        bufcolor[4 * k + 2] = numpy.byte(1)
        bufcolor[4 * k + 3] = numpy.byte(0)

    for k in range(math.floor(my_max)):
        bufcolor[4 * k] = numpy.byte(blue[k])
        bufcolor[4 * k + 1] = numpy.byte(green[k])
        bufcolor[4 * k + 2] = numpy.byte(red[k])
        bufcolor[4 * k + 3] = numpy.byte(0)
    bufcolor.tofile(f_bmp)

    # Image writing
    for lig in range(n_lig):
        for col in range(n_col):
            l = (int)(mat[n_lig - lig - 1][col])
            bufimg[lig * n_col_bmp + col] = numpy.byte(l)
    bufimg.tofile(f_bmp)
    f_bmp.close()

