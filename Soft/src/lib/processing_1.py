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

def diagonalisation(MatrixDim, HM, EigenVect, EigenVal):

    a = np.zeros((10, 10, 2))
    v = np.zeros((10, 10, 2))
    d = np.zeros((10))
    z = np.zeros((10))
    w = np.zeros((2))
    s = np.zeros((2))
    c = np.zeros((2))
    titi = np.zeros((2))
    gc = np.zeros((2))
    hc = np.zeros((2))

    n = MatrixDim
    pp = 0
    qq = 0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            a[i][j][0] = HM[i - 1][j - 1][0]
            a[i][j][1] = HM[i - 1][j - 1][1]
            v[i][j][0] = 0.
            v[i][j][1] = 0.
        v[i][i][0] = 1.
        v[i][i][1] = 0.

    for pp in range(1, n + 1):
        d[pp] = a[pp][pp][0]
        z[pp] = 0.

    for ii in range(1, 1000 * n * n):
        sm = 0.
        for pp in range(1, n):
            for qq in range(pp + 1, n + 1):
                sm = sm + 2. * np.sqrt(a[pp][qq][0] * a[pp][qq][0] + a[pp][qq][1] * a[pp][qq][1])
        sm = sm / (n * (n - 1))
        if sm < 1.E-16:
            break
        tresh = 1.E-17
        if ii < 4:
            tresh = 0.2 * sm / (n * n)
        x = -1.E-15
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                toto = np.sqrt(a[i][j][0] * a[i][j][0] + a[i][j][1] * a[i][j][1])
                if x < toto:
                    x = toto
                    pp = i
                    qq = j
        if toto > tresh:
            e = d[pp] - d[qq]
            w = [a[pp][qq][0], a[pp][qq][1]]
            g = np.sqrt(w[0] * w[0] + w[1] * w[1])
            g = g * g
            f = np.sqrt(e * e + 4. * g)
            d1 = e + f
            d2 = e - f
            if abs(d2) > abs(d1): 
                d1 = d2
            r = abs(d1) / np.sqrt(d1 * d1 + 4. * g)
            s = [r, 0.]
            titi = [2. * r / d1, 0.]
            c = [titi[0] * w[0] - titi[1] * w[1], titi[0] * w[1] + titi[1] * w[0]]
            r = np.sqrt(s[0] * s[0] + s[1] * s[1])
            r = r * r
            h = (d1 / 2. + 2. * g / d1) * r
            d[pp] -= h
            z[pp] -= h
            d[qq] += h
            z[qq] += h
            a[pp][qq] = [0., 0.]

            for j in range(1, pp):
                gc = [a[j][pp][0], a[j][pp][1]]
                hc = [a[j][qq][0], a[j][qq][1]]
                a[j][pp][0] = c[0] * gc[0] - c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1]
                a[j][pp][1] = c[0] * gc[1] + c[1] * gc[0] - s[0] * hc[1] + s[1] * hc[0]
                a[j][qq][0] = s[0] * gc[0] - s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1]
                a[j][qq][1] = s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0]
            for j in range(pp + 1, qq):
                gc = [a[pp][j][0], a[pp][j][1]]
                hc = [a[j][qq][0], a[j][qq][1]]
                a[pp][j][0] = c[0] * gc[0] + c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1]
                a[pp][j][1] = c[0] * gc[1] - c[1] * gc[0] + s[0] * hc[1] - s[1] * hc[0]
                a[j][qq][0] = s[0] * gc[0] + s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1]
                a[j][qq][1] = -s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0]

            for j in range(qq + 1, n + 1):
                gc = [a[pp][j][0], a[pp][j][1]]
                hc = [a[qq][j][0], a[qq][j][1]]
                a[pp][j][0] = c[0] * gc[0] + c[1] * gc[1] - s[0] * hc[0] + s[1] * hc[1]
                a[pp][j][1] = c[0] * gc[1] - c[1] * gc[0] - s[0] * hc[1] - s[1] * hc[0]
                a[qq][j][0] = s[0] * gc[0] + s[1] * gc[1] + c[0] * hc[0] - c[1] * hc[1]
                a[qq][j][1] = s[0] * gc[1] - s[1] * gc[0] + c[0] * hc[1] + c[1] * hc[0]

            for j in range(1, n + 1):
                gc = [v[j][pp][0], v[j][pp][1]]
                hc = [v[j][qq][0], v[j][qq][1]]
                v[j][pp][0] = c[0] * gc[0] - c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1]
                v[j][pp][1] = c[0] * gc[1] + c[1] * gc[0] - s[0] * hc[1] + s[1] * hc[0]
                v[j][qq][0] = s[0] * gc[0] - s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1]
                v[j][qq][1] = s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0]
    #Sortie
    for k in range(1, n + 1):
        d[k] = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            d[k] += v[i][k][0] * (HM[i - 1][j - 1][0] * v[j][k][0] - HM[i - 1][j - 1][1] * v[j][k][1])
            d[k] += v[i][k][1] * (HM[i - 1][j - 1][0] * v[j][k][1] + HM[i - 1][j - 1][1] * v[j][k][0])
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if d[j] > d[i]:
                x = d[i]
                d[i] = d[j]
                d[j] = x
                for k in range(1, n + 1):
                    c = [v[k][i][0], v[k][i][1]]
                    v[k][i][0] = v[k][j][0]
                    v[k][i][1] = v[k][j][1]
                    v[k][j][0] = c[0]
                    v[k][j][1] = c[1]

    for i in range(n):
        EigenVal[i] = d[i + 1]
        for j in range(n):
            EigenVect[i][j][0] = v[i + 1][j + 1][0]
            EigenVect[i][j][1] = v[i + 1][j + 1][1]
    return EigenVect