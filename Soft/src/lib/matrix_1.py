import numpy as np

def matrix3d_float(nz, nrh, nch):
    m = np.zeros((nz, nrh, nch), dtype=np.float32)
    return m

def matrix_float(nz, nrh):
    m = np.zeros((nz, nrh), dtype=np.float32)
    return m