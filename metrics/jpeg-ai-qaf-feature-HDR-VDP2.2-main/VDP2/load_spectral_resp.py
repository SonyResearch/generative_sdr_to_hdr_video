''' LOAD_SPECTRAL_RESP(intetrnal) load spectral response or emmision curve
    from a comma separated file.The first column must contain wavelengths in
    nm(lambda ) and the remaining columns response/sensitivity/emission
    data.The data is resampled to the 360-780 range with the step of 1nm.

    (C) Rafal Mantiuk < mantiuk @ gmail.com >
    This is an experimental code for internal use.Do not redistribute.

'''

import numpy as np
import csv
from scipy import interpolate
import os


def load_spectral_resp(file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    D = csv.reader(open(os.path.join(current_path, file_name)))
    Emission_Spe = []
    for line in D:
        Emission_Spe.append(line)
    Emission_Spe = np.array(Emission_Spe, dtype=float)
    l_minmax = [360, 780]
    l_step = 1
    lamb = np.linspace(l_minmax[0], l_minmax[1], num=int((l_minmax[1] - l_minmax[0]) / l_step))
    R = np.zeros([len(lamb), Emission_Spe.shape[1] - 1])
    for k in np.arange(1, Emission_Spe.shape[1]):
        R[:, k - 1] = interpolate.interp1d(Emission_Spe[:, 0], Emission_Spe[:, k], kind='cubic', fill_value=0,
                                           bounds_error=False)(
            lamb)

    return [lamb, R]
