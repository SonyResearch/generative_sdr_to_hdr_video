import numpy as np
from numpy import trapz
from VDP3.load_spectral_resp import load_spectral_resp


def hdrvdp_iturgb2native(itu_standard, IMG_E):
    """
    Find a transformation matrix from one of the standard ITU color spaces into the native color space of the display, given its emission spectra.
    The matrix operates on relative values: RGB=[1 1 1] result in Y=1
    """
    lambda_, XYZ_cmf = load_spectral_resp('ciexyz31.csv')
    M_native2XYZ = np.zeros((3, 3))
    for rr in range(3):
        for cc in range(3):
            M_native2XYZ[rr, cc] = np.trapz(XYZ_cmf[:, rr] * IMG_E[:, cc], lambda_) * 683.002

    # The absolute luminance must not change - we make sure that the row responsible for Y sums up to 1
    M_native2XYZ = M_native2XYZ / sum(M_native2XYZ[1, :])

    if itu_standard in ['rgb-bt.709', '709', 'rec709']:
        M_iturgb2xyz = np.array([[0.4124, 0.3576, 0.1805],
                                 [0.2126, 0.7152, 0.0722],
                                 [0.0193, 0.1192, 0.9505]])
    elif itu_standard in ['rgb-bt.2020', '2020', 'rec2020']:
        M_iturgb2xyz = np.array([[0.6370, 0.1446, 0.1689],
                                 [0.2627, 0.6780, 0.0593],
                                 [0.0000, 0.0281, 1.0610]])
    elif itu_standard == 'xyz':
        M_iturgb2xyz = np.eye(3)
    else:
        raise ValueError('Unknown RGB color space')
    M = np.linalg.solve(M_native2XYZ, M_iturgb2xyz)
    return M
