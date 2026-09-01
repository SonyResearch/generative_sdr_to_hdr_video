'''Convolve with a large support kernel in the Fourier domain.

Y = fast_conv_fft( X, fH, pad_value )

X - image to be convolved (in spatial domain)
fH - filter to convolve with in the Fourier domain, idealy 2x size of X
pad_value - value to use for padding when expanding X to the size of fH

(C) Rafal Mantiuk <mantiuk@gmail.com>
This is an experimental code for internal use. Do not redistribute.
'''
import numpy as np


def fast_conv_fft(X, fH, pad_value):
    width = fH.shape[0] - X.shape[0]
    height = fH.shape[1] - X.shape[1]

    fX = np.fft.fft2(np.pad(X, ((0, width), (0, height)), 'constant', constant_values=pad_value))

    Yl = np.real(np.fft.irfft2(fX * fH, [fX.shape[0], fX.shape[1]]))

    Y = Yl[:X.shape[0], :X.shape[1]]

    return Y
