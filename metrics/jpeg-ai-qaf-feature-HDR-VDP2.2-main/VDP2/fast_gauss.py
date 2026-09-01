import numpy as np
from scipy import fftpack
from scipy import ndimage
''' Convolve with a large support kernel in the Fourier domain.

    Y = fast_conv_fft( X, fH, pad_value )

    X - image to be convolved (in spatial domain)
    fH - filter to convolve with in the Fourier domain, idealy 2x size of X
    pad_value - value to use for padding when expanding X to the size of fH

    (C) Rafal Mantiuk <mantiuk@gmail.com>
    This is an experimental code for internal use. Do not redistribute.
    '''

def fast_gauss(X, sigma, do_norm=True):
    """
    Low-pass filter image using the Gaussian filter

    Parameters:
    X : numpy array
        Image to be filtered
    sigma : float
        Standard deviation for Gaussian kernel
    do_norm : bool, optional
        Normalize the results or not, 'true' by default

    Returns:
    Y : numpy array
        Filtered image
    """

    if do_norm:
        norm_f = 1
    else:
        norm_f = sigma * np.sqrt(2 * np.pi)

    if sigma >= 4.3:
        ks = np.array(X.shape[:2]) * 2

        NF = np.pi
        xx = np.concatenate((np.linspace(0, NF, ks[1] // 2), np.linspace(-NF, -NF / (ks[1] // 2), ks[1] // 2)))
        yy = np.concatenate((np.linspace(0, NF, ks[0] // 2), np.linspace(-NF, -NF / (ks[0] // 2), ks[0] // 2)))

        XX, YY = np.meshgrid(xx, yy)

        K = np.exp(-0.5 * (XX ** 2 + YY ** 2) * sigma ** 2) * norm_f

        Y = np.zeros(X.shape)
        for cc in range(X.shape[2]):
            Y[:, :, cc] = fast_conv_fft(X[:, :, cc], K)

    else:
        ksize = int(round(sigma * 5))
        h = ndimage.gaussian_filter(ksize, sigma) * norm_f
        Y = ndimage.convolve(X, h, mode='nearest')

    return Y


def fast_conv_fft(X, fH):
    """
    Convolve with a large support kernel in the Fourier domain.

    Parameters:
    X : numpy array
        Image to be convolved (in spatial domain)
    fH : numpy array
        Filter to convolve with in the Fourier domain

    Returns:
    Y : numpy array
        Convolved image
    """

    pad_size = np.array(fH.shape) - np.array(X.shape)

    fX = fftpack.fft2(np.pad(X, ((0, pad_size[0]), (0, pad_size[1])), 'constant'))

    Yl = fftpack.ifft2(fX * fH).real

    Y = Yl[:X.shape[0], :X.shape[1]]

    return Y