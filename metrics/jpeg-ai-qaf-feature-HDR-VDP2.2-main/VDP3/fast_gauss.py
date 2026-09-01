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


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def fast_gauss(X, sigma, do_norm=True, pad_value='replicate'):
    """
    Low-pass filter image using the Gaussian filter.
    do_norm - normalize the results or not, 'true' by default. If do_norm==true,
    filtering computes weighted average (defult), if do_norm==false, filtering computes a
    weighted sum.
    pad_value - value passed to padarray (0, 1, 'replicate', etc.)
    """

    if sigma >= 4.3:  # Experimentally found threshold when FFT is faster
        ks = np.array(X.shape[:2]) * 2
        N = ks[0]
        M = ks[1]
        KX0 = (np.mod(1 / 2 + np.arange(M) / M, 1) - 1 / 2)
        KX1 = KX0 * (2 * np.pi)
        KY0 = (np.mod(1 / 2 + np.arange(N) / N, 1) - 1 / 2)
        KY1 = KY0 * (2 * np.pi)

        XX, YY = np.meshgrid(KX1, KY1)

        K = np.exp(-0.5 * (XX ** 2 + YY ** 2) * sigma ** 2)

        if not do_norm:
            K = K / (np.sum(K) / K.size)

        Y = np.zeros(X.shape, X.dtype)
        if len(X.shape) == 2:
            X = X[:, :, np.newaxis]
            Y = Y[:, :, np.newaxis]

        for cc in range(X.shape[2]):
            Y[:, :, cc] = fast_conv_fft(X[:, :, cc], K, pad_value)
        Y = Y[:, :, 0]
    else:
        ksize = round(sigma * 6)
        ksize = ksize + 1 - ksize % 2  # Make sure the kernel size is always an odd number
        h = matlab_style_gauss2D((ksize, ksize), sigma)
        if not do_norm:
            h = h / h[int(h.shape[0] / 2), int(h.shape[1] / 2)]

        Y = ndimage.convolve(X, h, mode='constant', cval=pad_value)

    return Y


def pad_replicate(array, pad_size):
    for i in range(0, pad_size[0]):
        array[pad_size[0] + i, :pad_size[1]] = array[pad_size[0] - 1, :pad_size[1]]
    for j in range(0, pad_size[1]):
        array[:pad_size[0], pad_size[1] + j] = array[:pad_size[0], pad_size[1] - 1]

    array[pad_size[0]:, pad_size[1]:] = array[pad_size[0] - 1, pad_size[1] - 1]
    return array


def fast_conv_fft(X, fH, pad_value):
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
    if pad_value == 'replicate':

        array = np.pad(X, ((0, pad_size[0]), (0, pad_size[1])), 'constant')
        pad_array = pad_replicate(array, pad_size)
        fX = fftpack.fft2(pad_array)
    elif pad_value == 'symmetric':
        fX = np.fft.fft2(np.pad(X, ((0, pad_size[0]), (0, pad_size[1])), mode=pad_value))
    else:
        fX = np.fft.fft2(np.pad(X, ((0, pad_size[0]), (0, pad_size[1])), 'constant', constant_values=pad_value))

    Yl = fftpack.ifft2(fX * fH).real

    Y = Yl[:X.shape[0], :X.shape[1]]

    return Y
