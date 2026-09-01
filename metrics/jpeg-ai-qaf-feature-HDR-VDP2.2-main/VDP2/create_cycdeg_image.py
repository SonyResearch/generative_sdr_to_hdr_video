'''CREATE_CYCDEG_IMAGE (internal) create matrix that contains frequencies,
given in cycles per degree.

D = create_cycdeg_image( im_size, pix_per_deg )
im_size     - [height width] vector with image size
pix_per_deg - pixels per degree for both horizontal and vertical axis
              (assumes square pixels)

Useful for constructing Fourier-domain filters based on OTF or CSF data.

(C) Rafal Mantiuk <mantiuk@gmail.com>
This is an experimental code for internal use. Do not redistribute.
'''

import numpy as np


def create_cycdeg_image(im_size, pix_per_deg):
    nyquist_freq = 0.5 * pix_per_deg
    half_size = np.floor(im_size / 2)
    odd = im_size % 2
    freq_step = nyquist_freq / half_size
    if (odd[1]):
        xx = [np.linspace(0, nyquist_freq, num=int(half_size[1] + 1)),
              np.linspace(-nyquist_freq, -freq_step[1], num=int(half_size[1]))]
    else:
        xx = [np.linspace(0, nyquist_freq - freq_step[1], num=int(half_size[1]))
            , np.linspace(-nyquist_freq, -freq_step[1], num=int(half_size[1]))]

    if (odd[0]):
        yy = [np.linspace(0, nyquist_freq, num=int(half_size[0] + 1)),
              np.linspace(-nyquist_freq, -freq_step[0], num=int(half_size[0]))]
    else:
        yy = [np.linspace(0, nyquist_freq - freq_step[0], num=int(half_size[0])),
              np.linspace(-nyquist_freq, -freq_step[0], num=int(half_size[0]))]

    [XX, YY] = np.meshgrid(xx, yy)

    D = np.sqrt(XX ** 2 + YY ** 2)

    return D
