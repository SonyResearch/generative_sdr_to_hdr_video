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


def create_cycdeg_image(im_size, pix_per_deg, use_gpu=False):

    nyquist_freq = 0.5 * pix_per_deg

    KX0 = (np.mod(1 / 2 + np.arange(im_size[1]) / im_size[1], 1) - 1 / 2)
    KX1 = KX0 * nyquist_freq * 2
    KY0 = (np.mod(1 / 2 + np.arange(im_size[0]) / im_size[0], 1) - 1 / 2)
    KY1 = KY0 * nyquist_freq * 2

    XX, YY = np.meshgrid(KX1, KY1)
    D = np.sqrt(XX ** 2 + YY ** 2)
    return D

