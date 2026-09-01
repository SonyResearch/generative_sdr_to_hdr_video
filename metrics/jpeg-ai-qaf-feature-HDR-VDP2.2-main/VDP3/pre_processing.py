import math
import numpy as np

def hdrvdp_pix_per_deg(display_diagonal_in, resolution, viewing_distance):
    """
    HDRVDP_PIX_PER_DEG - computer pixels per degree given display parameters and viewing distance

    ppd = hdrvdp_pix_per_deg(display_diagonal_in, resolution, viewing_distance)

    This is a convenience function that can be used to provide angular resolution of input images for the HDR-VDP-2.

    display_diagonal_in - diagonal display size in inches, e.g. 19, 14
    resolution - display resolution in pixels as a vector, e.g. [1024 768]
    viewing_distance - viewing distance in meters, e.g. 0.5

    Note that the function assumes 'square' pixels, so that the aspect ratio is resolution[0]:resolution[1].

    EXAMPLE:
    ppd = hdrvdp_pix_per_deg(24, [1920, 1200], 0.5)
    """

    ar = resolution[0] / resolution[1]  # width/height
    height_mm = math.sqrt((display_diagonal_in * 25.4) ** 2 / (1 + ar ** 2))
    height_deg = 2 * math.atan(0.5 * height_mm / (viewing_distance * 1000))/math.pi*180
    ppd = resolution[1] / height_deg
    return ppd


def pq2lin(im_pq):
    """
    Transforms a PQ-encoded image into an image with absolute linear colour values (according to Rec. 2100).
    im_pq - a PQ-encoded image (HxWx3), the values are in the range 0-1
    im_lin - an image with linear colour values in the range 0.005 to 10000
    """
    # The maximum allowed value for PQ is 10000 for HDR frames
    Lmax = 10000

    n = 0.15930175781250000
    m = 78.843750000000000
    c1 = 0.83593750000000000
    c2 = 18.851562500000000
    c3 = 18.687500000000000

    im_t = np.maximum(im_pq, 0) ** (1 / m)

    im_lin = Lmax * (np.maximum(im_t - c1, 0) / (c2 - c3 * im_t)) ** (1 / n)

    return im_lin

class TransferFunction:
    """
    Create a transfer function coding object. The input (camera RAW) signal will be between 0 and 2**bits-1.
    The encoded values will be between 0 and 1.
    """

    def __init__(self, bits):
        self.bits = bits

    def decode(self, V):
        L_float = self.decode_float(V)
        return (L_float + 0.5).astype(int)  # add 0.5 for rounding

    def decode_float(self, V):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

class TF_HLG(TransferFunction):
    '''
    Hybrid Log-Gamma
    https://en.wikipedia.org/wiki/Hybrid_Log-Gamma
    '''
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    cut = 1 / 12

    def encode(self, L):
        V = L.astype(np.float32) / (2 ** self.bits - 1)
        acut = (V > self.cut)
        bcut = (V <= self.cut)
        V[bcut] = np.sqrt(3) * np.sqrt(V[bcut].astype(np.float32))
        V[acut] = self.a * np.log(12 * V[acut].astype(np.float32) - self.b) + self.c

        return V

    def decode_float(self, V):
        enc_cut = self.encode(np.array([self.cut * (2 ** self.bits - 1)]))[0]
        acut = (V > enc_cut)
        bcut = (V <= enc_cut)
        L = np.zeros(V.shape)
        L[bcut] = (V[bcut] / np.sqrt(3)) ** 2
        L[acut] = (np.e ** ((V[acut] - self.c) / self.a) + self.b) / 12

        return np.clip(L, 0, 1) * (2 ** self.bits - 1)