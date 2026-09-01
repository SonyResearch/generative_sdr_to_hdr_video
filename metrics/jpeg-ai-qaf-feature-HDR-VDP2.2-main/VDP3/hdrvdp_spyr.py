import numpy as np
from VDP3.PyrTools import *


class hdrvdp_spyr():
    def __init__(self, steerpyr_filter):
        self.steerpyr_filter = steerpyr_filter
        self.ppd = None
        self.img_sz = None
        self.band_freqs = None
        self.pyr = None
        self.pind = None
        self.sz = None

    def decompose(self, I, ppd):
        self.ppd = ppd
        self.img_sz = I.shape

        # We want the minimum frequency the band of 2cpd or higher
        # that is base-band can be 1cpd
        _, _, lofilt, _, _, _ = Filters(self.steerpyr_filter)
        pyr_height = min(int(np.ceil(np.log2(ppd))) - 2, maxPyrHt(np.array(self.img_sz), np.array(lofilt.shape[0])))

        self.band_freqs = 1 / 2 ** (np.arange(pyr_height + 2)) * self.ppd / 2


        self.pyr, self.pind, _, _ = buildSpyr(I, pyr_height, self.steerpyr_filter)
        self.sz = np.ones(int(spyrHt(self.pind) + 2))
        self.sz[1:-1] = spyrNumBands(self.pind)

    def reconstruct(self):

        return reconSpyr(self.pyr, self.pind, self.steerpyr_filter)

    def get_band(self, band, o):
        band_norm = 2 ** band  # This accounts for the fact that amplitudes increase in the steerable pyramid
        if self.sz[band] == 1:
            oc = 0
        else:
            oc = min(o, self.sz[band])
        return pyrBand(self.pyr, self.pind, sum(self.sz[:band]) + oc) / band_norm


    def set_band(self, band, o, B):
        band_norm = 2 ** band  # This accounts for the fact that amplitudes increase in the steerable pyramid
        a = sum(self.sz[:band]) + o
        self.pyr[pyrBandIndices(self.pind, sum(self.sz[:band]) + o)] = B.T.flatten() * band_norm


    def band_count(self):
        return len(self.sz)

    def orient_count(self, band):
        return int(self.sz[band])

    def band_size(self, band, o):
        return np.array(self.pind[int(sum(self.sz[:band]) + o)]).astype(int)

    def get_freqs(self):
        return self.band_freqs
