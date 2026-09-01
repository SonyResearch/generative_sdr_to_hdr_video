import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import pchip_interpolate
from VDP3.fast_gauss import fast_gauss

def hdrvdp_local_adapt(L_otf, ppd):
    """
    Computes the luminance the VS is adapted to at a particular pixel location.
    L_otf must be retinal luminance which accounts for glare.
    ppd is pixels per visual degree.
    """
    if 0:
        # Best model parameters - This model was the best fit, but it cannot handle
        # luminance levels <1 cd/m^2. It is also quite expensive to compute.

        p_nl1 = [0.030594263300886, 1.798038555035897, 0.072142302635550, 0.010000000000000]  # non-linearity 1
        sigma_nr_1 = 0.428078089769146  # Gaussian 1
        p_nl2 = [0.030205776178761, 2.934114110750746, 0.210878838078038, 0.084607721580785]  # non-linearity 2
        sigma_nr_2 = 0.082417606884968  # Gaussian 2
        alpha = 0.653547088004559  # summation weight

        L_la = alpha * inv_nonlin(gaussian_filter(fw_nonlin(L_otf, p_nl1), sigma_nr_1 * ppd), p_nl1) + (
                    1 - alpha) * inv_nonlin(gaussian_filter(fw_nonlin(L_otf, p_nl2), sigma_nr_2 * ppd), p_nl2)

    else:
        # This is Model #7 from the paper exp( log( I ) * gauss )
        # this is not the best model but it has only one degree of freedom, is
        # fast to computed and hopefully it better generalizes to low luminance

        sigma = 10 ** -0.781367 * ppd
        L_la = np.exp(fast_gauss(np.log(np.maximum(L_otf, 1e-6)), sigma))

    return L_la


def fw_nonlin(L_in, p):
    nodes = 4
    l_lut = np.linspace(-1, np.log10(5000), nodes + 1)
    delta = l_lut[1] - l_lut[0]
    v_lut = np.cumsum([0] + list(np.array(p) * delta))
    l = np.minimum(np.maximum(np.log10(L_in), l_lut[0]), l_lut[-1])
    L_out = pchip_interpolate(l_lut, v_lut, l)
    return L_out


def inv_nonlin(L_in, p):
    nodes = 4
    l_lut = np.linspace(-1, np.log10(5000), nodes + 1)
    delta = l_lut[1] - l_lut[0]
    v_lut = np.cumsum([0] + list(np.array(p) * delta))
    l_lut_dense = np.linspace(l_lut[0], l_lut[-1])
    v_lut_dense = pchip_interpolate(l_lut, v_lut, l_lut_dense)
    L_out = 10 ** np.interp(np.minimum(np.maximum(L_in, v_lut_dense[0]), v_lut_dense[-1]), v_lut_dense, l_lut_dense)
    return L_out
