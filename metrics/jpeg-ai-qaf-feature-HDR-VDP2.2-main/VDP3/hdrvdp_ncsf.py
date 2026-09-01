'''  Neural contrast sensitivity function

     S = hdrvdp_csf( rho, lum, metric_par )

     This is a naural contrast sensitivity function, which does not account
     for the optical component, nor luminance-dependent component. To compute
     a complete CSF, use:

     CSF = hdrvdp_csf( rho, lum, metric_par ) * hdrvdp_mtf( rho, metric_par ) *
        hdrvdp_joint_rod_cone_sens( lum, metric_par );

     Note that the peaks of nCSF are not normalized to 1. This is to account
     for the small variations in the c.v.i. (sensitivity due to adapting
     luminance).

     Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>

     Permission to use, copy, modify, and/or distribute this software for any
     purpose with or without fee is hereby granted, provided that the above
     copyright notice and this permission notice appear in all copies.

     THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
     WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
     MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
     ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
     WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
     ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
'''

import numpy as np
from VDP3.clamp import clamp
from scipy import interpolate
from VDP3.hdrvdp_mtf import hdrvdp_mtf


def hdrvdp_aesl(rho, metric_par):
    """
    Age-related empirical sensitivity loss

    S_corr = hdrvdp_aesl(rho, metric_par)

    rho - spatial frequency in cpd

    The function is based on the paper:

    Mantiuk, R. K., & Ramponi, G. (2018).
    Age-dependent predictor of visibility in complex scenes.
    Journal of the Society for Information Display, 1–21.
    https://doi.org/10.1002/jsid.623
    """

    gamma = 10 ** metric_par.aesl_base
    S_corr = 10 ** (-(10 ** metric_par.aesl_slope_freq * np.log2(rho + gamma)) * max(0, metric_par.age - 24))

    return S_corr

def hdrvdp_ncsf(rho, lum, metric_par):
    csf_pars = np.array(metric_par.csf_params)
    lum_lut = np.log10(metric_par.csf_lums)

    log_lum = np.log10(lum)
    par = np.zeros([lum.size, 4])
    log_lum = clamp(log_lum, lum_lut[0], lum_lut[-1])
    for k in np.arange(0, 4):
        par[:, k] = interpolate.interp1d(lum_lut, csf_pars[:, k + 1])(log_lum)

    S = par[:, 3] * 1 / ((1 + (par[:, 0] * rho) ** par[:, 1]) * 1 / (1 - np.exp(-(rho / 7) ** 2)) ** par[:, 2]) ** 0.5
    S = S / hdrvdp_mtf(rho, metric_par)
    #Inf if rho == 0
    S[rho <= 1e-4] = 0

    if (metric_par.do_aesl):
        S = S * hdrvdp_aesl(rho, metric_par)


    return S
