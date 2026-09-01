'''HDRVDP_VISUAL_PATHWAY (internal) Process image along the visual pathway to compute normalized perceptual response

img - image data (can be multi-spectral) name - string with the name of this map (shown in warnings and error messages)
options - cell array with the ‘option’, value pairs bands - CSF normalized freq. bands

Copyright © 2011, Rafal Mantiuk mantiuk@gmail.com

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted,
provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
'''
import copy

import numpy as np

from VDP3.create_cycdeg_image import create_cycdeg_image
from VDP3.hdrvdp_mtf import hdrvdp_mtf
from VDP3.load_spectral_resp import load_spectral_resp
from VDP3.clamp import clamp
from VDP3.hdrvdp_joint_rod_cone_sens import hdrvdp_joint_rod_cone_sens
from scipy import integrate
from VDP3.hdrvdp_rod_sens import hdrvdp_rod_sens
from VDP3.PyrTools import *
from VDP3.hdrvdp_local_adapt import hdrvdp_local_adapt
from VDP3.pupil_d_unified import pupil_d_unified
from VDP3.fast_gauss import *

def build_jndspace_from_S(l, S):
    L = 10 ** l
    dL = np.zeros(L.shape)

    for k in np.arange(0, len(L)):
        thr = L[k] / S[k]

        # Different than in the paper because integration is done in the log
        # domain - requires substitution with a Jacobian determinant
        dL[k] = 1 / thr * L[k] * np.log(10)

    Y = l
    jnd = integrate.cumulative_trapezoid(dL, l)
    return [Y, jnd]


#
#
def create_pn_jnd(metric_par):
    # Create lookup tables for intensity -> JND mapping

    c_l = np.logspace(-5, 5, 2048)

    s_A = hdrvdp_joint_rod_cone_sens(c_l, metric_par)
    s_R = hdrvdp_rod_sens(c_l, metric_par) * 10 ** metric_par.rod_sensitivity

    # s_C = s_L = S_M
    s_C = 0.5 * interpolate.interp1d(c_l, np.max((s_A - s_R, [1e-3] * 2048), axis=0))(
        np.min((c_l * 2, [c_l[-1]] * 2048), axis=0))

    keys = {"Y1", "Y2", "jnd1", "jnd2"}
    pn = dict.fromkeys(keys)

    [pn["Y1"], pn["jnd1"]] = build_jndspace_from_S(np.log10(c_l), s_C)
    [pn["Y2"], pn["jnd2"]] = build_jndspace_from_S(np.log10(c_l), s_R)
    return pn


def hdrvdp_visual_pathway(img, name, metric_par, bb_padvalue):
    global hdrvdp_cache
    if np.isnan(img).any():
        warnings.warn('hdrvdp:BadImageData', '%s image contains NaN pixel values', name)
        img[np.isnan(img)] = 1e-5

    # == == == == == == == == == == == == == == == == =
    # Precompute
    # precompute common variables and structures or take them from cache
    # == == == == == == == == == == == == == == == == =
    a = img[:, :, 0]
    width = img.shape[1]
    height = img.shape[0]
    img_sz = [height, width]  # image size

    img_ch = img.shape[2]

    rho2 = create_cycdeg_image(np.array(img_sz) * 2,
                               metric_par.pix_per_deg)  # spatial frequency for each FFT coefficient, for 2x image size
    # if (metric_par.do_mtf):
    #     mtf_filter = hdrvdp_mtf(rho2, metric_par)  # only the last param is adjusted
    # else:
    #     mtf_filter = np.ones(rho2.shape)

    # Load spectral sensitivity curves
    [lamb, LMSR_S] = load_spectral_resp('log_cone_smith_pokorny_1975.csv')
    LMSR_S[:20, :] = 0
    LMSR_S[LMSR_S == 0] = np.min(LMSR_S[:])
    LMSR_S = 10 ** LMSR_S

    [_, ROD_S] = load_spectral_resp('cie_scotopic_lum.txt')
    ROD_S[:20, :] = 0
    LMSR_S = np.hstack((LMSR_S, ROD_S))

    IMG_E = copy.deepcopy(metric_par.spectral_emission)

    # == == == == == == == == == == == == == == == == =
    # Precompute photoreceptor non - linearity
    # == == == == == == == == == == == == == == == == =

    pn = create_pn_jnd((metric_par))

    pn['jnd1'] = pn['jnd1'] * 10 ** metric_par.base_sensitivity_correction
    pn['jnd2'] = pn['jnd2'] * 10 ** metric_par.base_sensitivity_correction
    pn['jnd1'] = np.insert(pn['jnd1'], 0, 0)
    pn['jnd2'] = np.insert(pn['jnd2'], 0, 0)
    # == == == == == == == == == == == == == == == == =
    # Optical Transfer Function
    # == == == == == == == == == == == == == == == == =

    L_O = np.zeros(img.shape)
    for k in np.arange(0, img_ch):  # for each color channel
        if metric_par.mtf is not None:
            # Use per - channel or one per all channels surround value
            pad_value = metric_par.pad_value
            mtf_filter = hdrvdp_mtf(rho2, metric_par)
            L_O[:, :, k] = clamp(fast_conv_fft(img[:, :, k].astype(float), mtf_filter, pad_value), 1e-5, 1e10)
        else:
            # NO mtf
            L_O[:, :, k] = img[:, :, k]
    if not metric_par.debug:
        del mtf_filter, rho2

    if metric_par.do_aod:
        lam = np.array(
            [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
             600, 610, 620, 630, 640, 650])
        TL1 = np.array([0.600, 0.510, 0.433, 0.377, 0.327, 0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 0.147,
                        0.133, 0.120, 0.107, 0.093, 0.080, 0.067, 0.053, 0.040,
                        0.033, 0.027, 0.020, 0.013,
                        0.007,
                        0.000])
        TL2 = np.array([1.000,
                        .583,
                        .300,
                        .116,
                        .033,
                        .005] + [0] * 20)

        OD_y = TL1 + TL2

        if metric_par.age <= 60:
            OD = TL1 * (1 + .02 * (metric_par.age - 32)) + TL2
        else:
            OD = TL1 * (1.56 + .0667 * (metric_par.age - 60)) + TL2

        trans_filter = np.power(10, -interpolate.PchipInterpolator(lam,
                                                                   OD - OD_y,
                                                                   )(np.clip(lamb, lam[0], lam[-1])))

        IMG_E *= np.tile(trans_filter[np.newaxis].T, (1, np.shape(IMG_E)[1]))

    # TODO - MTF reduces luminance values

    # == == == == == == == == == == == == == == == == =
    # Photoreceptor spectral sensitivity
    # == == == == == == == == == == == == == == == == =

    M_img_lmsr = np.zeros([img_ch, 4])  # Color space transformation matrix
    for k in np.arange(0, 4):
        for l in np.arange(0, img_ch):
            M_img_lmsr[l, k] = np.trapz(LMSR_S[:, k] * IMG_E[:, l] * 683.002, lamb)

    # Normalization step We want RGB = [1 1 1] to result in L + M = 1(L + M is equivalent to luminance)
    M_img_lmsr = M_img_lmsr / np.sum(M_img_lmsr[:, 0: 2])

    # Color space conversion
    # a = np.dot(np.reshape(L_O, (width * height, img_ch)), M_img_lmsr)
    R_LMSR = np.reshape(np.dot(np.reshape(L_O, (width * height, img_ch)), M_img_lmsr), (height, width, 4))
    R_LMSR = clamp(R_LMSR, 1e-8, 1e10)
    # surround_LMSR = metric_par.surround_l * M_img_lmsr;
    if not metric_par.debug:
        del L_O
    # == == == == == == == == == == == == == == == == =
    # Adapting luminance
    # == == == == == == == == == == == == == == == == =

    # L_adapt = R_LMSR[:, :, 0] + R_LMSR[:, :, 1]
    L_adapt = hdrvdp_local_adapt(R_LMSR[:, :, 0] + R_LMSR[:, :, 1], metric_par.pix_per_deg)

    if metric_par.do_slum:
        # light reduction due to senile miosis (Unified Formula based on
        # the Stanley and Davies function [Watson & Yellott 2012]

        L_a = np.exp(np.mean(np.log(L_adapt)))
        area = img.shape[0] * img.shape[1] / metric_par.pix_per_deg ** 2

        d_ref = pupil_d_unified(L_a, area, 28)
        d_age = pupil_d_unified(L_a, area, metric_par.age)

        lum_reduction = d_age ** 2 / d_ref ** 2

        R_LMSR *= lum_reduction

    # == == == == == == == == == == == == == == == == =
    # Photoreceptor non - linearity
    # == == == == == == == == == == == == == == == == =

    # La = mean(L_adapt(:) )

    P_LMR = np.zeros_like(R_LMSR)
    for k in [0, 1, 3]:  # ignore S - does not influence luminance
        if (k == 3):
            ph_type = 2  # rod
            ii = 2
        else:
            ph_type = 1  # cone
            ii = k
        Ykeys = 'Y' + str(ph_type)
        jndKeys = 'jnd' + str(ph_type)
        P_LMR[:, :, ii] = interpolate.interp1d(pn[Ykeys], pn[jndKeys])(
            np.log10(clamp(R_LMSR[:, :, k], 10 ** pn[Ykeys][0], 10 ** pn[Ykeys][-1])))

        # == == == == == == == == == == == == == == == == =

        # Remove the DC component, from
        # cone and rod pathways separately
        # == == == == == == == == == == == == == == == == =

    # TODO - check if there is a better way to do it
    # cones
    P_C = P_LMR[:, :, 0] + P_LMR[:, :, 1]
    # mm = np.mean(P_C)
    # P_C = P_C - mm
    # rods
    # mm = np.mean(P_LMR[:, :, 2])
    P_R = P_LMR[:, :, 2]

    # == == == == == == == == == == == == == == == == =
    # Achromatic response
    # == == == == == == == == == == == == == == == == =

    P = P_C + P_R

    if (not metric_par.debug):
        # memory savingsfor huge images
        del P_LMR, P_C, P_R, R_LMSR

    # == == == == == == == == == == == == == == == == =
    # Multi - channel decomposition
    # == == == == == == == == == == == == == == == == =

    # TODO: Decomposition should also run on the GPU
    bands = metric_par.mult_scale
    bands.decompose(P, metric_par.pix_per_deg)
    # Remove DC from the base-band - this is important for contrast masking
    BB = bands.get_band(int(bands.band_count() - 1), 1)
    bands.set_band(int(bands.band_count() - 1), 0, BB - np.mean(BB))
    return [copy.deepcopy(bands), L_adapt, bb_padvalue, P]
