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


from VDP2.create_cycdeg_image import create_cycdeg_image
from VDP2.hdrvdp_mtf import hdrvdp_mtf
from VDP2.load_spectral_resp import load_spectral_resp
from VDP2.fast_conv_fft import fast_conv_fft
from VDP2.clamp import clamp
from VDP2.hdrvdp_joint_rod_cone_sens import hdrvdp_joint_rod_cone_sens
from scipy import integrate
from VDP2.hdrvdp_rod_sens import hdrvdp_rod_sens
from VDP2.hdrvdp_ncsf import hdrvdp_ncsf
from VDP2.PyrTools import *

def build_jndspace_from_S(l, S):
    L = 10 ** l
    dL = np.zeros(L.shape)

    for k in np.arange(0, len(L)):
        thr = L[k] / S[k]

        # Different than in the paper because integration is done in the log
        # domain - requires substitution with a Jacobian determinant
        dL[k] = 1 / thr * L[k] * np.log(10)

    Y = l
    jnd = integrate.cumtrapz(dL, l)
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

    width = img.shape[1]
    height = img.shape[0]
    img_sz = [height, width]  # image size

    img_ch = img.shape[2]

    rho2 = create_cycdeg_image(np.array(img_sz) * 2,
                               metric_par.pix_per_deg)  # spatial frequency for each FFT coefficient, for 2x image size
    if (metric_par.do_mtf):
        mtf_filter = hdrvdp_mtf(rho2, metric_par)  # only the last param is adjusted
    else:
        mtf_filter = np.ones(rho2.shape)

    # Load spectral sensitivity curves
    [lamb, LMSR_S] = load_spectral_resp('log_cone_smith_pokorny_1975.csv')
    LMSR_S[LMSR_S == 0] = np.min(LMSR_S[:])
    LMSR_S = 10 ** LMSR_S

    [_, ROD_S] = load_spectral_resp('cie_scotopic_lum.txt')
    LMSR_S = np.hstack((LMSR_S, ROD_S))

    IMG_E = metric_par.spectral_emission

    # == == == == == == == == == == == == == == == == =
    # Precompute photoreceptor non - linearity
    # == == == == == == == == == == == == == == == == =

    pn = create_pn_jnd((metric_par))

    pn['jnd1'] = pn['jnd1'] * metric_par.sensitivity_correction
    pn['jnd2'] = pn['jnd2'] * metric_par.sensitivity_correction
    pn['jnd1'] = np.insert(pn['jnd1'], 0, 0)
    pn['jnd2'] = np.insert(pn['jnd2'], 0, 0)
    # == == == == == == == == == == == == == == == == =
    # Optical Transfer Function
    # == == == == == == == == == == == == == == == == =

    L_O = np.zeros(img.shape)
    for k in np.arange(0, img_ch):  # for each color channel
        if (metric_par.do_mtf):
            # Use per - channel or one per all channels surround value
            pad_value = metric_par.surround_l[k]
            L_O[:, :, k] = clamp(fast_conv_fft(img[:, :, k].astype(float), mtf_filter, pad_value), 1e-5, 1e10)
        else:
            # NO mtf
            L_O[:, :, k] = img[:, :, k]

    # TODO - MTF reduces luminance values

    # == == == == == == == == == == == == == == == == =
    # Photoreceptor spectral sensitivity
    # == == == == == == == == == == == == == == == == =

    M_img_lmsr = np.zeros([img_ch, 4])  # Color space transformation matrix
    for k in np.arange(0, 4):
        for l in np.arange(0, img_ch):
            M_img_lmsr[l, k] = np.trapz(LMSR_S[:, k] * IMG_E[:, l], lamb)

    # Color space conversion
    # a = np.dot(np.reshape(L_O, (width * height, img_ch)), M_img_lmsr)
    R_LMSR = np.reshape(np.dot(np.reshape(L_O, (width * height, img_ch)), M_img_lmsr), (height, width, 4))

    # surround_LMSR = metric_par.surround_l * M_img_lmsr;

    # == == == == == == == == == == == == == == == == =
    # Adapting luminance
    # == == == == == == == == == == == == == == == == =

    L_adapt = R_LMSR[:, :, 0] + R_LMSR[:, :, 1]

    # == == == == == == == == == == == == == == == == =
    # Photoreceptor non - linearity
    # == == == == == == == == == == == == == == == == =

    # La = mean(L_adapt(:) )

    P_LMR = np.zeros([height, width, 4])
    # surround_P_LMR = zeros(1, 4);
    for k in [0, 1, 3]:  # ignore S - does not influence luminance
        if (k == 3):
            ph_type = 2  # rod
            ii = 2
        else:
            ph_type = 1  # cone
            ii = k
        Ykeys = 'Y' + str(ph_type)
        jndKeys = 'jnd' + str(ph_type)
        P_LMR[:, :, ii] = pointOp(np.log10(clamp(R_LMSR[:, :, k], 10 ** pn[Ykeys][0], 10 ** pn[Ykeys][-1])),
                                  pn[jndKeys], pn[Ykeys][0], pn[Ykeys][1] - pn[Ykeys][0])

    # surround_P_LMR(ii) = interp1(pn_Y {ph_type}, pn_jnd {ph_type}, ...
    # log10(clamp(surround_LMSR(k), 10 ^ pn_Y{ph_type}(1), 10 ^ pn_Y{ph_type}(end)) ) )

    # == == == == == == == == == == == == == == == == =

    # Remove the DC component, from
    # cone and rod pathways separately
    # == == == == == == == == == == == == == == == == =

    # TODO - check if there is a better way to do it
    # cones
    P_C = P_LMR[:, :, 0] + P_LMR[:, :, 1]
    mm = np.mean(P_C)
    P_C = P_C - mm
    # rods
    mm = np.mean(P_LMR[:, :, 2])
    P_R = P_LMR[:, :, 2] - mm

    # == == == == == == == == == == == == == == == == =
    # Achromatic response
    # == == == == == == == == == == == == == == == == =

    P = P_C + P_R
    # == == == == == == == == == == == == == == == == =
    # Multi - channel decomposition
    # == == == == == == == == == == == == == == == == =
    bandskeys = {"pyr", "pind", "sz"}

    bands = dict.fromkeys(bandskeys)

    [bands['pyr'], bands['pind'], _, _] = buildSpyr(P, 'auto', metric_par.steerpyr_filter)

    bands['sz'] = np.ones([int(spyrHt(bands['pind'])) + 2, 1])
    bands['sz'][1:-1] = spyrNumBands(bands['pind'])
    pindList = -np.arange(0, (spyrHt(bands['pind']) + 2))
    band_freq = 2 ** pindList * metric_par.pix_per_deg / 2

    # CSF - filter the base band
    L_mean = np.mean(L_adapt)

    BB = pyrBand(bands['pyr'], bands['pind'], sum(bands['sz']) - 1)
    # I wish to know why sqrt(2) works better, as it should be 2
    rho_bb = create_cycdeg_image(np.array(BB.shape) * 2, band_freq[-1] * 2 * np.sqrt(2))
    csf_bb = hdrvdp_ncsf(rho_bb, L_mean, metric_par)

    # pad value must be the same for test and reference images.
    if (bb_padvalue == -1):
        bb_padvalue = np.mean(BB)
    resizeshape = pyrBandIndices(bands['pind'], sum(bands['sz']) - 1).shape[0]
    bands['pyr'][pyrBandIndices(bands['pind'], sum(bands['sz']) - 1)] = np.reshape(
        fast_conv_fft(BB, csf_bb, bb_padvalue), [resizeshape], order="F")

    return [bands, L_adapt, band_freq, bb_padvalue]

