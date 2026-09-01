'''HDR-VDP-2 compute visually significant differences between an image pair.

diff = HDRVDP( test, reference, color_encoding, pixels_per_degree, options )

Parameters:
test - image to be tested (e.g. with distortions)

reference - reference image (e.g. without distortions)

color_encoding - color representation for both input images. See below.

pixels_per_degree - visual resolution of the image. See below.

options - cell array with { ‘option’, value } pairs. See the list of options below. Note if unknown
or misspelled option is passed, no warning message is issued.

The function returns a structure with the following fields:
 Q - Quality correlate (higher value, lower quality).
 Q_MOS -[Depreciated, do not use] Mean-opinion-score prediction (from 0 to 100, where 100 is the highest quality)

Test and references images are matrices of size (height, width, channel_count). If there is only one channel, it is
assumed to be achromatic (luminance) with D65 spectra. The values must be given in absolute luminance units (cd/m^2).
If there are three color channels, they are assumed to be RGB of an LCD display with red, green and blue LED backlight.
If different number of color channels is passed, their spectral emission curve should be stored in a comma separated
text file and its name should be passed as ‘spectral_emission’ option.

Note that the current version of HDR-VDP does NOT take color differences into account. Spectral channels are used to
properly compute luminance sensed by rods and cones.

COLOR ENCODING:

HDR-VDP-2 requires that the color encoding is specified explicitly to avoid mistakes when passing images. HDR-VDP
operates on absolue physical units, not pixel values that can be found in images. Therefore, it is necessary to specify
how the metric should interpret input images. The available options are:

‘luminance’ - images contain absolute luminance values provided in photopic cd/m^2. The images must contain exactly one
color channel.

‘luma-display’ - images contain grayscale pixel values, sometimes known as gamma-corrected luminance. The images must
contain exactly one color channel and the maximum pixel value must be 1 (not 256). It corresponds to a gray-scale
channel in YCrCb color spaces used for video encoding. Because ‘luma’ alone does not specify luminance, HDR-VDP-2
assumes the following display model:

L = 99 * V^2.2 + 1,

where L is luminance and V is luma. This corresponds to a display with 2.2 gamma, 100 cd/m^2 maximum luminance and
1 cd/m^2 black level.

‘sRGB-display’ - use this color encoding for standard (LDR) color images. This encoding assumes an sRGB display, with
100 cd/m^2 peak luminance and 1 cd/m^2 black level. Note that this is different from sRGB->XYZ transform, which assumes
the peak luminance of 80 cd/m^2 and 1 cd/m^2 black level. The maximum pixel value must be 1 (not 256).

‘rgb-bt.709’ - use this color space for HDR images AFTER they are at least roughly calibrated in absolute photometric
units. The encoding assumes the ITU-R BT.709 RGB color primaries (the same as for sRGB), but also non-gamma corrected
RGB color space.

‘XYZ’ - input image is provided as ABSOLUTE trichromatic color values in CIE XYZ (1931) color space. The Y channel must
be equal luminance in cd/m^2.

PIXELS_PER_DEGREE:
This argument specifies the angular resolution of the image in terms of the number of pixels per one visual degree.
It will change with a viewing distance and display resolution. A typical value for a stanard resolution computer display
 is around 30. A convenience function hdrvdp_pix_per_deg can be used to compute pixels per degree parameter.

OPTIONS: The options must be given as name-value pairs in a cell array. Default values are given in square brackets.

‘surround_l’ - [mean value of each color channel] luminance/intensity of the surround, which is assumed to be uniform
outside the input images. The default value is 1e-5 (10^-5), which corresponds to almost absolute darkness. Note that
surround_l=0 should be avoided. It is unrealistic to expect absolutely no light in physically plausible environment.
The special value -1 makes the surround surround_l equal to the geometric mean of the image.

‘spectral_emission’ -name of the comma separated file that contains spectral emission curves of color channels of
reference and test images.

‘rgb_display’ - [ccfl-lcd] use this option to specify one of the predefined emission spectra for typical displays.
Availaeble options are: crt - a typical CRT display ccfl-lcd - an LCD display with CCFL backlight led-lcd - an
LCD display with LED backlight

The following are the most important options used to fine-tune and calibrate HDR-VDP:
‘peak_sensitivity’ - absolute sensitivity of the HDR-VDP
‘mask_p’ - excitation of the visual contrast masking model
‘mask_q’ - inhibition of the visual contrast masking model

EXAMPLE: The following example creates a luminance ramp (gradient), distorts it with a random noise and computes
detection probabilities using HDR-VDP.

reference = logspace( log10(0.1), log10(1000), 512 )’ * ones( 1, 512 );
test = reference .* (1 + (rand( 512, 512 )-0.5)*0.1);
res = hdrvdp( test, reference, ‘luminance’, 30, { ‘surround_l’, 13 } );
clf; imshow( hdrvdp_visualize( res.P_map, test ) );

BUGS and LIMITATIONS:

If you suspect the predictions are wrong, check first the Frequently Asked Question section on the HDR-VDP-2 web-site
(http://hdrvdp.sourceforge.net). If it does not help, post your problem to the
HDR-VDP discussion group: http://groups.google.com/group/hdrvdp (preffered) or send an e-mail directly to the author.

REFERENCES

The metric is described in the paper: Mantiuk, Rafat, Kil Joong Kim, Allan G. Rempel, and Wolfgang Heidrich. ?
HDR-VDP-2: A Calibrated Visual Metric for Visibility and Quality Predictions in All Luminance Conditions.?
ACM Trans. Graph (Proc. SIGGRAPH) 30, no. 4 (July 01, 2011): 1. doi:10.1145/2010324.1964935.

When refering to the metric, please cite the paper above and include the version number, for example
“HDR-VDP 2.2.0”. To check the version number, see the ChangeLog.txt. To check the version in the code, call
hdrvdp_version.txt.

Copyright © 2011, Rafal Mantiuk mantiuk@gmail.com

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted,
provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
OF THIS SOFTWARE.

Zhendong Li 06.23 deliver into python code'''

import copy

import numpy as np

from VDP3.hdrvdp_parse_options import Metric_par
from VDP3.load_spectral_resp import load_spectral_resp
from VDP3.hdrvdp_visual_pathway import hdrvdp_visual_pathway
from VDP3.clamp import clamp
from VDP3.hdrvdp_ncsf import hdrvdp_ncsf
from VDP3.PyrTools import *
from scipy import signal
from scipy import interpolate
from VDP3.fast_gauss import fast_gauss
from VDP3.imresize import *
from VDP3.hdrvdp_iturgb2native import hdrvdp_iturgb2native
from VDP3.hdrvdp_colorspace_transform import hdrvdp_colorspace_transform
from VDP3.hdrvdp_spyr import hdrvdp_spyr
from VDP3.create_cycdeg_image import create_cycdeg_image
from VDP3.hdrvdp_determine_padding import hdrvdp_determine_padding


def mutual_masking(b, o, B_R, B_T, metric_par):
    test_band = B_T.get_band(b, o)
    reference_band = B_R.get_band(b, o)

    m = np.minimum(np.abs(test_band), np.abs(reference_band))

    if metric_par.do_si_gauss:
        m = fast_gauss(m, 10 ** metric_par.si_size, False, 0)
    else:
        F = np.ones((metric_par.masking_pool_size, metric_par.masking_pool_size))
        masking_norm = 10 ** metric_par.masking_norm
        m = signal.convolve2d(m, F / F.size, 'same') ** (1 / masking_norm)

    return m


def sign_pow(x, e):
    y = np.sign(x) * abs(x) ** e
    return y


def get_band(bands, b, o):
    if bands['sz'][b] == 1:
        oc = 0
    else:
        oc = min(o, bands['sz'][b])
    B = pyrBand(bands['pyr'], bands['pind'], sum(bands['sz'][0:b]) + oc)

    return B


def get_band_size(bands, b, o):
    sz = bands['pind'][int(sum(bands['sz'][0:b]) + o), :]
    return np.array(sz).astype(int)


def set_band(bands, b, o, B):
    bands['pyr'][pyrBandIndices(bands['pind'], np.sum(bands['sz'][0:b]) + o)] = B.T.flatten()
    return bands


def msre(X):
    return np.sqrt(np.sum(X ** 2)) / X.size


def display_model(V, gamma, peak, black_level):
    return peak * V ** gamma + black_level


def display_model_srgb(sRGB):
    a = 0.055
    thr = 0.04045

    RGB = np.zeros_like(sRGB)
    RGB[sRGB <= thr] = sRGB[sRGB <= thr] / 12.92
    RGB[sRGB > thr] = ((sRGB[sRGB > thr] + a) / (1 + a)) ** 2.4

    return 99 * RGB + 1


def xyz2rgb(XYZ):
    M = np.array([[3.240708, -1.537259, -0.498570],
                  [-0.969257, 1.875995, 0.041555],
                  [0.055636, -0.203996, 1.057069]])

    return np.reshape(np.dot(M, np.reshape(XYZ, (XYZ.shape[0] * XYZ.shape[1], 3)).T).T,
                      (XYZ.shape[0], XYZ.shape[1], 3))


def fix_out_of_gamut(RGB_in, metric_par):
    """
    Report if any color are out of native color gamut and clamp them
    """
    min_v = 1e-6
    clamp_pix = np.count_nonzero(np.any(RGB_in < min_v, axis=2))
    total_pix = RGB_in.shape[0] * RGB_in.shape[1]
    if clamp_pix > 0:
        if not metric_par.disable_lowvals_warning and (clamp_pix / total_pix >= 0.01) and not metric_par.quiet:
            print(
                  f"Warning: Image contains values that are out of the native colour gamut of the display. {clamp_pix / total_pix * 100:.2f} percent of pixels will be clamped. Consider using the correct EOTF. To disable this warning message, add option {{'disable_lowvals_warning', 'true'}}.")
        RGB_out = np.maximum(RGB_in, min_v)
    else:
        RGB_out = RGB_in
    return RGB_out


def minkowski_sum(X, p):
    d = np.sum(np.abs(X) ** p / np.size(X)) ** (1 / p)
    return d


def hdrvdp3(task, test, reference, color_encoding, pixels_per_degree, options={}):
    if test.shape != reference.shape:
        warnings.warn('reference and test images must be the same size')
    # if 'options' not in dir():
    #     options = {}

    metric_par = Metric_par(task, options)

    # The parameters overwrite the options
    if 'pixels_per_degree' in dir():
        metric_par.pix_per_deg = pixels_per_degree
    if 'color_encoding' in dir():
        metric_par.color_encoding = color_encoding

    # Load spectral emission curves
    if len(test.shape) == 2:
        test = test[:, :, np.newaxis]
        reference = reference[:, :, np.newaxis]

    img_channels = test.shape[2]

    if metric_par.color_encoding == 'luminance':
        if (img_channels != 1):
            warnings.warn('Only one channel must be provided for "luminance" color encoding')
        check_if_values_plausible(test, metric_par)
    elif metric_par.color_encoding == 'luma-display':
        if (img_channels != 1):
            warnings.warn('Only one channel must be provided for "luma-display" color encoding')
        test = display_model(test, 2.2, 99, 1)
        reference = display_model(reference, 2.2, 99, 1)
    elif metric_par.color_encoding == 'srgb-display':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "sRGB-display" color encoding')

        if not hasattr(metric_par, 'rgb_display'):
            metric_par.rgb_display = 'led-lcd-srgb'

        test = display_model_srgb(test)
        reference = display_model_srgb(reference)

    elif metric_par.color_encoding == 'rgb-bt.709':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "rgb-bt.709" color encoding')

        if not hasattr(metric_par, 'rgb_display'):
            metric_par.rgb_display = 'led-lcd-srgb'
        check_if_values_plausible(test[:, :, 1], metric_par)

    elif metric_par.color_encoding == 'rgb-bt.2020':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "rgb-bt.2020" color encoding')

        if not hasattr(metric_par, 'rgb_display'):
            metric_par.rgb_display = 'oled'
        check_if_values_plausible(test[:, :, 1], metric_par)

    elif metric_par.color_encoding == 'rgb-native':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "rgb_display" color encoding')

        if not hasattr(metric_par, 'rgb_display'):
            metric_par.rgb_display = 'led-lcd-srgb'
        check_if_values_plausible(test[:, :, 1], metric_par)


    elif metric_par.color_encoding == 'xyz':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "XYZ" color encoding')
        if not hasattr(metric_par, 'rgb_display'):
            metric_par.rgb_display = 'oled'
        check_if_values_plausible(test[:, :, 1], metric_par)

    elif metric_par.color_encoding == 'generic':
        if metric_par.spectral_emission not in dir():
            warnings.warn('"spectral_emission" option must be specified when using the "generic" color encoding')

    metric_par.pad_value = hdrvdp_determine_padding(reference, metric_par)

    if img_channels == 3 and not hasattr(metric_par, 'rgb_display'):
        metric_par.rgb_display = 'ccfl-lcd'

    if (metric_par.spectral_emission != []):
        [_, IMG_E] = load_spectral_resp(metric_par.spectral_emission)
    elif (hasattr(metric_par, 'rgb_display')):
        [_, IMG_E] = load_spectral_resp(f'emission_spectra_{metric_par.rgb_display}.csv')
    elif (img_channels == 1):
        [_, IMG_E] = load_spectral_resp('d65.csv')
    else:
        warnings.warn('spectral_emissiom option needs to be specified')

    if (img_channels == 1 and IMG_E.shape[1] > 1):
        # sum - up spectral responses of all channels for luminance - only data
        IMG_E = np.sum(IMG_E, axis=1)

    if (img_channels != IMG_E.shape[1]):
        warnings.warn(
            'Spectral response data is either missing or is specified for different number of color channels than the input image')
    metric_par.spectral_emission = IMG_E

    ##Tranform from a standard to native (RGB) colour space
    M_itu2native = None
    if metric_par.color_encoding.lower() == 'rgb-bt.2020':
        M_itu2native = hdrvdp_iturgb2native('rgb-bt.2020', IMG_E)
    elif metric_par.color_encoding.lower() in ['rgb-bt.709', 'srgb-display']:
        M_itu2native = hdrvdp_iturgb2native('rgb-bt.709', IMG_E)
    elif metric_par.color_encoding.lower() == 'xyz':
        M_itu2native = hdrvdp_iturgb2native('xyz', IMG_E)

    if M_itu2native is not None:
        test = fix_out_of_gamut(hdrvdp_colorspace_transform(test, M_itu2native), metric_par)

        reference = fix_out_of_gamut(hdrvdp_colorspace_transform(reference, M_itu2native), metric_par)
    metric_par.mult_scale = hdrvdp_spyr(metric_par.steerpyr_filter)
    # Compute spatially - and orientation - selective bands
    # Process reference first to reuse bb_padvalue
    [B_R, L_adapt_reference, bb_padvalue, _] = hdrvdp_visual_pathway(reference, 'reference', metric_par, -1)
    [B_T, L_adapt_test, _, _] = hdrvdp_visual_pathway(test, 'test', metric_par, bb_padvalue)
    band_freq = B_T.get_freqs()
    # precompute CSF
    csf_la = np.logspace(-5, 5, 256)
    csf_log_la = np.log10(csf_la)
    b_count = B_T.band_count()

    CSF = np.zeros([len(csf_la), b_count])
    for b in np.arange(0, b_count):
        CSF[:, b] = hdrvdp_ncsf(band_freq[b], csf_la.T, metric_par)
    L_mean_adapt = (L_adapt_test + L_adapt_reference) / 2
    log_La = np.log10(clamp(L_mean_adapt, csf_la[0], csf_la[-1]))

    D_bands = copy.deepcopy(B_T)

    # Pixels that are actually different
    diff_mask_Three = np.zeros_like(reference)
    diff_mask_Three[(abs(test - reference) / reference) > 0.001] = 1
    # diff_mask = diff_mask_Three[:, :, 0] + diff_mask_Three[:, :, 1] + diff_mask_Three[:, :, 2]
    diff_mask = np.sum(diff_mask_Three, axis=2)
    diff_mask[diff_mask >= 1.0] = 1.0
    Q_err = 0  # quality correlate

    # For each band
    for b in np.arange(0, b_count):
        # maskingparams
        p = 10 ** metric_par.mask_p
        q = 10 ** metric_par.mask_q
        pf = (10 ** metric_par.psych_func_slope) / p

        # accumulate masking activity across orientations(cross - orientation  masking)
        do_mask_xo = (B_T.orient_count(b) > 1)
        if do_mask_xo:
            mask_xo = np.zeros(B_T.band_size(b, 1))
            for i in np.arange(0, B_T.orient_count(b)):
                mask_xo = mask_xo + mutual_masking(b, i, B_R, B_T, metric_par)  # min(abs(B_T{b, o}), abs(B_R{b, o}) )

        log_La_rs = np.clip(imresize(log_La, output_shape=B_T.band_size(b, 0)),
                            csf_log_la[0], csf_log_la[-1])
        # per - pixel contrast sensitivity
        CSF_b = interpolate.interp1d(csf_log_la, CSF[:, b])(log_La_rs)

        # REMOVED: Transform CSF linear sensitivity to the non - linear space of the
        # photoreceptor response
        # CSF_b = CSF_b. * 1. / hdrvdp_joint_rod_cone_sens_diff(10. ^ log_La_rs, metric_par);

        for o in range(0, B_T.orient_count(b)):

            band_diff = B_T.get_band(b, o) - B_R.get_band(b, o)
            # band_diff = B_T.get_band_hdr(b, o) - B_R.get_band_hdr(b, o)

            if b == b_count - 1:
                # base band
                band_freq = B_T.get_freqs()

                # Find the dominant frequency and use the sensitivity for
                # that frequency (more robust than fft filtering used
                # before)
                rho_bb = create_cycdeg_image(np.array(band_diff.shape), band_freq[-1] * 2 * 2)
                band_diff_f = np.fft.fft2(band_diff)
                band_diff_f[0, 0] = 0  # Ignore DC
                ind = np.argmax(np.abs(band_diff_f))
                bb_freq = rho_bb.flatten()[ind]

                L_mean = 10 ** np.mean(log_La_rs)

                N_nCSF = 1 / hdrvdp_ncsf(bb_freq, L_mean, metric_par)
            else:
                N_nCSF = 1 / CSF_b

            # excitation difference
            ex_diff = sign_pow(band_diff, p)

            if metric_par.ignore_freqs_lower_than != [] and band_freq[b - 1] < metric_par.ignore_freqs_lower_than:
                # An option to ignore low-freq bands
                D_bands.set_band(b, o, np.zeros(ex_diff.shape))
            elif metric_par.do_masking:
                k_mask_self = 10 ** metric_par.mask_self  # self-masking
                k_mask_xo = 10 ** metric_par.mask_xo  # masking across orientations
                k_mask_xn = 10 ** metric_par.mask_xn  # masking across neighboring bands

                do_mask_xn = (metric_par.mask_xn > -10)  # skip if the effect is too weak

                self_mask = mutual_masking(b, o, B_R, B_T, metric_par)

                mask_xn = np.zeros(self_mask.shape)
                if b > 0 and do_mask_xn:
                    mask_xn = np.maximum(
                        imresize(mutual_masking(b - 1, o, B_R, B_T, metric_par), output_shape=self_mask.shape), 0)
                if b < (B_T.band_count() - 2) and do_mask_xn:
                    mask_xn += np.maximum(
                        imresize(mutual_masking(b + 1, o, B_R, B_T, metric_par), output_shape=self_mask.shape),
                        0)

                # correct activity for this particular channel
                if do_mask_xo:
                    band_mask_xo = np.maximum(mask_xo - self_mask, 0)
                else:
                    band_mask_xo = 0

                N_mask = (k_mask_self * (np.abs(self_mask)) ** q +
                          k_mask_xo * (np.abs(band_mask_xo)) ** q +
                          k_mask_xn * (np.abs(mask_xn)) ** q)

                D = ex_diff / np.sqrt(N_nCSF ** (2 * p) + N_mask ** 2)

                D_bands.set_band(b, o, sign_pow(D, pf))

            else:
                # NO masking
                D = ex_diff / N_nCSF ** p
                D_bands.set_band(b, o, sign_pow(D, pf))

            diff_mask_b = imresize(diff_mask.astype(float), output_shape=D.shape)
            D_p = D * diff_mask_b

            # Per-band error used for quality predictions

            Q_err += minkowski_sum(D_p.flatten(), 0.8) / B_T.band_count()
    S_map = abs(D_bands.reconstruct())

    keys = ['P_map', 'Q', 'Q_JOD']

    res = dict.fromkeys(keys)

    res['P_map'] = 1 - np.exp(np.log(0.5) * S_map)

    if metric_par.do_sprob_sum:
        si_sigma = 10 ** metric_par.si_sigma * metric_par.pix_per_deg
        res['P_map'] = 1 - np.exp(fast_gauss(np.log(1 - res['P_map'] + 1e-4), si_sigma, True, 0))
    if metric_par.do_pixel_threshold:
        m_thr = 0.01
        is_diff = 1 - np.exp(
            np.log(0.5) * (np.abs(L_adapt_reference - L_adapt_test) / L_adapt_reference / m_thr) ** 3.5)
        res['P_map'] = res['P_map'] * is_diff

    quality_floor = -4
    max_q = 10
    res['Q'] = max_q + quality_floor - np.log(np.exp(quality_floor) + Q_err)

    jod_pars = [0.5200, 1.2812]
    res['Q_JOD'] = 10 - jod_pars[0] * np.maximum(10 - res['Q'], 0) ** jod_pars[1]

    return res


def check_if_values_plausible(img, metric_par):
    # Check if the image is in plausible range and report a warning if not.
    # This is because the metric is often misused and used for with
    # non-absolute luminace data.

    if (~metric_par.disable_lowvals_warning):
        if (np.max(img) <= 1):
            warnings.warn(
                'hdrvdp:lowvals\n The images contain very low physical values, below 1 cd/m^2.\n The passed values are most probably not scaled in absolute units, as requied for this color encoding.\n See "doc hdrvdp" for details.\n')
