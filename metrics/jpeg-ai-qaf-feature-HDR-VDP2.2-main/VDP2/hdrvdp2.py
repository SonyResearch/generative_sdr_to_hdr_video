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
'''

import copy
from VDP2.hdrvdp_parse_options import Metric_par
from VDP2.load_spectral_resp import load_spectral_resp
from VDP2.hdrvdp_visual_pathway import hdrvdp_visual_pathway
from VDP2.clamp import clamp
from VDP2.hdrvdp_ncsf import hdrvdp_ncsf
from VDP2.PyrTools import *
from scipy import signal
from scipy import interpolate
from VDP2.fast_gauss import fast_gauss
from VDP2.imresize import *
from VDP2.geomean import geomean


def mutual_masking(b, o, B_R, B_T, metric_par):
    m = np.minimum(abs(get_band(B_T, b, o)), abs(get_band(B_R, b, o)))
    # simplistic phase - uncertainty mechanism
    # TODO - improve it

    F = np.ones([3, 3])
    m = signal.convolve2d(m, F / F.size, 'same')
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


def hdrvdp2(test, reference, color_encoding, pixels_per_degree, options={}):
    if test.shape != reference.shape:
        warnings.warn('reference and test images must be the same size')
    # if 'options' not in dir():
    #     options = {}

    metric_par = Metric_par(options)

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
        test = display_model_srgb(test)
        reference = display_model_srgb(reference)
    elif metric_par.color_encoding == 'rgb-bt.709':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "rgb-bt.709" color encoding')
        check_if_values_plausible(test[:, :, 1], metric_par)
    elif metric_par.color_encoding == 'xyz':
        if (img_channels != 3):
            warnings.warn('Exactly 3 color channels must be provided for "XYZ" color encoding')
        test = xyz2rgb(test)
        reference = xyz2rgb(reference)
        check_if_values_plausible(test[:, :, 1], metric_par)

    elif metric_par.color_encoding == 'generic':
        if metric_par.spectral_emission not in dir():
            warnings.warn('"spectral_emission" option must be specified when using the "generic" color encoding')

    if (metric_par.surround_l == -1):
        # If surround_l is set to - 1, use the geometric(log) mean of each color
        # channel of the reference image
        metric_par.surround_l = geomean(
            np.reshape(reference, [reference.shape[0] * reference.shape[1], reference.shape[2]]))
    elif (hasattr(metric_par, 'surround_l')):
        metric_par.surround_l = np.repeat(metric_par.surround_l, img_channels)

    if (img_channels == 3 and ~hasattr(metric_par, 'rgb_display')):
        metric_par.rgb_display = 'ccfl-lcd'
    if (metric_par.spectral_emission != []):
        # [tmp, IMG_E] = load_spectral_resp([metric_par.base_dir '/' metric_par.spectral_emission])
        [tmp, IMG_E] = load_spectral_resp(metric_par.spectral_emission)
    elif (hasattr(metric_par, 'rgb_display')):
        [tmp, IMG_E] = load_spectral_resp(f'emission_spectra_{metric_par.rgb_display}.csv')
    elif (img_channels == 1):
        [tmp, IMG_E] = load_spectral_resp('d65.csv')
    else:
        warnings.warn('spectral_emissiom option needs to be specified')

    if (img_channels == 1 and IMG_E.shape[1] > 1):
        # sum - up spectral responses of all channels for luminance - only data
        IMG_E = np.sum(IMG_E, axis=1)

    if (img_channels != IMG_E.shape[1]):
        warnings.warn(
            'Spectral response data is either missing or is specified for different number of color channels than the input image')
    metric_par.spectral_emission = IMG_E

    # Compute spatially - and orientation - selective bands
    # Process reference first to reuse bb_padvalue
    [B_R, L_adapt_reference, band_freq, bb_padvalue] = hdrvdp_visual_pathway(reference, 'reference', metric_par, -1)
    [B_T, L_adapt_test, _, _] = hdrvdp_visual_pathway(test, 'test', metric_par, bb_padvalue)

    L_adapt = (L_adapt_test + L_adapt_reference) / 2
    # precompute CSF
    csf_la = np.logspace(-5, 5, 256)
    csf_log_la = np.log10(csf_la)
    b_count = B_T['sz'].shape[0]

    CSF = np.zeros([len(csf_la), b_count])
    for b in np.arange(0, b_count):
        CSF[:, b] = hdrvdp_ncsf(band_freq[b], csf_la.T, metric_par)

    log_La = np.log10(clamp(L_adapt, csf_la[0], csf_la[-1]))

    D_bands = copy.deepcopy(B_T)

    # Pixels that are actually different
    diff_mask_Three = np.zeros_like(reference)
    diff_mask_Three[(abs(test - reference) / reference) > 0.001] = 1
    # diff_mask = diff_mask_Three[:, :, 0] + diff_mask_Three[:, :, 1] + diff_mask_Three[:, :, 2]
    diff_mask = np.sum(diff_mask_Three, axis=2)
    diff_mask[diff_mask >= 1.0] = 1.0
    Q = 0  # quality correlate

    # For each band
    for b in np.arange(0, b_count):
        # maskingparams
        p = 10 ** metric_par.mask_p
        q = 10 ** metric_par.mask_q
        pf = (10 ** metric_par.psych_func_slope) / p

        # accumulate masking activity across orientations(cross - orientation  masking)
        mask_xo = np.zeros(get_band_size(B_T, b, 0))
        for i in np.arange(0, B_T['sz'][b]):
            mask_xo = mask_xo + mutual_masking(b, i, B_R, B_T, metric_par)  # min(abs(B_T{b, o}), abs(B_R{b, o}) )

        log_La_rs = clamp(imresize(log_La, output_shape=get_band_size(B_T, b, 0)),
                          csf_log_la[0], csf_log_la[-1])
        # per - pixel contrast sensitivity
        CSF_b = interpolate.interp1d(csf_log_la, CSF[:, b])(log_La_rs)

        # REMOVED: Transform CSF linear sensitivity to the non - linear space of the
        # photoreceptor response
        # CSF_b = CSF_b. * 1. / hdrvdp_joint_rod_cone_sens_diff(10. ^ log_La_rs, metric_par);

        band_norm = 2 ** b  # = 1 / n_f from the paper  2 ^ -(b - 1);
        band_mult = 1

        for j in np.arange(0, B_T['sz'][b]):
            # excitation difference
            band_diff = (get_band(B_T, b, j) - get_band(B_R, b, j)) * band_mult

            if (metric_par.do_si_gauss):
                band_scale = band_diff.shape[0] / test.shape[0]
                # Sigma grows with lower frequencies to subtend a similar number of
                # cycles.Note that the scale differs between the bands.
                sigma = 10 ** metric_par.si_size / band_freq[b] * metric_par.pix_per_deg * band_scale
                local_sum = fast_gauss(abs(band_diff), sigma, False)
                ex_diff = (sign_pow(band_diff / band_norm, p - 1) * band_norm) * local_sum
            else:
                ex_diff = sign_pow(band_diff / band_norm, p) * band_norm

            if (b == b_count - 1):
                # base band
                N_nCSF = 1
            else:
                N_nCSF = (1. / CSF_b)

            if (metric_par.do_masking):

                k_mask_self = 10 ** metric_par.mask_self  # self-masking
                k_mask_xo = 10 ** metric_par.mask_xo  # masking across orientations
                k_mask_xn = 10 ** metric_par.mask_xn  # masking across neighboring bands

                self_mask = mutual_masking(b, j, B_R, B_T, metric_par)

                mask_xn = np.zeros(self_mask.shape)
                if (b > 0):
                    mask_xn = np.maximum(
                        imresize(mutual_masking(b - 1, j, B_R, B_T, metric_par), output_shape=self_mask.shape),
                        0) / (band_norm / 2)

                if (b < (b_count - 2)):
                    mask_xn = mask_xn + np.maximum(
                        imresize(mutual_masking(b + 1, j, B_R, B_T, metric_par), output_shape=self_mask.shape),
                        0) / (band_norm * 2)

                # correct activity for this particular channel
                band_mask_xo = np.maximum(mask_xo - self_mask, 0)

                N_mask = band_norm * (k_mask_self * (self_mask / N_nCSF / band_norm) ** q + k_mask_xo * (
                        band_mask_xo / N_nCSF / band_norm) ** q + k_mask_xn * (mask_xn / N_nCSF) ** q)

                D = ex_diff / np.sqrt(N_nCSF ** (2 * p) + N_mask ** 2)
                D_bands = set_band(D_bands, b, j, sign_pow(D / band_norm, pf) * band_norm)
            # end
            else:
                # NO masking
                D = ex_diff / N_nCSF ** p
                D_bands = set_band(D_bands, b, j, sign_pow(D / band_norm, pf) * band_norm)

                # Quality prediction
                # test
            w_f = interpolate.interp1d(metric_par.quality_band_freq, metric_par.quality_band_w)(
                clamp(band_freq[b], metric_par.quality_band_freq[-1], metric_par.quality_band_freq[0]))
            epsilon = 1e-12

            # Mask the pixels that are almost identical in test and
            # reference images.Improves predictions for small localized differences.
            # diff_mask_b = np.array(Image.fromarray(diff_mask.astype(float)).resize(D.shape[::-1]))
            diff_mask_b = imresize(diff_mask.astype(float), output_shape=D.shape)
            D_p = D * diff_mask_b

            Q = Q + np.log(msre(D_p) + epsilon) * w_f / sum(D_bands['sz'])
    res = 100 / (1 + np.exp(metric_par.quality_logistic_q1 * (Q + metric_par.quality_logistic_q2)))

    return res


def check_if_values_plausible(img, metric_par):
    # Check if the image is in plausible range and report a warning if not.
    # This is because the metric is often misused and used for with
    # non-absolute luminace data.

    if (~metric_par.disable_lowvals_warning):

        if (np.max(img) <= 1):
            warnings.warn(
                'hdrvdp:lowvals\n The images contain very low physical values, below 1 cd/m^2.\n The passed values are most probably not scaled in absolute units, as requied for this color encoding.\n See "doc hdrvdp" for details.\n')
