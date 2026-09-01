''' HDRVDP_PARSE_OPTIONS (internal) parse HDR-VDP options and create two
      structures: view_cond with viewing conditions and metric_par with metric
      parameters

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

      Defaults
      '''

import numpy as np
import warnings


class Metric_par(object):

    def __init__(self, task, options):

        self.debug = False

        # self.use_gpu = True
        self.use_gpu = False

        self.quiet = False  # Disable warning messages

        self.base_dir = ''  # Used for calibration
        self.threshold_p = 0.75
        self.p_obs = 0.8

        self.ms_decomp = 'spyr'
        self.do_new_diff = False
        self.do_sprob_sum = False

        self.mtf = 'hdrvdp'

        self.use_gpu = False

        self.ignore_freqs_lower_than = []

        # Peak contrast from Daly's CSF for L_adapt = 30 cd/m^2
        # daly_peak_contrast_sens = 0.006894596

        # self.sensitivity_correction = daly_peak_contrast_sens / 10 ** -2.355708
        # self.sensitivity_correction = daly_peak_contrast_sens / 10 ** -2.4

        self.view_dist = 0.5

        self.do_pixel_threshold = False

        self.age = 24

        self.do_aesl = True  # Empirical (neural) sensitivity loss as a function of age
        self.do_aod = True  # Aging optical density (Pokorny 1987).
        self.do_slum = True  # light reduction due to senile miosis (Unified Formula based on the Stanley and Davies function [Watson & Yellott 2012]

        # Below are the experimental aging models, found to perform worse then the once above. Disabled by default.
        self.do_asl = False  # Age related sensitivity loss - averaged [Sagawa 2001, JOSA, A, 18 (11), 2659-2667, 2001]
        self.do_aslw = False  # Age related sensitivity loss - wavelength dependent (note, do not use both do_asl and do_aslw) [Sagawa 2001, JOSA, A, 18 (11), 2659-2667, 2001]
        self.do_sl = False  # Senile miosis (smaller pupil at old age and low light) [Sloane 1988]

        # self.aesl_slope_age = -3.85951; # not used anymore
        self.aesl_slope_freq = -2.711
        self.aesl_base = -0.125539
        self.do_pu_dilate = False
        self.masking_norm = 0  # The L-p norm used for summing up energy from the band for masking
        self.masking_pool_size = 3  # The size of the kernel used for pooling masking signal energy

        self.spectral_emission = []

        self.w_low = 0
        self.w_med = 0
        self.w_high = 0

        self.orient_count = 4  # the number of orientations to consider

        # Various optional features
        self.do_masking = True
        self.noise_model = True
        self.do_quality = False  # Do quality predictions
        self.do_quality_raw_data = False  # for development purposes only
        self.do_si_gauss = False
        self.si_size = 1.008
        self.si_mode = 'none'

        self.do_civdm = False  # contrast independent structural difference metric
        self.disable_lowvals_warning = False

        self.steerpyr_filter = 'sp3Filters'

        # Psychometric function
        self.psych_func_slope = np.log10(3.5)
        # self.beta=self.psych_func_slope-self.mask_p;

        self.si_size = -0.034244

        self.pu_dilate = 3

        # Spatial summation
        self.si_slope = -0.850147
        self.si_sigma = -0.000502005
        self.si_ampl = 0

        # Cone and rod cvi functions
        self.cvi_sens_drop = 0.0704457
        self.cvi_trans_slope = 0.0626528
        self.cvi_low_slope = -0.00222585

        self.rod_sensitivity = 0
        # self.rod_sensitivity = -0.383324
        self.cvi_sens_drop_rod = -0.58342

        # Achromatic CSF
        self.csf_m1_f_max = 0.425509
        self.csf_m1_s_high = -0.227224
        self.csf_m1_s_low = -0.227224
        self.csf_m1_exp_low = np.log10(2)

        par = [0.061466549455263, 0.99727370023777070]  # old parametrization of MTF
        self.mtf_params_a = [par[1] * 0.426, par[1] * 0.574, (1 - par[1]) * par[0], (1 - par[1]) * (1 - par[0])]
        self.mtf_params_b = [0.028, 0.37, 37, 360]

        # self.quality_band_freq = [15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344]
        self.quality_band_freq = [60, 30, 15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344, 0.1172]

        # self.quality_band_w = [0.2963, 0.2111, 0.1737, 0.0581, -0.0280, 0.0586, 0.2302]

        # New quality calibration: LDR + HDR datasets - paper to be published
        # self.quality_band_w = [0.2832, 0.2142, 0.2690, 0.0398, 0.0003, 0.0003, 0.0002]
        self.quality_band_w = [0, 0.2832, 0.2832, 0.2142, 0.2690, 0.0398, 0.0003, 0.0003, 0, 0]

        self.quality_logistic_q1 = 3.455
        self.quality_logistic_q2 = 0.8886

        self.calibration_date = '07 April 2020'

        self.surround = 'none'

        self.ms_decomp = 'spyr'
        self.steerpyr_filter = 'sp0Filters'

        # HDR-VDP-csf refitted on 19/04/2020
        self.csf_params = [
            [0.699404, 1.26181, 4.27832, 0.361902, 3.11914],
            [1.00865, 0.893585, 4.27832, 0.361902, 2.18938],
            [1.41627, 0.84864, 3.57253, 0.530355, 3.12486],
            [1.90256, 0.699243, 3.94545, 0.68608, 4.41846],
            [2.28867, 0.530826, 4.25337, 0.866916, 4.65117],
            [2.46011, 0.459297, 3.78765, 0.981028, 4.33546],
            [2.5145, 0.312626, 4.15264, 0.952367, 3.22389]
        ]

        self.csf_lums = [0.0002, 0.002, 0.02, 0.2, 2, 20, 150]

        self.csf_sa = [315.98, 6.7977, 1.6008, 0.25534]
        self.csf_sr = [1.1732, 1.32, 1.095, 0.5547, 2.9899, 1.8]  # rod sensitivity function

        if task == 'side-by-side' or task == 'sbs':
            # Fitted to LocVisVC (http://dx.doi.org/10.1109/CVPR.2019.00558) using after all updates in HDR-VDP-3.0.6 (24/05/2020)
            # marking_2020_05_sbs-stdout

            self.base_sensitivity_correction = 0.203943775672
            self.mask_self = 1.41846291721
            self.mask_xn = 0.136877512403
            self.mask_xo = -50
            self.mask_q = 0.108934275615

            self.mask_p = 0.3424
            self.do_sprob_sum = True
            self.psych_func_slope = 0.34

            self.si_sigma = -0.502280453708

            self.do_robust_pdet = True
            self.do_spatial_total_pooling = False

        elif task == 'flicker':
            # Fitted to compression_flicker dataset  (23/05/2020)
            # test_results/marking_flicker_peak_mask-all-stdout

            self.base_sensitivity_correction = 0.359225308021 + 0.14178269059
            self.mask_self = 1.48041073222
            self.mask_xn = 0.00886149078207
            self.mask_xo = -50
            self.mask_q = 0.11339123107

            self.mask_p = 0.3424
            self.do_sprob_sum = True
            self.psych_func_slope = 0.591472024776

            self.si_sigma = -0.511779476487

            self.do_robust_pdet = True
            self.do_spatial_total_pooling = False

        elif task == 'quality':
            # The same parameters as side-by-side

            self.base_sensitivity_correction = 0.203943775672
            self.mask_self = 1.41846291721
            self.mask_xn = 0.136877512403
            self.mask_xo = -50
            self.mask_q = 0.108934275615

            self.mask_p = 0.3424
            self.do_sprob_sum = True
            self.psych_func_slope = 0.34

            self.si_sigma = -0.502280453708

            self.do_robust_pdet = True
            self.do_spatial_total_pooling = False
        if options !='{}':
            for key in options:
                if key == 'pixels_per_degree':
                    self.pix_per_deg = options[key]
                elif key == 'viewing_distance':
                    self.view_dist = options[key]
                elif key == 'sensitivity_correction':
                    self.sensitivity_correction = self.base_sensitivity_correction + options[key]
                elif key == 'age':
                    self.age = options[key]
                else:
                    warnings.warn('invalid options')
