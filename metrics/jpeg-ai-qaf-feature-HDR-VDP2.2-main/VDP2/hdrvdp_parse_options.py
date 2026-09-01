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

    def __init__(self, options):
        # Peak contrast from Daly's CSF for L_adapt = 30 cd/m^2
        daly_peak_contrast_sens = 0.006894596

        # metric_par.sensitivity_correction = daly_peak_contrast_sens / 10.^-2.355708;
        self.sensitivity_correction = daly_peak_contrast_sens / np.power(10, -2.4)

        self.view_dist = 0.5

        self.spectral_emission = []

        self.orient_count = 4  # the number of orientations to consider

        # Various optional features
        self.do_masking = True
        self.do_mtf = True
        self.do_spatial_pooling = True
        self.noise_model = True
        self.do_quality_raw_data = False  # for development purposes only
        self.do_si_gauss = False
        self.si_size = 1.008

        # Warning messages
        self.disable_lowvals_warning = False

        self.steerpyr_filter = 'sp3Filters'

        self.mask_p = 0.544068
        self.mask_self = 0.189065
        self.mask_xo = 0.449199
        self.mask_xn = 1.52512
        self.mask_q = 0.49576
        self.si_size = -0.034244

        self.psych_func_slope = np.log10(3.5)
        self.beta = self.psych_func_slope - self.mask_p

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

        self.csf_params = [[0.0160737, 0.991265, 3.74038, 0.50722, 4.46044],
                           [0.383873, 0.800889, 3.54104, 0.682505, 4.94958],
                           [0.929301, 0.476505, 4.37453, 0.750315, 5.28678],
                           [1.29776, 0.405782, 4.40602, 0.935314, 5.61425],
                           [1.49222, 0.334278, 3.79542, 1.07327, 6.4635],
                           [1.46213, 0.394533, 2.7755, 1.16577, 7.45665]]

        self.csf_lums = [0.002, 0.02, 0.2, 2, 20, 150]

        self.csf_sa = [30.162, 4.0627, 1.6596, 0.2712]

        self.csf_sr_par = [1.1732, 1.1478, 1.2167, 0.5547, 2.9899, 1.1414]  # rod sensitivity function

        par = [0.061466549455263, 0.99727370023777070]  # old parametrization of MTF
        self.mtf_params_a = [par[1] * 0.426, par[1] * 0.574, (1 - par[1]) * par[0], (1 - par[1]) * (1 - par[0])]
        self.mtf_params_b = [0.028, 0.37, 37, 360]

        self.quality_band_freq = [15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344]

        # metric_par.quality_band_w = [0.2963    0.2111    0.1737    0.0581   -0.0280    0.0586    0.2302]

        # New quality calibration: LDR + HDR datasets - paper to be published
        self.quality_band_w = [0.2832, 0.2142, 0.2690, 0.0398, 0.0003, 0.0003, 0.0002]

        self.quality_logistic_q1 = 3.455
        self.quality_logistic_q2 = 0.8886

        self.calibration_date = '30 Aug 2011'

        self.surround_l = 1e-5

        if options != '{}':
            for key in options:
                if key == 'pixels_per_degree':
                    self.pix_per_deg = options[key]
                elif key == 'viewing_distance':
                    self.view_dist = options[key]
                elif key == 'peak_sensitivity':
                    self.sensitivity_correction = daly_peak_contrast_sens / np.power(10, float(-options[key]))
                elif key == 'surround_l':
                    self.surround_l = options[key]
                elif key == 'spectral_emission':
                    self.spectral_emission = options[key]
                else:
                    warnings.warn('invalid options')
