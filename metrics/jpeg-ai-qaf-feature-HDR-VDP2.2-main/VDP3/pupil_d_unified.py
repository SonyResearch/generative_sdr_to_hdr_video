import numpy as np


def pupil_d_unified(L, area, age):
    y0 = 28.58  # Reference age from the paper
    y = np.clip(age, 20, 83)  # The formula applies to this age-range

    d_sd = pupil_d_stanley_davies(L, area)
    d = d_sd + (y - y0) * (0.02132 - 0.009562 * d_sd)

    return d


def pupil_d_stanley_davies(L, area):
    La = L * area

    d = 7.75 - 5.75 * ((La / 846) ** 0.41 / ((La / 846) ** 0.41 + 2))

    return d
