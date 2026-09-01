import numpy as np
from matplotlib import pyplot as py


def log_luminance(X):
    if X.shape[2] == 3:
        Y = X[:, :, 0] * 0.212656 + X[:, :, 1] * 0.715158 + X[:, :, 2] * 0.072186
    else:
        Y = X

    Y[Y <= 0] = np.min(Y[Y > 0])
    l = np.log(Y)
    return l


def vis_tonemap(b, dr):
    t = 3
    b_min = np.min(b)
    b_max = np.max(b)
    b_scale = np.linspace(b_min, b_max, 1024)
    b_scale = np.concatenate((b_scale, [np.inf]))
    b_p, _ = np.histogram(b.T.flatten(), b_scale)

    b_p = b_p / np.sum(b_p)
    sum_b_p = np.sum(np.power(b_p, 1 / t))
    dy = np.power(b_p, 1 / t) / sum_b_p
    v = np.cumsum(dy) * dr + (1 - dr) / 2
    tmo_img = np.interp(b, b_scale[:-1], v)
    return tmo_img


def get_luminance(img):
    # Return 2D matrix of luminance values for 3D matrix with an RGB image
    dims = np.max(np.nonzero(np.array(img.shape) > 1)) + 1
    if dims == 3:
        Y = img[:, :, 0] * 0.212656 + img[:, :, 1] * 0.715158 + img[:, :, 2] * 0.072186
    elif dims == 1 or dims == 2:
        Y = img
    else:
        raise ValueError('get_luminance: wrong matrix dimension')
    return Y


def hdrvdp_visualize(p1, varargin):
    P = p1
    tmo_img = vis_tonemap(log_luminance(varargin), 0.6)

    P[P < 0] = 0
    P[P > 1] = 1

    color_map = np.array([[0.2, 0.2, 1.0],
                          [0.2, 1.0, 1.0],
                          [0.2, 1.0, 0.2],
                          [1.0, 1.0, 0.2],
                          [1.0, 0.2, 0.2]])

    color_map_in = np.array([0, 0.25, 0.5, 0.75, 1])

    color_map_l = color_map @ [0.2126, 0.7152, 0.0722]
    color_map_ch = color_map / np.tile(color_map_l[:, None], [1, 3])

    map = np.zeros([P.shape[0], P.shape[1], 3])
    map[:, :, 0] = np.interp(P, color_map_in, color_map_ch[:, 0])
    map[:, :, 1] = np.interp(P, color_map_in, color_map_ch[:, 1])
    map[:, :, 2] = np.interp(P, color_map_in, color_map_ch[:, 2])

    map = map * np.tile(tmo_img[:, :, None], [1, 1, 3])

    return map
