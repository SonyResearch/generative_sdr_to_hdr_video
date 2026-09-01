import numpy as np


def hdrvdp_colorspace_transform(in_, M):
    """
    Transform an image or color vector (Nx3) to another color space using matrix M

    out = hdrvdp_colorspace_transform(in, M)

    "in" could be an image [width x height x 3] or [rows x 3] matrix of colour vectors
    M is a colour transformation matrix so that dest = M * src (where src is a column vector)
    """
    if M.shape != (3, 3):
        raise ValueError('Colour transformation must be a 3x3 matrix')

    if len(in_.shape) == 2:
        out = np.dot(in_, M.T)
    else:
        out = np.dot(M, in_.reshape((in_.shape[0] * in_.shape[1], 3), order="F").T).T.reshape(
            (in_.shape[0], in_.shape[1], 3), order="F")
    return out
