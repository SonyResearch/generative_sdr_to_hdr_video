import numpy as np


def geomean(x, dim=None, nanflag=None):
    """
    GEOMEAN Geometric mean.
    M = GEOMEAN(X) returns the geometric mean of the values in X. When X
    is an n element vector, M is the n-th root of the product of the n
    elements in X. For a matrix input, M is a row vector containing the
    geometric mean of each column of X. For N-D arrays, GEOMEAN operates
    along the first non-singleton dimension.

    GEOMEAN(X,'all') is the geometric mean of all the elements of X.

    GEOMEAN(X,DIM) takes the geometric mean along dimension DIM of X.

    GEOMEAN(X,VECDIM) finds the geometric mean of the elements of X based
    on the dimensions specified in the vector VECDIM.

    GEOMEAN(...,NANFLAG) specifies how NaN values are treated.
      'includenan' - Include NaN values when computing the geometric mean,
                     resulting in NaN (the default).
      'omitnan'    - Ignore all NaN values in the input.

    See also MEAN, HARMMEAN, TRIMMEAN.
    """

    if (x < 0).any() or not np.isreal(x).all():
        raise ValueError('Bad Data')

    if dim is None:
        # Figure out which dimension sum will work along.
        dim = np.argmin(x.shape == 1)

    # Take the n-th root of the product of elements of X, along dimension DIM.
    if nanflag == 'omitnan':
        m = np.exp(np.nanmean(np.log(x), axis=dim))
    else:
        m = np.exp(np.mean(np.log(x), axis=dim))

    return m
