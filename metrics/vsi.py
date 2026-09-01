import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage


def m_vsi(image1, image2):
    """
    Visual Saliency based Index (VSI) between two images.

    Parameters
    ----------
    image1 : np.ndarray
        First image, shape (H, W) or (H, W, 3). uint8 or float.
    image2 : np.ndarray
        Second image, same format/shape as image1.

    Returns
    -------
    sim : float
        Similarity score between two images.
    """

    # Ensure arrays
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # If grayscale, convert to 3-channel
    if image1.ndim == 2 or (image1.ndim == 3 and image1.shape[2] == 1):
        image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
        image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)

    # Convert to float64
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # Constants
    constForVS   = 1.27   # fixed
    constForGM   = 386    # fixed
    constForChrom = 130   # fixed
    alpha = 0.40          # fixed
    lambda_ = 0.020       # fixed
    sigmaF = 1.34         # fixed do not change
    omega0 = 0.0210       # fixed
    sigmaD = 145          # fixed
    sigmaC = 0.001        # fixed

    # Compute visual saliency map using SDSP
    saliencyMap1 = SDSP(image1, sigmaF, omega0, sigmaD, sigmaC)
    saliencyMap2 = SDSP(image2, sigmaF, omega0, sigmaD, sigmaC)

    rows, cols, _ = image1.shape

    # Transform into opponent color space
    L1 = 0.06 * image1[:, :, 0] + 0.63 * image1[:, :, 1] + 0.27 * image1[:, :, 2]
    L2 = 0.06 * image2[:, :, 0] + 0.63 * image2[:, :, 1] + 0.27 * image2[:, :, 2]
    M1 = 0.30 * image1[:, :, 0] + 0.04 * image1[:, :, 1] - 0.35 * image1[:, :, 2]
    M2 = 0.30 * image2[:, :, 0] + 0.04 * image2[:, :, 1] - 0.35 * image2[:, :, 2]
    N1 = 0.34 * image1[:, :, 0] - 0.60 * image1[:, :, 1] + 0.17 * image1[:, :, 2]
    N2 = 0.34 * image2[:, :, 0] - 0.60 * image2[:, :, 1] + 0.17 * image2[:, :, 2]

    # --------------------------
    # Downsample the image
    # --------------------------
    minDimension = min(rows, cols)
    F = max(1, int(np.rint(minDimension / 256.0)))

    aveKernel = np.ones((F, F), dtype=np.float64) / (F * F)

    def downsample_after_average(x):
        ave = convolve2d(x, aveKernel, mode='same', boundary='fill', fillvalue=0)
        return ave[::F, ::F]

    M1 = downsample_after_average(M1)
    M2 = downsample_after_average(M2)
    N1 = downsample_after_average(N1)
    N2 = downsample_after_average(N2)
    L1 = downsample_after_average(L1)
    L2 = downsample_after_average(L2)
    saliencyMap1 = downsample_after_average(saliencyMap1)
    saliencyMap2 = downsample_after_average(saliencyMap2)

    # --------------------------
    # Calculate the gradient map
    # --------------------------
    dx = np.array([[3, 0, -3],
                   [10, 0, -10],
                   [3, 0, -3]], dtype=np.float64) / 16.0
    dy = np.array([[3, 10, 3],
                   [0,  0,  0],
                   [-3, -10, -3]], dtype=np.float64) / 16.0

    IxL1 = convolve2d(L1, dx, mode='same', boundary='fill', fillvalue=0)
    IyL1 = convolve2d(L1, dy, mode='same', boundary='fill', fillvalue=0)
    gradientMap1 = np.sqrt(IxL1 ** 2 + IyL1 ** 2)

    IxL2 = convolve2d(L2, dx, mode='same', boundary='fill', fillvalue=0)
    IyL2 = convolve2d(L2, dy, mode='same', boundary='fill', fillvalue=0)
    gradientMap2 = np.sqrt(IxL2 ** 2 + IyL2 ** 2)

    # --------------------------
    # Calculate VSI
    # --------------------------
    VSSimMatrix = (2 * saliencyMap1 * saliencyMap2 + constForVS) / \
                  (saliencyMap1 ** 2 + saliencyMap2 ** 2 + constForVS)

    gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + constForGM) / \
                        (gradientMap1 ** 2 + gradientMap2 ** 2 + constForGM)

    weight = np.maximum(saliencyMap1, saliencyMap2)

    ISimMatrix = (2 * M1 * M2 + constForChrom) / (M1 ** 2 + M2 ** 2 + constForChrom)
    QSimMatrix = (2 * N1 * N2 + constForChrom) / (N1 ** 2 + N2 ** 2 + constForChrom)

    base = ISimMatrix * QSimMatrix  # can be negative

    with np.errstate(invalid='ignore'):
        chroma_term = np.real(np.power(base.astype(np.complex128), lambda_))

    SimMatrixC = (gradientSimMatrix ** alpha) * VSSimMatrix * chroma_term * weight
    sim = SimMatrixC.sum() / weight.sum()

    return float(sim)


def SDSP(image, sigmaF, omega0, sigmaD, sigmaC):
    """
    SDSP: Saliency Detection by combining Simple Priors.
    """

    oriRows, oriCols, _ = image.shape
    image = image.astype(np.float64)

    # Downsample to 256x256 using bilinear (order=1)
    dsImage = np.zeros((256, 256, 3), dtype=np.float64)
    zoom_y = 256.0 / oriRows
    zoom_x = 256.0 / oriCols
    for c in range(3):
        dsImage[:, :, c] = ndimage.zoom(image[:, :, c], (zoom_y, zoom_x), order=1)

    # Convert to Lab
    lab = RGB2Lab(dsImage)
    LChannel = lab[:, :, 0]
    AChannel = lab[:, :, 1]
    BChannel = lab[:, :, 2]

    # FFT of channels
    LFFT = np.fft.fft2(LChannel)
    AFFT = np.fft.fft2(AChannel)
    BFFT = np.fft.fft2(BChannel)

    rows, cols, _ = dsImage.shape

    LG = logGabor(rows, cols, omega0, sigmaF)

    FinalLResult = np.real(np.fft.ifft2(LFFT * LG))
    FinalAResult = np.real(np.fft.ifft2(AFFT * LG))
    FinalBResult = np.real(np.fft.ifft2(BFFT * LG))

    SFMap = np.sqrt(FinalLResult ** 2 + FinalAResult ** 2 + FinalBResult ** 2)

    # Spatial prior (center bias)
    y_coords = np.arange(rows).reshape(-1, 1)
    x_coords = np.arange(cols).reshape(1, -1)
    centerY = rows / 2.0
    centerX = cols / 2.0
    SDMap = np.exp(-((y_coords - centerY) ** 2 + (x_coords - centerX) ** 2) / (sigmaD ** 2))

    # Color prior (warm colors)
    maxA = AChannel.max()
    minA = AChannel.min()
    normalizedA = (AChannel - minA) / (maxA - minA + 1e-12)

    maxB = BChannel.max()
    minB = BChannel.min()
    normalizedB = (BChannel - minB) / (maxB - minB + 1e-12)

    labDistSquare = normalizedA ** 2 + normalizedB ** 2
    SCMap = 1 - np.exp(-labDistSquare / (sigmaC ** 2))

    VSMap = SFMap * SDMap * SCMap

    # Resize back to original size
    zoom_y_back = oriRows / float(rows)
    zoom_x_back = oriCols / float(cols)
    VSMap = ndimage.zoom(VSMap, (zoom_y_back, zoom_x_back), order=1)

    # mat2gray: normalize to [0, 1]
    VSMap = VSMap - VSMap.min()
    if VSMap.max() > 0:
        VSMap = VSMap / VSMap.max()

    return VSMap


def RGB2Lab(image):
    """
    RGB to Lab conversion matching your MATLAB implementation.
    Input image: float64, range [0, 255].
    Output: Lab image, same shape, float64.
    """
    image = image.astype(np.float64)
    normalizedR = image[:, :, 0] / 255.0
    normalizedG = image[:, :, 1] / 255.0
    normalizedB = image[:, :, 2] / 255.0

    # Gamma correction
    def gamma_correct(channel):
        smaller = channel <= 0.04045
        larger = ~smaller
        out = np.zeros_like(channel)
        out[smaller] = channel[smaller] / 12.92
        out[larger] = ((channel[larger] + 0.055) / 1.055) ** 2.4
        return out

    tmpR = gamma_correct(normalizedR)
    tmpG = gamma_correct(normalizedG)
    tmpB = gamma_correct(normalizedB)

    # RGB to XYZ (D65-ish coefficients from your code)
    X = tmpR * 0.4124564 + tmpG * 0.3575761 + tmpB * 0.1804375
    Y = tmpR * 0.2126729 + tmpG * 0.7151522 + tmpB * 0.0721750
    Z = tmpR * 0.0193339 + tmpG * 0.1191920 + tmpB * 0.9503041

    epsilon = 0.008856  # actual CIE standard
    kappa = 903.3       # actual CIE standard

    Xr = 0.9642  # reference white D50
    Yr = 1.0
    Zr = 0.8251

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    def f(t):
        greater = t > epsilon
        smaller = ~greater
        out = np.zeros_like(t)
        out[greater] = np.cbrt(t[greater])
        out[smaller] = (kappa * t[smaller] + 16.0) / 116.0
        return out

    fx = f(xr)
    fy = f(yr)
    fz = f(zr)

    labImage = np.zeros_like(image)
    labImage[:, :, 0] = 116.0 * fy - 16.0
    labImage[:, :, 1] = 500.0 * (fx - fy)
    labImage[:, :, 2] = 200.0 * (fy - fz)

    return labImage


def logGabor(rows, cols, omega0, sigmaF):
    """
    Log-Gabor filter construction.
    """

    # Meshgrid similar to MATLAB
    u1, u2 = np.meshgrid(
        (np.arange(1, cols + 1) - (cols // 2 + 1)) / (cols - (cols % 2)),
        (np.arange(1, rows + 1) - (rows // 2 + 1)) / (rows - (rows % 2))
    )

    mask = np.ones((rows, cols), dtype=np.float64)
    mask[u1 ** 2 + u2 ** 2 > 0.25] = 0.0

    u1 *= mask
    u2 *= mask

    # ifftshift
    u1 = np.fft.ifftshift(u1)
    u2 = np.fft.ifftshift(u2)

    radius = np.sqrt(u1 ** 2 + u2 ** 2)
    radius[0, 0] = 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        LG = np.exp(- (np.log(radius / omega0) ** 2) / (2 * (sigmaF ** 2)))

    LG[0, 0] = 0

    return LG
