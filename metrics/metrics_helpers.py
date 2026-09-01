import sys
from pathlib import Path

_metrics_dir = Path(__file__).resolve().parent
path_hdr_vdp3 = _metrics_dir / "jpeg-ai-qaf-feature-HDR-VDP2.2-main"
sys.path.insert(0, str(path_hdr_vdp3))
# pycvvdp lives inside ColorVideoVDP (import as pycvvdp)
sys.path.insert(0, str(_metrics_dir / "ColorVideoVDP"))

from VDP3.hdrvdp3 import hdrvdp3
from pu21 import PU21Encoder
from vsi import m_vsi
from NRVQA.piqe import piqe as piqe_func
pu21 = PU21Encoder("banding_glare")
import torch

# PyTorch 2.6+ defaults to weights_only=True; cdfvd and other checkpoints need weights_only=False
_orig_torch_load = torch.load
def _torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from cdfvd import fvd
import pycvvdp

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

def read_image(path):
    if path.endswith(".hdr") or path.endswith(".exr"):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    img = np.clip(img, 0, None)
    return img

def initialize_cvvdp(display_name='ours_standard_hdr_linear', heatmap='threshold', device=None):
    cvvdp_obj = pycvvdp.cvvdp(display_name=display_name, heatmap=heatmap, device=device)
    return cvvdp_obj

def initialize_fid(device="cuda"):
    fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    return fid_metric

def initialize_fvd(device="cuda"):
    fvd_metric = fvd.cdfvd(model='videomae', device=device)
    return fvd_metric

def pu(img):
    return pu21.encode(img).astype(np.float32)

def reinhard_tonemap(img):
    return img / (1.0 + img)

@torch.no_grad()
def lpips(pred, gt, device="cuda"):

    pred = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).to(device).float()
    gt   = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).to(device).float()

    lpips_metric.to(device)
    return float(lpips_metric(pred, gt).item())

@torch.no_grad()
def cvvdp(preds, gt, metric):

    preds = np.stack(preds, axis=0).astype(np.float32)
    gts = np.stack(gt, axis=0).astype(np.float32)
    return metric.predict(preds, gts, dim_order="FHWC", frames_per_second=30)[0].item()

def hdr_vdp3(pred, gt): 
    pred = np.clip(pred, 0, None)
    gt = np.clip(gt, 0, None)
    jod = hdrvdp3('quality', pred, gt, 'rgb-bt.709', 30)
    return jod['Q_JOD']

def psnr(pred, gt):
    mse = np.mean((pred - gt)**2)
    return 10 * np.log10((256**2) / mse)

def vsi(pred, gt):
    return m_vsi(pred, gt)

def piqe(pred):
    piqe_out = piqe_func(pred.astype(np.float32))
    return piqe_out[0]

# def pu_piqe(pred):
#     pu21_pred = pu21.encode(pred)
#     piqe_out = piqe(pu21_pred.astype(np.float32))
#     return piqe_out[0]

@torch.no_grad()
def fid_update(pred_video, gt_video, fid_metric):
    pred_video = np.stack(pred_video, axis=0)  # (T, H, W, C)
    gt_video = np.stack(gt_video, axis=0)      # (T, H, W, C)

    # Convert to torch tensors and add batch dimension
    pred_tensor = torch.from_numpy(pred_video).permute(0, 3, 1, 2).float().to(fid_metric.device)
    gt_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2).float().to(fid_metric.device)

    # Update FID metric with the features from the current video pair
    fid_metric.update(pred_tensor, real=False)
    fid_metric.update(gt_tensor, real=True)

@torch.no_grad()
def fvd_update(pred_video, gt_video, fvd_metric):
    pred_video = np.stack(pred_video, axis=0)  # (T, H, W, C)
    gt_video = np.stack(gt_video, axis=0)      # (T, H, W, C)

    # VideoMAE (cdfvd) internally divides by 255; it expects pixel range [0, 255].
    pred_video = (np.clip(pred_video, 0, 1) * 255.0).astype(np.float32)
    gt_video = (np.clip(gt_video, 0, 1) * 255.0).astype(np.float32)

    fvd_metric.add_fake_stats(pred_video[None, :])  # Add batch dimension
    fvd_metric.add_real_stats(gt_video[None, :])    # Add batch dimension

 
    
def compute_fid(fid_metric):
    # Compute FID score from the accumulated features
    return float(fid_metric.compute().item())

def compute_fvd(fvd_metric):
    return float(fvd_metric.compute_fvd_from_stats())


def align_hdr_pred_to_gt(pred, gt, percentile_low=10, percentile_high=90, eps=1e-8, ab=None):
    """
    Align HDR prediction to GT using least-squares scale+bias (a, b),
    computed only on pixels whose GT luminance lies between given percentiles.

    Returns:
        aligned_pred, (a, b)
    """

    pred = np.clip(pred, 0, None)
    gt_clamped = np.clip(gt, 0, 1000)

    if ab is not None:
        a, b = float(ab[0]), float(ab[1])
        aligned_pred = np.clip(pred * a + b, 0, 1000)
        return aligned_pred, gt_clamped, (a, b)
    # Compute luminance mask from GT
    gt_lum = np.sum(gt, axis=2)
    p10, p90 = np.percentile(gt_lum, percentile_low), np.percentile(gt_lum, percentile_high)
    mask = (gt_lum >= p10) & (gt_lum <= p90)

    pred_masked = pred[mask]   # (N, 3)
    gt_masked   = gt[mask]     # (N, 3)

    # Flatten over channels
    x = pred_masked.reshape(-1)
    y = gt_masked.reshape(-1)

    # Solve min ||a*x + b - y||^2
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

    aligned_pred = pred * a + b
    #clamp aligned pred and gt to 1000 max
    aligned_pred = np.clip(aligned_pred, 0, 1000)
    return aligned_pred, gt_clamped, (a, b)

def pre_hdr_p3(img):
    eps = 1e-8
    flat = img.reshape(-1)

    # 0.1 and 99.9 percentiles
    p_low = np.percentile(flat, 0.1)
    p_high = np.percentile(flat, 99.9)

    c = 1.0
    d = 1000.0

    # linear mapping
    out = (img - p_low) * (d - c) / (p_high - p_low + eps) + c

    # clamp negative values
    out = np.clip(out, c, d)

    return out

def polyfit_ridge(x, y, deg, lam=1e-3, w0=None):
    # Build Vandermonde matrix
    V = np.vander(x, deg+1)

    # Prior center
    if w0 is None:
        w0 = np.zeros(deg+1)
        #make linear term 1
        w0[-2] = 1.0

    # Solve (V^T V + lam I) w = V^T y + lam w0
    A = V.T @ V + lam * np.eye(deg+1)
    b = V.T @ y + lam * w0
    w = np.linalg.solve(A, b)

    return w

import numpy as np
import cv2

def downsample_inter_area(img, factor):
    """
    Downsample using cv2.INTER_AREA (proper averaging).
    """
    if factor <= 1:
        return img
    H, W = img.shape[:2]
    return cv2.resize(
        img,
        (W // factor, H // factor),
        interpolation=cv2.INTER_AREA,
    )

def fit_and_apply_crf(
    pred,
    gt,
    deg=3,
    down_factor=4,
    p_low=10,
    p_high=90,
    coeffs=None,
):
    """
    Fit a per-channel polynomial CRF in log space (pred → gt) using
    only values where gt lies between chosen percentiles.
    Applies the CRF to the full-resolution pred.
    """
    eps = 1
    H, W, C = pred.shape
    assert C == 3

    if coeffs is not None:
        coeffs = np.asarray(coeffs, dtype=np.float32)
        assert coeffs.shape == (3, deg + 1)
        pred_safe = np.clip(pred, eps, None)
        log_p = np.log(pred_safe)
        corrected = np.zeros_like(pred_safe)
        for c in range(3):
            corrected[..., c] = np.exp(np.polyval(coeffs[c], log_p[..., c]))
        return corrected, coeffs

    # --- Downsample for fitting using INTER_AREA ---
    pred_fit = downsample_inter_area(pred, down_factor)
    gt_fit   = downsample_inter_area(gt,   down_factor)

    # --- Flatten ---
    pred_f = pred_fit.reshape(-1, 3)
    gt_f   = gt_fit.reshape(-1, 3)
    num_pixels = pred_f.shape[0] * pred_f.shape[1]

    # --- Compute percentiles per channel on gt ---
    p_lo_vals = np.percentile(gt_f, p_low,  axis=0)
    p_hi_vals = np.percentile(gt_f, p_high, axis=0)

    # --- Valid mask: positive + percentile range ---
    valid_positive = (pred_f > 0).all(axis=1) & (gt_f > 0).all(axis=1)
    valid_range = np.ones(len(gt_f), dtype=bool)
    for c in range(3):
        valid_range &= (gt_f[:, c] >= p_lo_vals[c]) & (gt_f[:, c] <= p_hi_vals[c])

    valid = valid_positive & valid_range

    pred_f = pred_f[valid]
    gt_f   = gt_f[valid]

    # --- Log regression ---
    log_pred = np.log(pred_f + eps)
    log_gt   = np.log(gt_f   + eps)

    coeffs = np.zeros((3, deg + 1), dtype=np.float32)
    for c in range(3):
        x = log_pred[:, c]
        y = log_gt[:, c]
        coeffs[c] = polyfit_ridge(
            x, y, deg,
            lam=0.1 * num_pixels / (deg + 1)
        )

    # --- Apply CRF to full-res pred ---
    pred_safe = np.clip(pred, eps, None)
    log_pred_full = np.log(pred_safe)

    corrected = np.zeros_like(pred_safe)
    for c in range(3):
        p = coeffs[c]
        log_corr = np.polyval(p, log_pred_full[..., c])
        corrected[..., c] = np.exp(log_corr)

    return corrected, coeffs


# ---------------------------------------------------------------------------
# Hanji et al. (SIGGRAPH 2022) CRF correction
# ---------------------------------------------------------------------------
# Reference: "Comparison of single image HDR reconstruction methods —
#  the caveats of quality assessment" (Hanji et al., SIGGRAPH 2022).
# Decouples luminance (PQ-space cubic) and chrominance (2D cubic in CIE u'v')
# corrections, unlike the per-channel log-space fit above.

_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = (2523.0 / 4096.0) * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = (2413.0 / 4096.0) * 32.0
_PQ_C3 = (2392.0 / 4096.0) * 32.0
_PQ_LPEAK = 10000.0

_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)


def _pq_encode(L):
    """Absolute luminance (cd/m^2) -> PQ code in [0, 1]."""
    Ln = np.clip(L, 0.0, _PQ_LPEAK) / _PQ_LPEAK
    Lm = np.power(np.maximum(Ln, 0.0), _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * Lm) / (1.0 + _PQ_C3 * Lm), _PQ_M2)


def _pq_decode(P):
    """PQ code in [0, 1] -> absolute luminance (cd/m^2)."""
    Pm = np.power(np.clip(P, 0.0, 1.0), 1.0 / _PQ_M2)
    num = np.maximum(Pm - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * Pm
    return _PQ_LPEAK * np.power(num / np.maximum(den, 1e-12), 1.0 / _PQ_M1)


def _rgb_to_xyz(rgb):
    return rgb @ _RGB2XYZ.T


def _xyz_to_rgb(xyz):
    return xyz @ _XYZ2RGB.T


def _xyz_to_Yuv(xyz, eps=1e-12):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = X + 15.0 * Y + 3.0 * Z + eps
    return Y, 4.0 * X / denom, 9.0 * Y / denom


def _Yuv_to_xyz(Y, up, vp, eps=1e-6):
    vp_safe = np.where(np.abs(vp) < eps, eps, vp)
    X = Y * (9.0 * up) / (4.0 * vp_safe)
    Z = Y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp_safe)
    return np.stack([X, Y, Z], axis=-1)


def _chroma_basis(u, v):
    """2D cubic basis with cross term: [u^3, v^3, u^2, v^2, u*v, u, v, 1]."""
    return np.stack(
        [u**3, v**3, u**2, v**2, u * v, u, v, np.ones_like(u)],
        axis=1,
    )


def fit_and_apply_crf_hanji(
    pred,
    gt,
    down_factor=4,
    lum_deg=3,
    chroma_lam_scale=0.01,
    pq_median_scale=500.0,
):
    """
    CRF correction following Hanji et al., SIGGRAPH 2022.

    Differences from fit_and_apply_crf (the per-channel log-space version):
      - Luminance:   one cubic fit in PQ space (no regularization, as in paper).
      - Chrominance: 2D cubic in CIE u'v' with Tikhonov regularization and
                     identity prior (so neutral colors stay neutral).
      - No per-channel drift — hue is preserved by construction.

    Returns:
        corrected (same dtype as pred), info dict with fitted weights and medians.
    """
    assert pred.shape == gt.shape and pred.shape[-1] == 3

    pred64 = np.clip(pred.astype(np.float64), 0.0, None)
    gt64   = np.clip(gt.astype(np.float64),   0.0, None)

    # 1. Downsample for fit
    pred_fit = downsample_inter_area(pred64, down_factor)
    gt_fit   = downsample_inter_area(gt64,   down_factor)

    # 2. Convert to (Y, u', v')
    Yp, up_p, vp_p = _xyz_to_Yuv(_rgb_to_xyz(pred_fit))
    Yg, up_g, vp_g = _xyz_to_Yuv(_rgb_to_xyz(gt_fit))

    med_p = np.median(Yp[Yp > 0]) if np.any(Yp > 0) else 1.0
    med_g = np.median(Yg[Yg > 0]) if np.any(Yg > 0) else 1.0
    med_p = max(float(med_p), 1e-12)
    med_g = max(float(med_g), 1e-12)

    # 3. PQ-encode, fit luminance cubic (no regularization — paper Eq. 3, λ=0)
    Pp = _pq_encode(Yp / med_p * pq_median_scale)
    Pg = _pq_encode(Yg / med_g * pq_median_scale)

    valid = (
        np.isfinite(Pp) & np.isfinite(Pg) &
        np.isfinite(up_p) & np.isfinite(vp_p) &
        np.isfinite(up_g) & np.isfinite(vp_g) &
        (Yp > 0) & (Yg > 0)
    )
    Pp_v, Pg_v = Pp[valid], Pg[valid]
    up_p_v, vp_p_v = up_p[valid], vp_p[valid]
    up_g_v, vp_g_v = up_g[valid], vp_g[valid]

    V_lum = np.vander(Pp_v, lum_deg + 1)
    w_lum, *_ = np.linalg.lstsq(V_lum, Pg_v, rcond=None)

    # 4. Chrominance fit: 2D cubic in u'v' with Tikhonov + identity prior (paper Eq. 4-5)
    X_ch = _chroma_basis(up_p_v, vp_p_v)       # [N, 8]
    Y_ch = np.stack([up_g_v, vp_g_v], axis=1)  # [N, 2]
    N, K = X_ch.shape
    lam = chroma_lam_scale * N / K

    # Identity prior: u'->u' (index 5), v'->v' (index 6), rest zero
    W0 = np.zeros((K, 2), dtype=np.float64)
    W0[5, 0] = 1.0
    W0[6, 1] = 1.0

    A = X_ch.T @ X_ch + lam * np.eye(K)
    B = X_ch.T @ Y_ch + lam * W0
    W_ch = np.linalg.solve(A, B)   # [8, 2]

    # 5. Apply to full-resolution prediction
    xyz_full = _rgb_to_xyz(pred64)
    Yf, uf, vf = _xyz_to_Yuv(xyz_full)

    # Luminance: encode with pred's median, apply poly, decode, rescale to GT's scale
    Pf = _pq_encode(Yf / med_p * pq_median_scale)
    Pf_corr = np.clip(np.polyval(w_lum, Pf), 0.0, 1.0)
    Yf_corr = _pq_decode(Pf_corr) / pq_median_scale * med_g

    # Chrominance: apply 2D polynomial
    uf_flat, vf_flat = uf.reshape(-1), vf.reshape(-1)
    uv_corr = _chroma_basis(uf_flat, vf_flat) @ W_ch
    uf_corr = uv_corr[:, 0].reshape(uf.shape)
    vf_corr = uv_corr[:, 1].reshape(vf.shape)

    # Recombine (Y, u', v') -> XYZ -> linear RGB
    xyz_corr = _Yuv_to_xyz(Yf_corr, uf_corr, vf_corr)
    rgb_corr = np.clip(_xyz_to_rgb(xyz_corr), 0.0, None)

    info = {
        "w_lum": w_lum,
        "W_chroma": W_ch,
        "median_pred": med_p,
        "median_gt": med_g,
    }
    return rgb_corr.astype(pred.dtype, copy=False), info


def apply_crf_hanji_fitted(pred, info, pq_median_scale=500.0):
    """
    Apply a pre-fitted Hanji CRF to pred without re-fitting.
    `info` must be the dict returned by fit_and_apply_crf_hanji
    (keys: w_lum, W_chroma, median_pred, median_gt).
    Used for the first-frame-fit path: fit once on frame 0, apply to all frames.
    """
    w_lum = info["w_lum"]
    W_ch  = info["W_chroma"]
    med_p = info["median_pred"]
    med_g = info["median_gt"]

    pred64   = np.clip(pred.astype(np.float64), 0.0, None)
    xyz_full = _rgb_to_xyz(pred64)
    Yf, uf, vf = _xyz_to_Yuv(xyz_full)

    Pf      = _pq_encode(Yf / med_p * pq_median_scale)
    Pf_corr = np.clip(np.polyval(w_lum, Pf), 0.0, 1.0)
    Yf_corr = _pq_decode(Pf_corr) / pq_median_scale * med_g

    uf_flat, vf_flat = uf.reshape(-1), vf.reshape(-1)
    uv_corr  = _chroma_basis(uf_flat, vf_flat) @ W_ch
    uf_corr  = uv_corr[:, 0].reshape(uf.shape)
    vf_corr  = uv_corr[:, 1].reshape(vf.shape)

    xyz_corr = _Yuv_to_xyz(Yf_corr, uf_corr, vf_corr)
    rgb_corr = np.clip(_xyz_to_rgb(xyz_corr), 0.0, None)
    return rgb_corr.astype(pred.dtype, copy=False)



# ─────────────────────────────────────────────────────────────────────────────
# v2 PIPELINE PRIMITIVES (matches the notebook's v2 setup).
# Appended for production-script use. Existing v1 functions above are untouched.
# ─────────────────────────────────────────────────────────────────────────────

import re as _re
import cv2 as _cv2_v2  # alias to avoid clobbering any earlier `cv2` import

_W_BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _luma_bt709(img):
    """BT.709 luminance from an (H, W, 3) RGB image."""
    return img.astype(np.float64) @ _W_BT709


# def align_v2(pred, gt, percentile_low=20, percentile_high=80,
#              clip_percentile=99.9, ab=None):
#     """v2 alignment: same affine fit as align_hdr_pred_to_gt but the upper
#     clip is np.percentile(gt_lum, clip_percentile) per frame instead of a
#     hard 1000. Caller passes native relative-linear gt (no pre_hdr_p3)."""
#     pred = np.clip(pred, 0.0, None)


#     #map gt to 1000 max NITS
#     ###########################################################
#     anchor_target_nits = 1000
#     Yg, ug, vg = _xyz_to_Yuv(_rgb_to_xyz(gt))
#     ref_g = float(np.percentile(Yg, clip_percentile)) if Yg.size else 1.0
#     output_yg = np.clip(Yg / ref_g * anchor_target_nits, 0.0, anchor_target_nits)
#     output_ug = ug
#     output_vg = vg
#     output_g = _rgb_to_xyz(_Yuv_to_xyz(output_yg, output_ug, output_vg))
#     gt = output_g
#     ###########################################################


    

#     gt_lum = _luma_bt709(gt)
#     upper  = max(float(np.percentile(gt_lum, clip_percentile)), 1e-3)

#     if ab is not None:
#         a, b = float(ab[0]), float(ab[1])
#         aligned_pred = np.clip(pred * a + b, 0.0, upper)
#         return aligned_pred, gt, (a, b)

#     p_lo, p_hi = np.percentile(gt_lum, [percentile_low, percentile_high])
#     mask = (gt_lum >= p_lo) & (gt_lum <= p_hi)

#     x = pred[mask].reshape(-1)
#     y = gt[mask].reshape(-1)

#     A = np.stack([x, np.ones_like(x)], axis=1)
#     (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

#     aligned_pred = np.clip(pred * a + b, 0.0, upper)
#     return aligned_pred, gt, (a, b)



def align_v2(pred, gt, percentile_low=20, percentile_high=80,
             clip_percentile=99.9, ab=None):
    """v2 alignment: same affine fit as align_hdr_pred_to_gt but the upper
    clip is np.percentile(gt_lum, clip_percentile) per frame instead of a
    hard 1000. Caller passes native relative-linear gt (no pre_hdr_p3)."""
    pred = np.clip(pred, 0.0, None)


    
    eps = 1e-8
    flat = gt.reshape(-1)

    # 0.1 and 99.9 percentiles
    p_low = np.percentile(flat, 0.1)
    p_high = np.percentile(flat, 99.9)

    c = 1.0
    d = 1000.0

    # linear mapping
    out = (gt - p_low) * (d - c) / (p_high - p_low + eps) + c

    # clamp negative values
    gt = np.clip(out, c, d)


    

    gt_lum = _luma_bt709(gt)
    upper  = max(float(np.percentile(gt_lum, clip_percentile)), 1e-3)

    if ab is not None:
        a, b = float(ab[0]), float(ab[1])
        aligned_pred = np.clip(pred * a + b, 0.0, upper)
        return aligned_pred, gt, (a, b)

    p_lo, p_hi = np.percentile(gt_lum, [percentile_low, percentile_high])
    mask = (gt_lum >= p_lo) & (gt_lum <= p_hi)

    x = pred[mask].reshape(-1)
    y = gt[mask].reshape(-1)

    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

    aligned_pred = np.clip(pred * a + b, 0.0, upper)
    return aligned_pred, gt, (a, b)



# def whatever():
#     eps = 1e-8
#     flat = img.reshape(-1)

#     # 0.1 and 99.9 percentiles
#     p_low = np.percentile(flat, 0.1)
#     p_high = np.percentile(flat, 99.9)

#     c = 1.0
#     d = 1000.0

#     # linear mapping
#     out = (img - p_low) * (d - c) / (p_high - p_low + eps) + c

#     # clamp negative values
#     out = np.clip(out, c, d)

#     return out

def fit_and_apply_crf_hanji_v2(
    pred, gt,
    down_factor=4,
    lum_deg=3,
    chroma_lam_scale=0.01,
    anchor_percentile=99.99,
    anchor_target_nits=1000.0,
):
    """v2 Hanji CRF: percentile-PQ normalization (anchor on percentile of
    luminance, not median). Otherwise identical to fit_and_apply_crf_hanji.
    Returns (corrected_pred, info) where info captures the fit so it can be
    reapplied via apply_crf_hanji_v2_fitted on subsequent frames.
    """
    assert pred.shape == gt.shape and pred.shape[-1] == 3

    pred64 = np.clip(pred.astype(np.float64), 0.0, None)
    gt64   = np.clip(gt.astype(np.float64),   0.0, None)

    pred_fit = downsample_inter_area(pred64, down_factor)
    gt_fit   = downsample_inter_area(gt64,   down_factor)

    Yp, up, vp = _xyz_to_Yuv(_rgb_to_xyz(pred_fit))
    Yg, ug, vg = _xyz_to_Yuv(_rgb_to_xyz(gt_fit))

    Yp_pos = Yp[Yp > 0]
    Yg_pos = Yg[Yg > 0]
    ref_p = float(np.percentile(Yp_pos, anchor_percentile)) if Yp_pos.size else 1.0
    ref_g = float(np.percentile(Yg_pos, anchor_percentile)) if Yg_pos.size else 1.0
    ref_p = max(ref_p, 1e-12)
    ref_g = max(ref_g, 1e-12)

    # Pp = _pq_encode(Yp / ref_p * anchor_target_nits)
    # Pg = _pq_encode(Yg / ref_g * anchor_target_nits)

    Pp = _pq_encode(Yp)
    Pg = _pq_encode(Yg)


    valid = (
        np.isfinite(Pp) & np.isfinite(Pg) &
        np.isfinite(up) & np.isfinite(vp) &
        np.isfinite(ug) & np.isfinite(vg) &
        (Yp > 0) & (Yg > 0)
    )
    Pp_v, Pg_v = Pp[valid], Pg[valid]
    up_valid, vp_valid = up[valid], vp[valid]
    ug_valid, vg_valid = ug[valid], vg[valid]

    V_lum = np.vander(Pp_v, lum_deg + 1)
    w_lum, *_ = np.linalg.lstsq(V_lum, Pg_v, rcond=None)

    X_ch = _chroma_basis(up_valid, vp_valid)
    Y_ch = np.stack([ug_valid, vg_valid], axis=1)
    N, K = X_ch.shape
    lam = chroma_lam_scale * N / K
    W0 = np.zeros((K, 2), dtype=np.float64)
    W0[5, 0] = 1.0
    W0[6, 1] = 1.0
    A = X_ch.T @ X_ch + lam * np.eye(K)
    B = X_ch.T @ Y_ch + lam * W0
    W_ch = np.linalg.solve(A, B)



    info = {
        "w_lum": w_lum, "W_chroma": W_ch,
        "ref_pred": ref_p, 
        "anchor_target_nits": anchor_target_nits,
    }

    rgb_corr = apply_crf_hanji_v2_fitted(pred, info)
    return rgb_corr, info


def apply_crf_hanji_v2_fitted(pred, info):
    """Apply a pre-fit v2 Hanji CRF (info dict from fit_and_apply_crf_hanji_v2)
    to a new frame without re-fitting. Used for first-frame-fit pipeline mode."""
    w_lum = info["w_lum"]
    W_ch  = info["W_chroma"]
    ref_p = info["ref_pred"]
    target = info["anchor_target_nits"]

    pred64 = np.clip(pred.astype(np.float64), 0.0, None)
    xyz_full = _rgb_to_xyz(pred64)
    Yf, uf, vf = _xyz_to_Yuv(xyz_full)

    # Pf = _pq_encode(Yf / ref_p * target)
    Pf = _pq_encode(Yf)
    Pf_corr = np.clip(np.polyval(w_lum, Pf), 0.0, 1.0)
    Yf_corr = _pq_decode(Pf_corr) 
    Yf_corr = np.clip(Yf_corr, 0.0, target) #clip luminance to target

    #
    #Yf_corr = _pq_decode(Pf_corr) / target * target / ref_g#should this be target / ref_g?
    #then we should use the sacle in teh other appropriate place...

    uf_flat, vf_flat = uf.reshape(-1), vf.reshape(-1)
    uv_corr = _chroma_basis(uf_flat, vf_flat) @ W_ch
    uf_corr = uv_corr[:, 0].reshape(uf.shape)
    vf_corr = uv_corr[:, 1].reshape(vf.shape)


    xyz_corr = _Yuv_to_xyz(Yf_corr, uf_corr, vf_corr)
    rgb_corr = np.clip(_xyz_to_rgb(xyz_corr), 0.0, None)
    return rgb_corr.astype(pred.dtype, copy=False)


# def pipeline_v2(raw_pred, raw_gt):
#     """v2 end-to-end: native gt (no pre_hdr_p3), align_v2, then Hanji v2.
#     Returns (pred_aligned_only, pred_corrected_with_crf, gt_aligned)."""
#     pred_al, gt_al, _ = align_v2(raw_pred, raw_gt)
#     pred_corr, _ = fit_and_apply_crf_hanji_v2(pred_al, gt_al)
#     return pred_al, pred_corr, gt_al


# ─────────────────────────────────────────────────────────────────────────────
# Tonemap operators for tonemap-domain metrics (Drago-FID, Mantiuk-FID, etc.)
# Mirrors the notebook's _tm_drago and the Mantiuk operator from the
# tonemapper-comparison cell.
# ─────────────────────────────────────────────────────────────────────────────

def _tm_drago(img, gamma=2.2, bias=0.85, saturation=1.0):
    """Drago tonemap (γ=2.2, bias=0.85 by default) → float32 [0,1] RGB image."""
    x = np.clip(img, 0, None).astype(np.float32)
    x = np.maximum(x, 1e-8)  # prevent log(0) inside Drago
    out = _cv2_v2.createTonemapDrago(
        gamma=gamma, saturation=saturation, bias=bias
    ).process(x[..., ::-1])
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out[..., ::-1], 0.0, 1.0).astype(np.float32)


def _tm_mantiuk(img, gamma=2.2, scale=0.7, saturation=1.0):
    """Mantiuk tonemap → float32 [0,1] RGB image. Defaults match the notebook
    visualization cell (γ=2.2, scale=0.7, saturation=1.0)."""
    x = np.clip(img, 0, None).astype(np.float32)
    out = _cv2_v2.createTonemapMantiuk(
        gamma=gamma, scale=scale, saturation=saturation
    ).process(x[..., ::-1])
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out[..., ::-1], 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Frame-name helpers (suffix handling, robust to e.g. frame_0009_s.exr)
# ─────────────────────────────────────────────────────────────────────────────

_FRAME_RE = _re.compile(r"frame_(\d+)")


def gt_name_from_pred(pred_name: str) -> str:
    """Map a pred frame filename (which may have suffixes like '_s' or
    '_anything', and may use a different zero-pad width than GT) to the
    corresponding GT filename 'frame_NNNN.exr' with 4-digit padding.
    Falls back to the input name if no 'frame_<digits>' pattern is present."""
    m = _FRAME_RE.match(pred_name)
    if m:
        return f"frame_{int(m.group(1)):04d}.exr"
    return pred_name

# ═════════════════════════════════════════════════════════════════════════════
# TEMPORAL CONSISTENCY METRICS (appended)
# ─────────────────────────────────────────────────────────────────────────────
# Implements five flickering / temporal-consistency metrics for HDR video:
#
#   • E_warp   (Lai et al., ECCV 2018)   — flow warping error (ref + blind)
#   • tOF      (Chu et al., ACM TOG 2020)— temporal optical-flow consistency (ref)
#   • tLP      (Chu et al., ACM TOG 2020)— temporal LPIPS consistency       (ref)
#   • HDR-TAE  (Mono4DGS-HDR, 2025)      — HDR temporal alignment error     (ref)
#   • VBench-TF (Huang et al., CVPR 2024)— temporal flickering MAE          (blind)
#
# Domain conventions (see accompanying IMPLEMENTATION_DETAILS.md):
#   - E_warp / tOF / tLP / VBench-TF are computed in the µ-law tone-mapped
#     domain (µ=5000), because the optical-flow network (RAFT) and the VGG/AlexNet
#     LPIPS backbone were trained on 8-bit display-referred imagery and are not
#     meaningful on unbounded linear-HDR values. This matches the evaluation
#     protocol used by HDR video reconstruction papers (DeepHDRVideo / LAN-HDR /
#     HDRFlow / HDR-V-Diff), which report metrics in the µ-law domain.
#   - HDR-TAE follows its own paper: optical flow is estimated on a tone-mapped
#     copy of the HDR frames, but the error itself is a scale-invariant ratio
#     computed directly on the (aligned) HDR luminance.
#
# Reference (GT-aware) metrics first align the prediction to GT scale with
# align_v2 (the same affine pred→GT alignment used elsewhere in this repo) so
# that pred and GT live on a common luminance scale before tone mapping. Blind
# metrics (VBench-TF, and the blind E_warp variant) use only the prediction.
# ═════════════════════════════════════════════════════════════════════════════

import torch.nn.functional as _F


# ─────────────────────────────────────────────────────────────────────────────
# µ-law tone mapping (Kalantari & Ramamoorthi 2017; µ=5000)
# ─────────────────────────────────────────────────────────────────────────────
def mu_law_tonemap(img, mu: float = 5000.0, norm: float = None):
    """Tone-map a linear-HDR RGB frame to [0,1] via µ-law.

        T(x) = log(1 + µ * x_norm) / log(1 + µ)

    `norm` is the SEQUENCE-LEVEL normalization constant (NOT a per-frame
    percentile): the frame is divided by `norm` before applying the µ-law
    curve. This must be the same value for every frame of a given video (both
    pred and GT) — see `sequence_norm_constant()` below. Using a per-frame
    normalizer would make the normalization itself a function of frame
    content, which injects spurious inter-frame variation into a temporal-
    consistency measurement (exactly the artifact these metrics exist to
    detect). If `norm` is not given, falls back to this frame's own
    99.9th-percentile (NOT recommended for temporal metrics — only provided
    for standalone/single-frame use). Returns float32 in [0,1].
    """
    x = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
    if norm is None:
        norm = float(np.percentile(x, 99.9))
    if norm <= 0:
        norm = float(x.max()) if x.max() > 0 else 1.0
    x = np.clip(x / norm, 0.0, 1.0)
    out = np.log1p(mu * x) / np.log1p(mu)
    return out.astype(np.float32)


def pu_tonemap(img, norm: float = None):
    """Tone-map a linear-HDR RGB frame to [0,1] via PU21 encoding.

    This is the PU-domain analogue of `mu_law_tonemap`, matching the PU21
    convention already used by `pu()` / PU-PSNR / PU-LPIPS / PU-VSI elsewhere
    in this repo (Mantiuk & Azimi, PU21, PCS 2021) — the same encoding the
    paper's spatial-fidelity metrics use.

    `norm` is the SEQUENCE-LEVEL normalization constant: the PU21-encoded
    frame (raw range roughly [0, ~600]) is divided by `norm` to map into
    [0,1]. As with `mu_law_tonemap`, `norm` must be fixed across the whole
    video (both pred and GT) — see `sequence_norm_constant()`. If `norm` is
    not given, falls back to this frame's own PU21-encoded max (NOT
    recommended for temporal metrics).

    Returns float32 in [0,1].
    """
    v = pu(np.clip(np.asarray(img, dtype=np.float64), 0.0, None))
    if norm is None:
        norm = float(v.max())
    if norm <= 0:
        norm = 1.0
    return np.clip(v / norm, 0.0, 1.0).astype(np.float32)


def sequence_norm_constant(frames_hdr, mode: str = "mu_law", percentile: float = 99.9):
    """Compute a SINGLE normalization constant shared by every frame of a video
    (and by both pred and GT, when called once on the GT sequence).

    mode='mu_law' : returns the `percentile`-th percentile of linear-HDR
                    luminance pooled across all frames (denominator for
                    `mu_law_tonemap`'s x_norm = x / norm).
    mode='pu'     : returns the max PU21-encoded value pooled across all
                    frames (denominator for `pu_tonemap`'s v / norm), matching
                    the `denom = max(pu_gt)` convention used elsewhere in this
                    repo, just pooled over the sequence instead of per-frame.

    Computing this ONCE from the GT sequence and reusing it for both GT and
    pred (and for every frame) is what makes the tone-mapped domain itself
    temporally stable — a prerequisite for any flicker metric computed in
    that domain to be measuring the model's flicker rather than the curve's.
    """
    if mode == "mu_law":
        pooled = np.concatenate([np.clip(f, 0.0, None).reshape(-1) for f in frames_hdr])
        norm = float(np.percentile(pooled, percentile))
        return norm if norm > 0 else 1.0
    elif mode == "pu":
        vmax = 0.0
        for f in frames_hdr:
            v = pu(np.clip(f, 0.0, None))
            vmax = max(vmax, float(v.max()))
        return vmax if vmax > 0 else 1.0
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'mu_law' or 'pu')")


# ─────────────────────────────────────────────────────────────────────────────
# Optical flow (RAFT, torchvision) — lazy singleton
# ─────────────────────────────────────────────────────────────────────────────
_raft_model = None
_raft_device = None


def _get_raft(device="cuda"):
    """Lazily build a RAFT-Large model (torchvision, C_T_SKHT_V2 weights).

    RAFT is used by E_warp, tOF, and HDR-TAE. Trained on Sintel/KITTI (SDR),
    so it must be fed tone-mapped [0,1] frames, not linear HDR.
    """
    global _raft_model, _raft_device
    if _raft_model is not None and _raft_device == device:
        return _raft_model
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).to(device).eval()
    _raft_model = model
    _raft_device = device
    return model


def _to_chw_tensor(img01, device):
    """(H,W,3) float [0,1] ndarray -> (1,3,H,W) torch tensor on device."""
    t = torch.from_numpy(np.ascontiguousarray(img01)).permute(2, 0, 1).unsqueeze(0)
    return t.to(device).float()


def _pad_to_multiple(t, mult=8):
    """Pad a (1,3,H,W) tensor so H,W are multiples of `mult` (RAFT requires /8).
    Returns (padded, (H, W)) so the flow can be cropped back."""
    _, _, h, w = t.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        t = _F.pad(t, (0, pw, 0, ph), mode="replicate")
    return t, (h, w)


@torch.no_grad()
def compute_flow(img1_01, img2_01, device="cuda"):
    """Dense optical flow from img1 -> img2 (both (H,W,3) float [0,1]).

    Returns flow as an (H, W, 2) float32 ndarray (dx, dy) in pixels.
    RAFT expects inputs normalised to [-1, 1].
    """
    model = _get_raft(device)
    t1 = _to_chw_tensor(img1_01, device) * 2.0 - 1.0
    t2 = _to_chw_tensor(img2_01, device) * 2.0 - 1.0
    t1, (h, w) = _pad_to_multiple(t1)
    t2, _ = _pad_to_multiple(t2)
    flow = model(t1, t2)[-1]              # (1,2,H,W), last refinement iter
    flow = flow[:, :, :h, :w]             # crop padding
    return flow[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _flow_to_grid(flow, device):
    """Build a normalised sampling grid for F.grid_sample from an (H,W,2) flow.

    Backward warping: output(x) = input(x + flow(x)). Returns a (1,H,W,2) grid.
    """
    h, w = flow.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    # normalise to [-1, 1]
    gx = 2.0 * map_x / max(w - 1, 1) - 1.0
    gy = 2.0 * map_y / max(h - 1, 1) - 1.0
    grid = np.stack([gx, gy], axis=-1)[None]  # (1,H,W,2)
    return torch.from_numpy(grid.astype(np.float32)).to(device)


@torch.no_grad()
def warp_image(img01, flow, device="cuda"):
    """Backward-warp (H,W,3) img01 by (H,W,2) flow. Returns (H,W,3) float32."""
    t = _to_chw_tensor(img01, device)
    grid = _flow_to_grid(flow, device)
    warped = _F.grid_sample(t, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return warped[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _occlusion_mask(flow_fw, flow_bw, device="cuda", thresh=1.5):
    """Forward/backward consistency occlusion mask (1 = non-occluded / valid).

    A pixel is valid if |flow_fw(x) + flow_bw(x + flow_fw(x))| is small, i.e.
    flow is round-trip consistent. Returns (H,W) float32 in {0,1}.
    """
    h, w = flow_fw.shape[:2]
    # warp backward flow into frame-1 coordinates using forward flow
    grid = _flow_to_grid(flow_fw, device)
    bw_t = torch.from_numpy(flow_bw.transpose(2, 0, 1)[None]).to(device).float()
    bw_warped = _F.grid_sample(bw_t, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
    bw_warped = bw_warped[0].permute(1, 2, 0).cpu().numpy()
    diff = flow_fw + bw_warped                      # should be ~0 if consistent
    mag = np.sqrt((diff ** 2).sum(axis=-1))
    fw_mag = np.sqrt((flow_fw ** 2).sum(axis=-1))
    tol = 0.01 * fw_mag + thresh
    return (mag <= tol).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# tLP backbone — reuse the existing AlexNet-LPIPS metric from this module
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _lpips_pair(a01, b01, device="cuda"):
    """LPIPS distance between two (H,W,3) [0,1] frames. Reuses lpips_metric
    (AlexNet, normalize=True) already constructed at module load."""
    return float(lpips(a01.astype(np.float32), b01.astype(np.float32), device=device))


# ═════════════════════════════════════════════════════════════════════════════
# METRIC 1 — E_warp  (Lai et al., ECCV 2018)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ewarp_sequence(frames_tm, device="cuda", long_term=True):
    """Flow warping error over a tone-mapped frame sequence.

    frames_tm : list of (H,W,3) float [0,1] tone-mapped frames (the SAME video;
                this is a *self*-consistency metric — flow is computed on the
                sequence itself, so it works both for the blind case (pred only)
                and the reference case (pred aligned to GT then tone-mapped)).

    For each t, warps frame O_{t-1} (and optionally O_1) into frame O_t using
    flow estimated O_t -> O_{t-1}, and accumulates the occlusion-masked L1
    residual. Mirrors Lai et al. Eq. (6)-(7) and the DVP (NeurIPS 2020) protocol.

    Returns the mean warping error (lower = more temporally stable).
    """
    n = len(frames_tm)
    if n < 2:
        return float("nan")

    def _pair_error(O_t, O_s):
        flow_ts = compute_flow(O_t, O_s, device)         # O_t -> O_s
        flow_st = compute_flow(O_s, O_t, device)         # for occlusion check
        mask = _occlusion_mask(flow_ts, flow_st, device)
        warped = warp_image(O_s, flow_ts, device)        # O_s aligned to O_t
        l1 = np.abs(O_t - warped).sum(axis=-1)           # per-pixel L1 over RGB
        denom = mask.sum()
        if denom <= 0:
            return float("nan")
        return float((mask * l1).sum() / denom)

    errs = []
    for t in range(1, n):
        e_short = _pair_error(frames_tm[t], frames_tm[t - 1])
        if long_term:
            e_long = _pair_error(frames_tm[t], frames_tm[0])
            pair = np.nanmean([e_short, e_long])
        else:
            pair = e_short
        if np.isfinite(pair):
            errs.append(pair)
    return float(np.mean(errs)) if errs else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# METRIC 2 — tOF  (Chu et al., ACM TOG 2020 / TecoGAN)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tof_sequence(pred_tm, gt_tm, device="cuda"):
    """Temporal optical-flow consistency between pred and GT sequences.

        tOF_t = || OF(pred_{t-1}, pred_t) - OF(gt_{t-1}, gt_t) ||_1   (per pixel)

    Both sequences are tone-mapped [0,1] frame lists of equal length. Returns
    the mean over t of the mean per-pixel flow end-point L1 difference.
    Lower = motion of the prediction matches the GT motion (less flicker).
    """
    n = min(len(pred_tm), len(gt_tm))
    if n < 2:
        return float("nan")
    vals = []
    for t in range(1, n):
        of_pred = compute_flow(pred_tm[t - 1], pred_tm[t], device)
        of_gt = compute_flow(gt_tm[t - 1], gt_tm[t], device)
        diff = np.abs(of_pred - of_gt).sum(axis=-1)   # L1 over (dx,dy)
        vals.append(float(diff.mean()))
    return float(np.mean(vals)) if vals else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# METRIC 3 — tLP  (Chu et al., ACM TOG 2020 / TecoGAN)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tlp_sequence(pred_tm, gt_tm, device="cuda", scale=100.0):
    """Temporal LPIPS consistency between pred and GT sequences.

        tLP_t = | LPIPS(gt_{t-1}, gt_t) - LPIPS(pred_{t-1}, pred_t) |

    Measures whether the perceptual frame-to-frame change of the prediction
    matches that of the GT. Following TecoGAN, the result is scaled by 100.
    Lower = better. Returns the mean over t.
    """
    n = min(len(pred_tm), len(gt_tm))
    if n < 2:
        return float("nan")
    vals = []
    for t in range(1, n):
        d_pred = _lpips_pair(pred_tm[t - 1], pred_tm[t], device)
        d_gt = _lpips_pair(gt_tm[t - 1], gt_tm[t], device)
        vals.append(abs(d_gt - d_pred) * scale)
    return float(np.mean(vals)) if vals else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# METRIC 4 — HDR-TAE  (Mono4DGS-HDR, 2025)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def hdr_tae_sequence(frames_hdr, frames_tm, device="cuda"):
    """HDR Temporal Alignment Error.

    frames_hdr : list of (H,W,3) linear-HDR (aligned) frames — the error ratio
                 is computed on these so it is scale-invariant across the HDR range.
    frames_tm  : list of (H,W,3) tone-mapped [0,1] frames (SAME video) used ONLY
                 for optical-flow + visibility-mask estimation with RAFT.

    Symmetric, bidirectional, normalised difference (Mono4DGS-HDR Eq. 11):

        HDR-TAE = 1/(2(N-1)) Σ_t
              | V_{t→t+1} ⊙ (W(H_{t+1}, F_{t→t+1}) - H_t) / (W(H_{t+1}) + H_t) |_1
            + | V_{t+1→t} ⊙ (W(H_t,   F_{t+1→t}) - H_{t+1}) / (W(H_t)   + H_{t+1}) |_1

    Lower = better temporal consistency of HDR radiance.
    """
    n = min(len(frames_hdr), len(frames_tm))
    if n < 2:
        return float("nan")
    eps = 1e-6
    total = 0.0
    count = 0
    for t in range(n - 1):
        H_t, H_t1 = frames_hdr[t], frames_hdr[t + 1]
        # flows estimated on tone-mapped copies
        F_t_to_t1 = compute_flow(frames_tm[t], frames_tm[t + 1], device)
        F_t1_to_t = compute_flow(frames_tm[t + 1], frames_tm[t], device)
        V_t_to_t1 = _occlusion_mask(F_t_to_t1, F_t1_to_t, device)[..., None]
        V_t1_to_t = _occlusion_mask(F_t1_to_t, F_t_to_t1, device)[..., None]

        # warp H_{t+1} -> t frame, and H_t -> t+1 frame
        Hwarp_t1_to_t = warp_image(H_t1, F_t_to_t1, device)
        Hwarp_t_to_t1 = warp_image(H_t,  F_t1_to_t, device)

        term_a = np.abs(V_t_to_t1 * (Hwarp_t1_to_t - H_t)
                        / (Hwarp_t1_to_t + H_t + eps)).sum(axis=-1)
        term_b = np.abs(V_t1_to_t * (Hwarp_t_to_t1 - H_t1)
                        / (Hwarp_t_to_t1 + H_t1 + eps)).sum(axis=-1)
        total += float(term_a.mean() + term_b.mean())
        count += 1
    if count == 0:
        return float("nan")
    return total / (2.0 * count)


# ═════════════════════════════════════════════════════════════════════════════
# METRIC 5 — VBench Temporal Flickering  (Huang et al., CVPR 2024) — BLIND
# ═════════════════════════════════════════════════════════════════════════════
def vbench_tf_sequence(frames_tm):
    """VBench Temporal Flickering (mean absolute difference across frames).

    frames_tm : list of (H,W,3) tone-mapped frames in [0,1] (prediction only;
                this is a no-reference metric).

        S_flicker = mean_t mean_pixels |f_t - f_{t+1}|   (computed in [0,255])
        score     = (255 - S_flicker) / 255             (higher = less flicker)

    Returns the normalised score in [0,1] (higher = better), matching VBench's
    sign convention so it can sit alongside the other "higher = better" metrics.
    """
    n = len(frames_tm)
    if n < 2:
        return float("nan")
    maes = []
    for t in range(n - 1):
        a = np.clip(frames_tm[t], 0.0, 1.0) * 255.0
        b = np.clip(frames_tm[t + 1], 0.0, 1.0) * 255.0
        maes.append(float(np.abs(a - b).mean()))
    s_flicker = float(np.mean(maes))
    return (255.0 - s_flicker) / 255.0
