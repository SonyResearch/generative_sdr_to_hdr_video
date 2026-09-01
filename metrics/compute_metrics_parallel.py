# CONDA ENVIRONMENT P312
"""
Per-(method, dataset, type) metric computation for HDR video evaluation.

This script:
  • Uses the v2 pipeline (no pre_hdr_p3, percentile-PQ Hanji CRF) for all
    full-reference metrics that need it.
  • Computes NR metrics (PIQE) on aligned-only pred (no CRF).
  • Computes tonemap-domain FID/FVD (Drago, Mantiuk) on aligned-only pred.
  • PU-FID/PU-FVD are computed on aligned-only pred encoded with PU21.
  • PU-TF (PU21-domain VBench Temporal Flickering, Huang et al. CVPR 2024) is a
    per-video, no-reference temporal-consistency score computed on the RAW
    (unaligned, no-CRF) pred sequence tone-mapped with PU21 using a single
    sequence-level normalization constant. See the PU-TF section below for why
    its pipeline differs from the full-reference metrics.

Metrics are toggled via the `METRICS_ENABLED` dict below — comment out what
you don't want for a given run.

Frame-name handling supports suffixes like 'frame_0009_s.exr' or
'frame_0009_anything.exr' (extracts the frame_NNNN prefix to find the
corresponding GT file).

Empty video directories are skipped with a warning instead of raising.
"""

import argparse
import os
import csv
import gc
import threading
from pathlib import Path
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from metrics_helpers import (
    # File I/O
    read_image,
    # v1 helpers (kept available for legacy code paths if needed)
    pre_hdr_p3, align_hdr_pred_to_gt,
    # Per-frame metrics
    psnr, vsi, piqe, lpips, hdr_vdp3, pu, reinhard_tonemap,
    # FID/FVD
    initialize_fid, initialize_fvd,
    compute_fid, fid_update, compute_fvd, fvd_update,
    # CVVDP
    cvvdp, initialize_cvvdp,
    # v2 pipeline (added in append)
    align_v2, fit_and_apply_crf_hanji_v2, apply_crf_hanji_v2_fitted,
    # Tonemap operators
    _tm_drago, _tm_mantiuk,
    # Frame-name helper
    gt_name_from_pred,
    # PU-TF (PU21-domain VBench Temporal Flickering)
    pu_tonemap, sequence_norm_constant, vbench_tf_sequence,
)

# ----------------------------
# Config
# ----------------------------
EVAL_BASE        = "/datasets/gencamedit/hdreval/evaluations_transfer"
# EVAL_BASE        = "/projects/gencamedit/hdreval/evaluations"
EVAL_OUTPUT_DIR  = "/datasets/sai/sdr2hdrvid/metrics/eval_output"
DATASETS         = ("stuttgart", "ubc")
METHODS = ( "oursapr21", "eilertsen_pytorch_crf", "ltx", "lediff", "x2hdr", "single_crf", "dvitmo_crf",  "zhdrv_crf", "hdrtv_crf", "ablatedebevec", "ablatel2", "ablaterope", "oursmay5", "release")



# # LTM metrics computing (commented out as this was a special config) 
# EVAL_BASE        = "/datasets/gencamedit/hdreval/evaluations_transfer"
# # EVAL_BASE        = "/projects/gencamedit/hdreval/evaluations"
# EVAL_OUTPUT_DIR  = "/projects/gencamedit/hdreval/evaluations_output_ltm"
# DATASETS         = ("stuttgart", "ubc")
# METHODS = ("single_crf", "dvitmo_crf")

SCRIPT_NAME = "compute_metrics_parallel.py"


# ----------------------------
# Metrics enabled — comment out what you DON'T want for this run.
# Keys here drive which accumulators are created and which CSV columns
# are written. Required:  CVVDP, PU-PSNR, PU-PIQE, Drago-FID, Mantiuk-FID.
# Other available metrics (uncomment to enable):
# ----------------------------
METRICS_ENABLED = {
    # ── Core (required for the current run) ────────────────────────────────
    "CVVDP":      True,
    "PU-PSNR":    True,
    "PU-PIQE":    True,
    "Drago-FID":  True,
    "Mantiuk-FID": True,
    # ── Optional per-frame metrics (uncomment to enable) ───────────────────
    "PU-VSI":     True,
    "PU-LPIPS":   True,
    "HDR-VDP3":   True,
    # ── Temporal consistency (per-video, no-reference) ────────────────────
    "PU-TF":      True,
    # ── Optional distributional metrics ───────────────────────────────────
    "PU-FID":     True,
    "Drago-FVD":  True,
    "Mantiuk-FVD": True,
    "PU-FVD":     True,
    # ── Legacy (Reinhard) — keep available if you want to compare ─────────
    "Reinhard-FID": False,
    "Reinhard-FVD": False,
    "Reinhard-FVD-FIRST": False,
    "PU-FVD-FIRST": False,
}

# v2 pipeline anchors (mirroring the notebook setup)
V2_ANCHOR_PERCENTILE  = 99.0
V2_ANCHOR_TARGET_NITS = 1000.0

# ----------------------------
# PU-TF — PU21-domain VBench Temporal Flickering (Huang et al., CVPR 2024)
# ----------------------------
# Unlike every other metric in this script, PU-TF is a *sequence*-level,
# no-reference measurement, so its input pipeline deliberately differs:
#
#   1. NO alignment and NO Hanji CRF. Both are fit per-frame against GT, so
#      applying them would inject inter-frame variation of their own into a
#      measurement whose whole purpose is to detect inter-frame variation.
#      PU-TF therefore runs on the raw pred frames (clipped at 0), never on
#      pred_al / pref_crfal.
#   2. SEQUENCE-LEVEL PU21 normalization, not per-frame. The constant is the
#      max PU21-encoded value pooled over the whole pred sequence
#      (`sequence_norm_constant(..., mode="pu")`) and is reused for every frame.
#      Normalizing each frame by its own max would make the tone curve a
#      function of per-frame content — which either fabricates flicker in a
#      stable sequence or hides real brightness drift.
#
# Score convention (VBench): S_flicker = mean_t mean_pixels |f_t - f_{t+1}| in
# [0,255]; the reported value is (255 - S_flicker)/255, so HIGHER = less
# flicker = better, matching the other "higher is better" columns.
PU_TF_NORM_MODE = "pu"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute HDR video metrics")
    parser.add_argument("dataset", type=str, choices=DATASETS,
                        help="Dataset: stuttgart or ubc")
    parser.add_argument("method", type=str, choices=list(METHODS),
                        help="Method name")
    parser.add_argument("type", type=str,
                        help="Subfolder under method (e.g. auto, over, under)")
    parser.add_argument("--num-files", type=int, default=17, metavar="N",
                        help="Max frames per video (default 17)")
    parser.add_argument("--ds", type=int, default=1, metavar="K",
                        help="Use every K-th frame from each video dir (default 1)")
    parser.add_argument("--videos", type=str, default=None, metavar="V1,V2,...",
                        help="Optional comma-separated video folder names to run")
    parser.add_argument("--videos-exclude", type=str, default=None, metavar="V1,V2,...",
                        help="Optional comma-separated video folder names to exclude")
    return parser.parse_args()


def _find_eval_dir(base: str, method: str, dataset: str) -> str:
    from pathlib import Path as _Path
    for name in (f"{method}_{dataset}", f"{dataset}_{method}"):
        p = _Path(base) / name
        if p.is_dir():
            return str(p)
    raise FileNotFoundError(
        f"No evaluation directory found for method='{method}' dataset='{dataset}' "
        f"(tried '{method}_{dataset}' and '{dataset}_{method}' under {base})"
    )


args = parse_args()
gt_dir   = os.path.join(EVAL_BASE, args.dataset, "hdr")
if args.method == "release":
    # Predictions from this repo's own test.py, kept local to the release
    # checkout rather than requiring a symlink into the shared EVAL_BASE tree.
    pred_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluations", f"test_output_{args.dataset}", args.type,
    )
else:
    pred_dir = os.path.join(_find_eval_dir(EVAL_BASE, args.method, args.dataset), args.type)

print(f"GT Directory:   {gt_dir}")
print(f"Pred Directory: {pred_dir}")

NUM_FILES = args.num_files
DS = max(1, args.ds)

# Discover videos
video_paths = sorted([d for d in os.listdir(pred_dir)
                      if os.path.isdir(os.path.join(pred_dir, d))])
# Some methods (e.g. ltx) emit a duplicate '<name>_hdr' pred dir alongside '<name>'.
# Drop only that duplicate-pair variant — preserve any '_hdr' name without a sibling
# so genuinely missing GT still surfaces as an error.
_pred_set = set(video_paths)
_dropped_hdr = [v for v in video_paths
                if v.endswith("_hdr") and v[:-len("_hdr")] in _pred_set]
if _dropped_hdr:
    print(f"Skipping {len(_dropped_hdr)} '_hdr' duplicate pred dir(s): {_dropped_hdr}")
    _dropped_set = set(_dropped_hdr)
    video_paths = [v for v in video_paths if v not in _dropped_set]
if args.videos:
    requested = {v.strip() for v in args.videos.split(",") if v.strip()}
    video_paths = [v for v in video_paths if v in requested]
    if not video_paths:
        raise ValueError(f"No matching videos for --videos={args.videos} under {pred_dir}")
    print(f"Running limited video set ({len(video_paths)}): {video_paths}")
if args.videos_exclude:
    excluded = {v.strip() for v in args.videos_exclude.split(",") if v.strip()}
    video_paths = [v for v in video_paths if v not in excluded]
    if not video_paths:
        raise ValueError(f"No videos left after --videos-exclude={args.videos_exclude}")
    print(f"Excluding videos: {sorted(excluded)}")


def _int_env(name: str, default: int, min_val: int = 1, max_val: int = 32) -> int:
    try:
        v = int(os.environ.get(name, default))
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return default


def get_cuda_free_mb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if hasattr(torch.cuda, "mem_get_info"):
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return (total - reserved) / (1024 ** 2)
    except Exception:
        return None


max_workers = _int_env("METRICS_MAX_WORKERS", 1)
MAX_IN_FLIGHT = _int_env("METRICS_MAX_IN_FLIGHT", 2)
SAFETY_MB = _int_env("METRICS_SAFETY_MB", 1500, min_val=256, max_val=16000)

_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
print(f"CUDA_VISIBLE_DEVICES={_visible} | METRICS_MAX_WORKERS={max_workers} "
      f"MAX_IN_FLIGHT={MAX_IN_FLIGHT}")
free_mb = get_cuda_free_mb()
if free_mb is not None:
    print(f"GPU free memory at start: {free_mb:.0f} MB; backpressure threshold: {SAFETY_MB} MB")


# ----------------------------
# Initialize metrics & accumulators (only those enabled)
# ----------------------------
def _enabled(name: str) -> bool:
    return bool(METRICS_ENABLED.get(name, False))


cvvdp_metric = initialize_cvvdp() if _enabled("CVVDP") else None

# FID/FVD accumulators (one per enabled distributional metric)
fid_objs = {}
fvd_objs = {}
if _enabled("PU-FID"):       fid_objs["PU-FID"]       = initialize_fid()
if _enabled("Drago-FID"):    fid_objs["Drago-FID"]    = initialize_fid()
if _enabled("Mantiuk-FID"):  fid_objs["Mantiuk-FID"]  = initialize_fid()
if _enabled("Reinhard-FID"): fid_objs["Reinhard-FID"] = initialize_fid()

if _enabled("PU-FVD"):       fvd_objs["PU-FVD"]       = initialize_fvd()
if _enabled("Drago-FVD"):    fvd_objs["Drago-FVD"]    = initialize_fvd()
if _enabled("Mantiuk-FVD"):  fvd_objs["Mantiuk-FVD"]  = initialize_fvd()
if _enabled("Reinhard-FVD"): fvd_objs["Reinhard-FVD"] = initialize_fvd()
if _enabled("PU-FVD-FIRST"):       fvd_objs["PU-FVD-FIRST"]       = initialize_fvd()
if _enabled("Reinhard-FVD-FIRST"): fvd_objs["Reinhard-FVD-FIRST"] = initialize_fvd()

# Per-frame scalar accumulators (across all videos for this run)
psnr_scores    = []
piqe_scores    = []
vsi_scores     = []
lpips_scores   = []
hdrvdp3_scores = []
cvvdp_scores   = []   # one entry per video (CVVDP runs on the whole sequence)
putf_scores    = []   # one entry per video (PU-TF runs on the whole sequence)

lpips_lock = threading.Lock()


# ----------------------------
# Per-frame computation
# ----------------------------
def process_one_frame(idx, pred_im_path, gt_im_path, ab_first, hanji_info_first):
    """Run all per-frame computations.

    Returns dict with everything needed by the main thread to:
      - update FVD/FID accumulators in domain-specific order
      - append per-frame scalar metrics
      - run CVVDP after all frames are gathered (uses pred_corr / gt_al)
    """
    raw_pred = read_image(pred_im_path)
    raw_gt   = read_image(gt_im_path)

    # v2 alignment + Hanji-v2 CRF (current frame fit)
    pred_al, gt_al, _ = align_v2(raw_pred, raw_gt)
    gt_crfal = gt_al
    pref_crfal, _ = fit_and_apply_crf_hanji_v2(
        pred_al, gt_al,
        anchor_percentile=V2_ANCHOR_PERCENTILE,
        anchor_target_nits=V2_ANCHOR_TARGET_NITS,
    )

    out = {"idx": idx}

    # ── Full-reference per-frame metrics (use pred_corr, gt_al) ───────────
    # # print align max and percentiles
    # print("Max of gt_al:", np.max(gt_al))
    # print("max of pred_al:", np.max(pred_al))
    #
    # # print percentiles of gt_al and pred_corr
    # print("99 Percentiles of gt_al:", np.percentile(gt_al, [ 99, 99.9]))
    # print("99 Percentiles of pred_al:", np.percentile(pred_al, [99, 99.9]))
    #
    #
    # print("########")
    #
    # # ── Full-reference per-frame metrics (use pred_corr, gt_al) ───────────
    # print("Max of gt_crfal:", np.max(gt_crfal))
    # print("max of pref_crfal:", np.max(pref_crfal))
    #
    # # print percentiles of gt_al and pred_corr
    # print("99 Percentiles of gt_crfal:", np.percentile(gt_crfal, [ 99, 99.9]))
    # print("99 Percentiles of pref_crfal:", np.percentile(pref_crfal, [99, 99.9]))
    #
    #
    # print("################\n\n")

    pu_pred = pu(pref_crfal); pu_gt = pu(gt_crfal)
    denom = float(np.max(pu_gt)) if np.max(pu_gt) > 0 else 1.0
    pu_pred_01 = np.clip(pu_pred / denom, 0.0, 1.0).astype(np.float32)
    pu_gt_01   = np.clip(pu_gt   / denom, 0.0, 1.0).astype(np.float32)

    if _enabled("PU-PSNR"):
        out["pu_psnr"] = float(psnr(pu_pred, pu_gt))
    if _enabled("PU-VSI"):
        out["pu_vsi"]  = float(vsi(pu_pred, pu_gt))
    if _enabled("HDR-VDP3"):
        out["hdrvdp3"] = float(hdr_vdp3(pref_crfal, gt_crfal))
    if _enabled("PU-LPIPS"):
        with lpips_lock:
            out["pu_lpips"] = float(lpips(pu_pred_01, pu_gt_01))

    # ── No-reference / NR metrics on pred_nr (aligned only, no CRF) ───────
    if _enabled("PU-PIQE"):
        out["pu_piqe"] = float(piqe(pu(pred_al)))

    # ── Tonemap-domain frames (for FID/FVD; main thread updates accumulators) ─
    # Drago and Mantiuk are applied on pred_nr (no CRF) and on gt_al, both
    # native relative-linear. Output is float32 [0,1] RGB.
    if _enabled("Drago-FID") or _enabled("Drago-FVD"):
        out["drago_pred"] = _tm_drago(pred_al)
        out["drago_gt"]   = _tm_drago(gt_al)
    if _enabled("Mantiuk-FID") or _enabled("Mantiuk-FVD"):
        out["mantiuk_pred"] = _tm_mantiuk(pred_al)
        out["mantiuk_gt"]   = _tm_mantiuk(gt_al)

    # ── PU-domain frames (for PU-FID/PU-FVD) ──────────────────────────────
    if _enabled("PU-FID") or _enabled("PU-FVD"):
        pu_pred_nr = pu(pred_al)
        pu_gt_nr = pu(gt_al)
        denom_nr = float(np.max(pu_gt_nr)) if np.max(pu_gt_nr) > 0 else 1.0
        out["pu_pred_nr_01"] = np.clip(pu_pred_nr / denom_nr, 0.0, 1.0).astype(np.float32)
        out["pu_gt_01"]      = np.clip(pu_gt_nr / denom_nr, 0.0, 1.0).astype(np.float32)

    # ── Reinhard-domain frames (legacy, only if enabled) ──────────────────
    if _enabled("Reinhard-FID") or _enabled("Reinhard-FVD"):
        out["reinhard_pred"] = reinhard_tonemap(pred_al)
        out["reinhard_gt"]   = reinhard_tonemap(gt_al)

    # # ── First-frame-fit alternates (Reinhard-FVD-FIRST, PU-FVD-FIRST) ─────
    # if _enabled("Reinhard-FVD-FIRST") or _enabled("PU-FVD-FIRST"):
    #     # Apply pre-fit affine and pre-fit CRF to raw_pred
    #     pred_al_f, _, _ = align_v2(raw_pred, raw_gt, ab=ab_first)
    #     pred_corr_f = apply_crf_hanji_v2_fitted(pred_al_f, hanji_info_first)
    #     if _enabled("Reinhard-FVD-FIRST"):
    #         out["reinhard_pred_firstfit"] = reinhard_tonemap(pred_corr_f)
    #     if _enabled("PU-FVD-FIRST"):
    #         pu_pred_firstfit = pu(pred_corr_f)
    #         denom_f = float(np.max(pu_gt)) if np.max(pu_gt) > 0 else 1.0
    #         out["pu_pred_norm_firstfit"] = (pu_pred_firstfit / denom_f).astype(np.float32)

    # ── Hold pred_corr / gt_al for the per-video CVVDP pass ───────────────
    if _enabled("CVVDP"):
        out["pred_corr"] = pref_crfal
        out["gt_al"]     = gt_crfal

    # ── Hold the RAW pred frame for the per-video PU-TF pass ──────────────
    # Deliberately raw_pred, not pred_al / pref_crfal: PU-TF is blind and must
    # not see per-frame alignment or CRF fits (see PU-TF notes at the top).
    if _enabled("PU-TF"):
        out["raw_pred"] = np.clip(raw_pred, 0.0, None).astype(np.float32)

    return out


# ----------------------------
# Main per-video loop
# ----------------------------
for video_path in tqdm(video_paths, desc="Videos", unit="video"):
    pred_video_dir = os.path.join(pred_dir, video_path)
    gt_video_dir   = os.path.join(gt_dir, video_path)

    all_frames = sorted(f for f in os.listdir(pred_video_dir) if f.endswith(".exr"))
    im_paths = all_frames[::DS][:NUM_FILES]
    if len(im_paths) == 0:
        print(f"[WARNING] No frames in {pred_video_dir} (ds={DS}), skipping.")
        continue

    # First-frame fit (used by *-FIRST FVD modes)
    im0 = im_paths[0]
    pred0 = read_image(os.path.join(pred_video_dir, im0))
    gt0   = read_image(os.path.join(gt_video_dir,   gt_name_from_pred(im0)))
    pred0_aligned, gt0_aligned, ab_first = align_v2(pred0, gt0)
    _, hanji_info_first = fit_and_apply_crf_hanji_v2(
        pred0_aligned, gt0_aligned,
        anchor_percentile=V2_ANCHOR_PERCENTILE,
        anchor_target_nits=V2_ANCHOR_TARGET_NITS,
    )

    # Per-video accumulators (for the per-video CSV)
    v_psnr    = []
    v_vsi     = []
    v_piqe    = []
    v_lpips   = []
    v_hdrvdp3 = []

    n = len(im_paths)
    pred_corrs    = [None] * n
    gt_als        = [None] * n
    drago_preds   = [None] * n
    drago_gts     = [None] * n
    mantiuk_preds = [None] * n
    mantiuk_gts   = [None] * n
    pu_preds_nr   = [None] * n
    pu_gts        = [None] * n
    reinhard_preds = [None] * n
    reinhard_gts   = [None] * n
    reinhard_preds_firstfit = [None] * n
    pu_preds_firstfit = [None] * n
    raw_preds     = [None] * n   # PU-TF: raw (unaligned, no-CRF) pred frames

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = enumerate(im_paths)
        futures = {}

        # Initial submission window
        for idx, im_name in islice(it, MAX_IN_FLIGHT):
            pred_im_path = os.path.join(pred_video_dir, im_name)
            gt_im_path   = os.path.join(gt_video_dir,   gt_name_from_pred(im_name))
            fut = ex.submit(process_one_frame, idx, pred_im_path, gt_im_path,
                            ab_first, hanji_info_first)
            futures[fut] = idx

        pbar = tqdm(total=n, desc=f"Frames [{video_path}]", unit="frame", leave=False)
        while futures:
            for fut in as_completed(list(futures.keys())):
                _ = futures.pop(fut)
                out = fut.result()
                idx = out["idx"]

                if "pred_corr" in out:
                    pred_corrs[idx] = out["pred_corr"]
                    gt_als[idx]     = out["gt_al"]
                if "drago_pred" in out:
                    drago_preds[idx] = out["drago_pred"]
                    drago_gts[idx]   = out["drago_gt"]
                if "mantiuk_pred" in out:
                    mantiuk_preds[idx] = out["mantiuk_pred"]
                    mantiuk_gts[idx]   = out["mantiuk_gt"]
                if "pu_pred_nr_01" in out:
                    pu_preds_nr[idx] = out["pu_pred_nr_01"]
                    pu_gts[idx]      = out["pu_gt_01"]
                if "reinhard_pred" in out:
                    reinhard_preds[idx] = out["reinhard_pred"]
                    reinhard_gts[idx]   = out["reinhard_gt"]
                if "reinhard_pred_firstfit" in out:
                    reinhard_preds_firstfit[idx] = out["reinhard_pred_firstfit"]
                if "pu_pred_norm_firstfit" in out:
                    pu_preds_firstfit[idx] = out["pu_pred_norm_firstfit"]
                if "raw_pred" in out:
                    raw_preds[idx] = out["raw_pred"]

                # Append per-frame scalar scores
                if "pu_psnr" in out:
                    psnr_scores.append(out["pu_psnr"]); v_psnr.append(out["pu_psnr"])
                if "pu_vsi" in out:
                    vsi_scores.append(out["pu_vsi"]);   v_vsi.append(out["pu_vsi"])
                if "pu_piqe" in out:
                    piqe_scores.append(out["pu_piqe"]); v_piqe.append(out["pu_piqe"])
                if "pu_lpips" in out:
                    lpips_scores.append(out["pu_lpips"]); v_lpips.append(out["pu_lpips"])
                if "hdrvdp3" in out:
                    hdrvdp3_scores.append(out["hdrvdp3"]); v_hdrvdp3.append(out["hdrvdp3"])

                del out
                pbar.update(1)

                try:
                    idx2, im_name2 = next(it)
                    pred_im_path2 = os.path.join(pred_video_dir, im_name2)
                    gt_im_path2   = os.path.join(gt_video_dir,   gt_name_from_pred(im_name2))
                    fut2 = ex.submit(process_one_frame, idx2, pred_im_path2, gt_im_path2,
                                     ab_first, hanji_info_first)
                    futures[fut2] = idx2
                except StopIteration:
                    pass

                break
        pbar.close()

    # Per-video CVVDP (whole sequence)
    cvvdp_score = float("nan")
    if _enabled("CVVDP"):
        print(f"Computing CVVDP for video: {video_path}")
        cvvdp_score = float(cvvdp(pred_corrs, gt_als, cvvdp_metric))
        cvvdp_scores.append(cvvdp_score)
        print(f"CVVDP Score: {cvvdp_score}")

    # Per-video PU-TF (whole sequence, blind, PU21 domain)
    putf_score = float("nan")
    if _enabled("PU-TF"):
        print(f"Computing PU-TF for video: {video_path}")
        if n < 2:
            print(f"[WARNING] <2 frames in {pred_video_dir}; PU-TF is NaN.")
        else:
            # One normalization constant for the whole sequence, from the same
            # raw pred frames the metric is measured on (blind — no GT).
            putf_norm = sequence_norm_constant(raw_preds, mode=PU_TF_NORM_MODE)
            putf_tm = [pu_tonemap(f, putf_norm) for f in raw_preds]
            putf_score = float(vbench_tf_sequence(putf_tm))
            del putf_tm
        putf_scores.append(putf_score)
        print(f"PU-TF Score: {putf_score}")

    # Per-video CSV
    Path(EVAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_csv_video = (Path(EVAL_OUTPUT_DIR) /
        f"results_{args.method}_{args.dataset}_{args.type}_{video_path}_{NUM_FILES}_ds{DS}.csv")
    video_row = {}
    video_fieldnames = []
    for col, arr in [("PU-PSNR", v_psnr), ("PU-VSI", v_vsi), ("PU-PIQE", v_piqe),
                     ("PU-LPIPS", v_lpips), ("HDR-VDP3", v_hdrvdp3)]:
        if arr:
            video_fieldnames.append(col)
            video_row[col] = float(np.mean(arr))
    if _enabled("CVVDP"):
        video_fieldnames.append("CVVDP"); video_row["CVVDP"] = cvvdp_score
    if _enabled("PU-TF"):
        video_fieldnames.append("PU-TF"); video_row["PU-TF"] = putf_score
    if video_fieldnames:
        with open(out_csv_video, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=video_fieldnames)
            writer.writeheader(); writer.writerow(video_row)
        print(f"Wrote per-video: {out_csv_video}")

    # Update FID/FVD accumulators
    print(f"Updating FID/FVD accumulators for video: {video_path}")
    if _enabled("PU-FID"):       fid_update(pu_preds_nr,   pu_gts,        fid_objs["PU-FID"])
    if _enabled("Drago-FID"):    fid_update(drago_preds,   drago_gts,     fid_objs["Drago-FID"])
    if _enabled("Mantiuk-FID"):  fid_update(mantiuk_preds, mantiuk_gts,   fid_objs["Mantiuk-FID"])
    if _enabled("Reinhard-FID"): fid_update(reinhard_preds, reinhard_gts, fid_objs["Reinhard-FID"])

    if _enabled("PU-FVD"):       fvd_update(pu_preds_nr,   pu_gts,        fvd_objs["PU-FVD"])
    if _enabled("Drago-FVD"):    fvd_update(drago_preds,   drago_gts,     fvd_objs["Drago-FVD"])
    if _enabled("Mantiuk-FVD"):  fvd_update(mantiuk_preds, mantiuk_gts,   fvd_objs["Mantiuk-FVD"])
    if _enabled("Reinhard-FVD"): fvd_update(reinhard_preds, reinhard_gts, fvd_objs["Reinhard-FVD"])
    if _enabled("Reinhard-FVD-FIRST"):
        fvd_update(reinhard_preds_firstfit, reinhard_gts, fvd_objs["Reinhard-FVD-FIRST"])
    if _enabled("PU-FVD-FIRST"):
        fvd_update(pu_preds_firstfit, pu_gts, fvd_objs["PU-FVD-FIRST"])

    del pred_corrs, gt_als, drago_preds, drago_gts, mantiuk_preds, mantiuk_gts
    del pu_preds_nr, pu_gts, reinhard_preds, reinhard_gts
    del reinhard_preds_firstfit, pu_preds_firstfit, raw_preds, im_paths
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

# ----------------------------
# Final aggregates + CSV
# ----------------------------
def _mean_or_nan(arr):
    arr = [v for v in arr if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else float("nan")


fieldnames = []
row = {}

if _enabled("CVVDP"):
    row["CVVDP"]   = _mean_or_nan(cvvdp_scores);   fieldnames.append("CVVDP")
if _enabled("HDR-VDP3"):
    row["HDR-VDP3"] = _mean_or_nan(hdrvdp3_scores); fieldnames.append("HDR-VDP3")
if _enabled("PU-PSNR"):
    row["PU-PSNR"]  = _mean_or_nan(psnr_scores);    fieldnames.append("PU-PSNR")
if _enabled("PU-VSI"):
    row["PU-VSI"]   = _mean_or_nan(vsi_scores);     fieldnames.append("PU-VSI")
if _enabled("PU-PIQE"):
    row["PU-PIQE"]  = _mean_or_nan(piqe_scores);    fieldnames.append("PU-PIQE")
if _enabled("PU-LPIPS"):
    row["PU-LPIPS"] = _mean_or_nan(lpips_scores);   fieldnames.append("PU-LPIPS")
if _enabled("PU-TF"):
    row["PU-TF"]    = _mean_or_nan(putf_scores);    fieldnames.append("PU-TF")

# Compute FID/FVD aggregates
for name in ("PU-FID", "Drago-FID", "Mantiuk-FID", "Reinhard-FID"):
    if _enabled(name):
        try:
            row[name] = float(compute_fid(fid_objs[name]))
        except Exception as e:
            print(f"  {name} compute failed: {e}")
            row[name] = float("nan")
        fieldnames.append(name)
for name in ("PU-FVD", "Drago-FVD", "Mantiuk-FVD", "Reinhard-FVD",
             "PU-FVD-FIRST", "Reinhard-FVD-FIRST"):
    if _enabled(name):
        try:
            row[name] = float(compute_fvd(fvd_objs[name]))
        except Exception as e:
            print(f"  {name} compute failed: {e}")
            row[name] = float("nan")
        fieldnames.append(name)

print("\n── Final results ──")
for k in fieldnames:
    print(f"  {k:>22}: {row[k]}")

out_csv = (Path(EVAL_OUTPUT_DIR) /
    f"results_{args.method}_{args.dataset}_{args.type}_{NUM_FILES}_ds{DS}.csv")
Path(EVAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader(); writer.writerow(row)
print(f"\nWrote: {out_csv}")