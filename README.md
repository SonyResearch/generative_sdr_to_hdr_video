# Generating HDR Video from SDR Video

**SaiKiran Tedla, Francesco Banterle, Trevor Canham, Karanpreet Raja, David B. Lindell, Kiriakos N. Kutulakos, Jiacheng Li, Feiran Li, Daisuke Iso**

### 📄 [Paper (arXiv)](https://arxiv.org/abs/2605.14703)
### 🌐 [Project Page](https://sdr2hdrvideo.github.io/)
### 💾 [Data and Checkpoints](https://ln5.sync.com/dl/763f41330#eufvner4-ix38nwnh-yv79iqna-9pwzn4y8)

---

## 📌 Citation

If you use our code or method, please cite:

```bibtex
@inproceedings{Tedla2026SDR2HDR,
  title     = {Generating HDR Video from SDR Video},
  author    = {Tedla, SaiKiran and Banterle, Francesco and Canham, Trevor and Raja, Karanpreet
               and Lindell, David B. and Kutulakos, Kiriakos N. and Li, Jiacheng and Li, Feiran
               and Iso, Daisuke},
  booktitle = {SIGGRAPH Asia 2026},
  year      = {2026},
  publisher = {ACM}
}
```

---

## Method Overview

We convert standard dynamic range (SDR) video into high dynamic range (HDR) video with two models built on top of a video diffusion backbone ([Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)):

- **Multi-Exposure Video Model (MEVM)** — predicts exposure-bracketed linear SDR video sequences from a single nonlinear SDR input. Code: `diffsynth/models/wan_video_dit.py`, trained via `finetune.sh`.
- **Video Merging Model (VMM)** — a learnable decoder that merges the exposure brackets into a single HDR output. Code: `diffsynth/models/wan_video_vae_merge_decoder.py`, trained via `finetune_decoder.sh`.

Both models are fine-tuned on the [Stuttgart HDR Video Dataset](http://hdm-hdr-2014.hdm-stuttgart.de/).

This repo is a trimmed-down, training/testing-focused fork of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — most of the general `diffsynth` framework (other model families, demo apps, etc.) has been removed; only the pieces needed for this method remain.

---

## 🔧 Environment Setup

```bash
conda create -n diffsynth python=3.10 -y
conda activate diffsynth
pip install -r requirements.txt
pip install -e .
```

---

## 📦 Downloads

Trained checkpoints and evaluation data are too large for git, so they're distributed separately via the **Data and Checkpoints** link at the top of this page:

- **`checkpoints.zip`** — our trained unet + decoder checkpoints. See **Model / checkpoint layout** below.
- **`inputs_gt.zip`** — SDR inputs + HDR ground truth for both eval datasets (stuttgart, ubc), for reproducing evaluation without the raw gated datasets. See **Dataset Setup** below.
- **`<method>.zip`** (e.g. `ours.zip`, containing `ours_stuttgart/` and `ours_ubc/`) — precomputed predictions for a given method. Zips for the other baseline methods' results are there too, if you want to run your own comparisons without regenerating them. Unzip directly into `evaluations/` to use them with the metrics pipeline without re-running inference yourself — see **Metrics / Evaluation** below.

---

## 🚀 Quick Start (try it now)

A small demo clip is bundled in `assets/demo_input/` so you can verify everything works without any dataset setup. This requires the base [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) weights and a trained checkpoint — see **Model / checkpoint layout** below for where to put them.

```bash
conda activate diffsynth
python inference.py --input_dir assets/demo_input --output_dir output/demo_output
```

Or, on a slurm cluster:

```bash
sbatch inference_slurm.sbatch                      # runs the demo clip
sbatch inference_slurm.sbatch <input_dir> <out_dir> # or your own video
```

This loads the model (~1–2 min) and generates 17 HDR frames (~2 min more) as EXR files.

### Model / checkpoint layout

The code expects (relative to the repo root):

```
models/
  Wan-AI/
    Wan2.2-TI2V-5B/          # base video diffusion backbone
    Wan2.1-T2V-1.3B/         # only used for its bundled T5 tokenizer (google/umt5-xxl)
  train/<run_name>/
    checkpoints/
      epoch-N.safetensors            # fine-tuned Multi-Exposure Video Model (dit)
      merge_checkpoint/
        epoch-M.safetensors          # fine-tuned Video Merging Model (decoder)
```

`output_path` in the config must equal `models/train/<run_name>`, and `decoder_path` must point at the merge-decoder checkpoint. `set_load_paths()` (in `examples/wanvideo/model_training/train.py`) auto-resumes from the newest `epoch-*.safetensors` it finds under `<output_path>/checkpoints`. If you don't have your own trained checkpoints yet, train them first (see **Training** below) — `model_paths: null` with no checkpoints present just runs the base pretrained model untuned.

The base `Wan-AI/` weights auto-download the first time you run anything (from HuggingFace/ModelScope via `model_id_with_origin_paths` in the config) — no manual step needed there.

**Using our trained checkpoints (`checkpoints.zip` — see Downloads above):** this repo's config (`diffsynth/configs/threeexposures_crffixed_test_val.yaml`) uses `run_name = three_exposures_crfchanging_multimode_17_val`. After downloading, from the repo root:

```bash
mkdir -p models/train/three_exposures_crfchanging_multimode_17_val
unzip checkpoints.zip -d models/train/three_exposures_crfchanging_multimode_17_val/
```

This places `epoch-0.safetensors` (unet) and `merge_checkpoint/epoch-0.safetensors` (decoder) exactly where the config expects them — nothing else to configure.

If you'd rather keep the 9.4GB extracted elsewhere (e.g. shared storage across multiple checkouts) and just link it in:

```bash
unzip checkpoints.zip -d /path/to/storage/
mkdir -p models/train/three_exposures_crfchanging_multimode_17_val
ln -s /path/to/storage/checkpoints models/train/three_exposures_crfchanging_multimode_17_val/checkpoints
```

The same pattern works for any other checkpoint directory already on disk (e.g. from a previous training run) — just symlink it in instead of unzipping:

```bash
mkdir -p models/Wan-AI models/train/<run_name>
ln -s /path/to/Wan2.2-TI2V-5B          models/Wan-AI/Wan2.2-TI2V-5B
ln -s /path/to/Wan2.1-T2V-1.3B         models/Wan-AI/Wan2.1-T2V-1.3B
ln -s /path/to/your/checkpoints_dir    models/train/<run_name>/checkpoints
```

---

## 📂 Dataset Setup

**Training** uses the [Stuttgart HDR Video Dataset](http://hdm-hdr-2014.hdm-stuttgart.de/) (access is gated by HDM Stuttgart). Place scenes under:

```
data/stuttgart/<scene_name>/
```

Then build train/val splits, which also synthesizes SDR input videos at three simulated exposure/auto-exposure settings (`under`, `over`, `auto`) from the HDR ground truth:

```bash
python setup_splits.py stuttgart
```

**Evaluation** (see **Testing** below) additionally uses a second held-out dataset, referred to as **UBC**. Set it up the same way:

```
data/ubc/<scene_name>/
python setup_splits.py ubc
```

**Skipping raw-dataset setup (`inputs_gt.zip` — see Downloads above):** if you just want to run/reproduce evaluation without gaining access to the raw gated datasets, this zip already contains exactly what `setup_splits.py` above would produce — SDR inputs (`auto`/`over`/`under`, first 17 frames/scene) and HDR ground truth (`hdr/`) for both datasets — so it replaces both `data/` and the `setup_splits.py` step entirely. From the repo root:

```bash
unzip inputs_gt.zip -d evaluations/
```

This gives you `evaluations/stuttgart/` and `evaluations/ubc/`, each with `auto/`, `over/`, `under/`, and `hdr/` subfolders — ready for **Testing** below. As with checkpoints, you can instead extract it to shared storage and symlink the two dataset folders into `evaluations/` if you'd rather not duplicate the ~4GB per checkout.

Both commands write into `evaluations/<dataset>/<exposure_type>/<scene_name>/` (synthesized SDR input, read by `test.py`) and `evaluations/<dataset>/hdr/<scene_name>/` (HDR ground truth, read by the metrics pipeline).

---

## 🏋️ Training

Two stages — the multi-exposure model, then the video merging (decoder) model:

```bash
bash finetune.sh          # trains the Multi-Exposure Video Model (dit)
bash finetune_decoder.sh  # trains the Video Merging Model (VAE decoder)
```

Configs are in `diffsynth/configs/`:
- `threeexposures_crffixed.yaml` — main multi-exposure model training
- `finetune_decoder_justmerger_stage2ea.yaml` — merge-decoder training

Edit `dataset_base_path`, `output_path`, and `decoder_path` in these configs to point at your data/checkpoint locations. `decoder_path` in the multi-exposure config should point at a checkpoint produced by the decoder training stage.

---

## 🧪 Testing

Run the held-out validation/test pipeline. This runs every scene under `evaluations/<dataset>/<ae_type>/` (produced by `setup_splits.py`) through the model and writes predicted HDR frames to `evaluations/<method>_<dataset>/<ae_type>/<scene_name>/`, where `<method>` defaults to `ours` (pass `--method` to `test.py` to change it). This naming convention lets you drop other methods' outputs alongside for comparison, e.g. `evaluations/lediff_stuttgart/`.

```bash
bash test.sh                              # dataset=stuttgart, ae_types=auto,over,under (defaults)
bash test.sh ubc                          # test on the UBC dataset instead
bash test.sh ubc auto                      # test just one exposure mode
sbatch test_slurm.sbatch                  # same, submitted as a slurm job (qos=gpu1-32h, 1 GPU)
sbatch test_slurm.sbatch ubc auto,over,under
```

Both scripts default to a single GPU but automatically scale to however many are available: `test.sh` uses however many device IDs are in `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3 bash test.sh`), and `test_slurm.sbatch` uses however many GPUs slurm allocates (e.g. `sbatch --gpus-per-node=4 test_slurm.sbatch`) — scenes are split across GPUs via `accelerate`'s distributed data loading, no code changes needed.

Or run inference on a single video (a folder of numbered PNG frames):

```bash
python inference.py --input_dir <path_to_input_frames> --output_dir <output_dir>
sbatch inference_slurm.sbatch <path_to_input_frames> <output_dir>
```

Output is written as EXR frames (linear HDR radiance). Both paths were verified end-to-end on this repo (model load → checkpoint load → 50-step denoising → merge-decoder → EXR write), including a live multi-GPU run confirming scenes are correctly split across processes.

---

## 📊 Metrics / Evaluation

`metrics/` computes the full-reference and no-reference HDR video quality metrics reported in the paper (CVVDP, HDR-VDP3, PU-PSNR, PU-VSI, PU-PIQE, PU-LPIPS, PU-TF, and PU/Drago/Mantiuk-tonemapped FID/FVD) by comparing predictions from **Testing** above against the HDR ground truth.

This uses a separate conda environment from the main `diffsynth` one (different, newer package versions):

```bash
conda env create -f metrics/metrics_environment.yml -n hdrmetric   # or: pip install -r metrics/requirements.txt
conda activate hdrmetric
```

Compute metrics for one `(dataset, method, exposure_type)` triple — reads predictions from `evaluations/<method>_<dataset>/<ae_type>/` (see **Testing**) and ground truth from `evaluations/<dataset>/hdr/`:

```bash
cd metrics
python compute_metrics_parallel.py stuttgart ours auto   # dataset, method, exposure type
python compute_metrics_parallel.py ubc ours under
```

`method` is whatever `evaluations/<method>_<dataset>/` is named — `ours` is what `test.py` writes by default, but any other method's predictions dropped in alongside it (e.g. `evaluations/lediff_stuttgart/`) work the same way, no symlinks required. Precomputed predictions for a method are also distributed as `<method>.zip` (see **Downloads**) — unzip directly into `evaluations/` (e.g. `unzip ours.zip -d evaluations/` gives you `evaluations/ours_stuttgart/` and `evaluations/ours_ubc/`) to compute metrics without running **Testing** yourself.

This writes one aggregate CSV per `(dataset, method, type)` plus one per-scene CSV to `metrics/eval_output/results_<method>_<dataset>_<type>_17_ds1.csv`. `metric_gathering.py` can drive this across many combinations at once (`--gpus 0,1,2,3 --workers-per-gpu N` to parallelize across GPUs) and skips any CSV that already exists, so it's safe to re-run after an interruption.

We verified our numbers reproduce the paper's quantitative comparison table (within normal inference-nondeterminism noise, ≲1%) across both datasets and all three exposure settings — see [Repo Structure](#repo-structure) for where the resulting CSVs live.

`ColorVideoVDP/` (CVVDP), `jpeg-ai-qaf-feature-HDR-VDP2.2-main/` (HDR-VDP3), and `NRVQA/` (PIQE) are vendored third-party metric implementations — see their own `README`/`LICENSE` files for attribution. `ColorVideoVDP` is [gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) at commit `bf0ab37` plus one addition to `pycvvdp/vvdp_data/display_models.json` (a custom `ours_standard_hdr_linear` display profile used by all our CVVDP calls).

---

## Repo Structure

```
diffsynth/                          # forked video-diffusion framework (Wan2.2 support + our additions)
  models/wan_video_dit.py           # Multi-Exposure Video Model
  models/wan_video_vae_merge_decoder.py  # Video Merging Model
  pipelines/wan_video_sampler_scheduler.py  # autoregressive multi-exposure sampling scheduler
  trainers/stuttgart_dataset*.py    # dataset loaders
  configs/                          # training/testing configs
examples/wanvideo/model_training/   # train.py / train_decoder.py entry points
metrics/                            # HDR video quality metrics (separate "hdrmetric" conda env)
  pu21.py                           # PU21 perceptually-uniform encode/decode (also used by utils.py)
  compute_metrics_parallel.py       # per (dataset, method, exposure-type) metric computation
  metric_gathering.py               # drives compute_metrics_parallel.py across many combinations
  eval_output/                      # resulting per-scene/aggregate result CSVs
  ColorVideoVDP/, jpeg-ai-qaf-feature-HDR-VDP2.2-main/, NRVQA/  # vendored metric implementations
assets/demo_input/                  # small bundled clip for a quick smoke test
setup_splits.py                     # build train/val splits + evaluation sets from raw datasets
inference.py                        # single-video inference
test.py                             # test/validation loop (multi-GPU via accelerate)
finetune.sh / finetune_decoder.sh / test.sh / val.sh
test_slurm.sbatch / inference_slurm.sbatch  # slurm launchers (defaults to 1 GPU, qos=gpu1-32h)
```

---

### 📨 Contact

For questions, please reach out through the [project page](https://sdr2hdrvideo.github.io/) or contact [SaiKiran Tedla](mailto:tedlasai@cs.toronto.edu).
