# Generating HDR Video from SDR Video

**SaiKiran Tedla, Francesco Banterle, Trevor Canham, Karanpreet Raja, David B. Lindell, Kiriakos N. Kutulakos, Jiacheng Li, Feiran Li, Daisuke Iso**

### 📄 [Paper (arXiv)](https://arxiv.org/abs/2605.14703)
### 🌐 [Project Page](https://sdr2hdrvideo.github.io/)

---

## 📌 Citation

If you use our code or method, please cite:

```bibtex
@article{Tedla2026SDR2HDR,
  title   = {Generating HDR Video from SDR Video},
  author  = {Tedla, SaiKiran and Banterle, Francesco and Canham, Trevor and Raja, Karanpreet
             and Lindell, David B. and Kutulakos, Kiriakos N. and Li, Jiacheng and Li, Feiran
             and Iso, Daisuke},
  journal = {arXiv preprint arXiv:2605.14703},
  year    = {2026}
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
pip install -r requirements.txt
pip install -e .
```

---

## 📂 Dataset Setup

Download the [Stuttgart HDR Video Dataset](http://hdm-hdr-2014.hdm-stuttgart.de/) yourself (access is gated by HDM Stuttgart) and place scenes under:

```
data/stuttgart/<scene_name>/
```

Then build train/val splits:

```bash
python setup_splits.py stuttgart
```

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

Run the held-out validation/test pipeline:

```bash
bash test.sh   # uses diffsynth/configs/threeexposures_crffixed_test_val.yaml
```

Or run inference on a single video:

```bash
python inference.py --input_dir <path_to_input_frames> --output_dir <output_dir>
```

Output is written as EXR frames (linear HDR radiance).

---

## Repo Structure

```
diffsynth/                          # forked video-diffusion framework (Wan2.2 support + our additions)
  models/wan_video_dit.py           # Multi-Exposure Video Model
  models/wan_video_vae_merge_decoder.py  # Video Merging Model
  trainers/stuttgart_dataset*.py    # dataset loaders
  configs/                          # training/testing configs
examples/wanvideo/model_training/   # train.py / train_decoder.py entry points
setup_splits.py                     # build train/val splits from raw dataset
inference.py                        # single-video inference
test.py                             # test/validation loop
finetune.sh / finetune_decoder.sh / test.sh / val.sh
```

---

### 📨 Contact

For questions, please reach out through the [project page](https://sdr2hdrvideo.github.io/) or contact [SaiKiran Tedla](mailto:tedlasai@cs.toronto.edu).
