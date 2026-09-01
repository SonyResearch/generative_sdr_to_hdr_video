
# Abstract

Examples of usage JPEG-AI Objective Quality Assessment Framework with complexity calculation library (ptflops)

# Installation

1) Create an envirnment `conda create -n jpegai_metrics_examples python=3.6.7`
2) Activate it by a command `conda activate jpegai_metrics_examples`
3) Upgrade `pip` by a command `python -m pip install --upgrade pip`
4) Install requirments from the base framework and examples: `pip install -r ../requirements.txt -r requirements.txt`

# Usage 

1) Run example with output in `txt` format by a command `python example.py --metrics msssim_torch msssim_iqa psnr vif fsim nlpd vmaf psnr_hvs`. IW-SSIM excluded from calculation, because it crashes on simple syntetic example.
2) Output should have `Metrics matched` as the last line.
3) Run example with output in `csv` format by a command `python example.py --metrics msssim_torch msssim_iqa psnr vif fsim nlpd vmaf psnr_hvs`. IW-SSIM excluded from calculation, because it crashes on simple syntetic example.
4) Output should have `Metrics matched` as the last line.