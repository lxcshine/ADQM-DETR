# ADQM-DETR

**Open-source PyTorch implementation of axis-decoupled quality estimation for real-time oriented object detection.** Axis-level alignment scores supervise and recalibrate box confidence on DIOR-R and DOTA-1.0. Built on RT-DETRv2-OBB with TensorRT-compatible deployment.

> This is a **non-official reimplementation** of the ADQM module described in *"Rethinking Localization Quality in Real-Time Oriented Detection"* (RiO-DETR). The original code and weights are not publicly released due to institutional IP restrictions. This repo reproduces the ADQM training/inference pipeline on the public RT-DETRv2-OBB baseline.

## Overview

Conventional oriented detectors use classification confidence as the box-level score, which cannot distinguish *angular deviation* (minor-axis collapse) from *center translation* (uniform shift). ADQM (Axis-Decoupled Quality Module) adds a lightweight head (0.09% parameters) on the decoder embeddings that predicts per-axis alignment quality:

- **Axis-level quality scores** `p_parallel`, `p_perp` (major/minor axis), supervised by the Axis Alignment Rate (AAR) between prediction and GT.
- **Directional predictions** `s_parallel`, `s_perp`, supervised by binary direction labels.
- **Box-level confidence aggregation** (inference only): `p_box = p_cls × p_parallel × p_perp`, used for thresholding and ranking instead of `p_cls`.
- **Calibration**: reduces Localization-aware Expected Calibration Error (LaECE) — the confidence becomes proportional to the actual IoU.

All ops in the head are TRT-compatible (`Linear / Sigmoid / ReLU / cat`), exportable to ONNX.

## Repository Structure

```
adqm/                        # ADQM core module
├── aar.py                   # AAR + direction label computation (training only, Eq.9/10)
├── axis_quality_head.py     # AxisQualityHead (Eq.12-15), train & inference
├── losses.py                # AAR (BCE) + direction losses (Eq.16-19)
└── test_aar.py              # unit tests (9 cases)
engine/                      # DEIM/RT-DETR training framework (modified)
├── rtv4/rtdetrv2_obb_decoder.py  # decoder + AxisQualityHead hook
├── rtv4/rtv4_obb_criterion.py    # criterion + ADQM losses
├── rtv4/postprocessor.py         # p_box aggregation in post-processing
└── misc/vis_obb.py               # OBB visualization
configs/
├── rtdetrv2_obb/
│   ├── rtdetrv2_obb_hgnetv2_m_diorr.yml          # base config (ADQM enabled)
│   ├── rtdetrv2_obb_hgnetv2_m_diorr_ft.yml      # full-data fine-tune
│   └── rtdetrv2_obb_hgnetv2_m_diorr_baseline.yml# ADQM-disabled baseline
└── dataset/dior_detection.yml                     # multi-IoU evaluator (AP50/75/50-95)
tools/
├── inference/torch_inf_obb.py    # inference + visualization + DOTA submission + latency benchmark
├── inference/visualize_gt_obb.py# GT visualization
├── analysis/generate_calibration_json.py  # match preds to GT, dump calibration JSON
├── analysis/calibration_analysis.py      # LaECE + axis ECE + reliability diagram
├── analysis/error_mode_disentanglement.py# translation vs angular error mode analysis
└── deployment/export_onnx.py     # ONNX export (TRT-compatible)
train.py                     # training / evaluation entry point
model_zoo/                   # result manifests and metrics
```

## Installation

```bash
conda create -n adqm python=3.10 -y
conda activate adqm
pip install -r requirements.txt
# PyTorch >= 2.0 with CUDA; tested with torch 2.0.1+cu118
```

## Data Preparation

### DIOR-R

```
datasets/DIOR-R/
├── JPEGImages-trainval/    # training + val images (800x800)
├── JPEGImages-test/        # test images
├── Annotations/Oriented Bounding Boxes/   # XML annotations
└── ImageSets/Main/         # train.txt / val.txt / test.txt (+ optional smoke lists)
```

### DOTA-1.0

Crop originals into 1024x1024 sub-images with stride 824 (200-pixel overlap):

```
datasets/DOTA/
├── trainval1024/  test1024/   # cropped patches
└── annotations/                 # txt labels (x1 y1 x2 y2 x3 y3 x4 y4 category difficult)
```

## Training

All commands run from the repo root. AMP is required to fit an 8 GB GPU (peak ~6.5 GB at batch 8, 800x800 input).

```bash
# Train with ADQM (subset config as shipped; edit ann_file in configs/dataset/dior_detection.yml for your split)
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml -d cuda:0 --use-amp

# Fine-tune from a checkpoint on full data (loads EMA weights only, restarts scheduler)
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr_ft.yml \
    -t outputs/rtdetrv2_obb_m_diorr_adqm/best_stg1.pth -d cuda:0 --use-amp

# Resume an interrupted run (epoch / optimizer / scheduler / EMA state restored)
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr_ft.yml \
    -r outputs/rtdetrv2_obb_m_diorr_adqm_ft/last.pth -d cuda:0 --use-amp

# Train the ADQM-disabled baseline for ablation
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr_baseline.yml -d cuda:0 --use-amp
```

Notes:
- `-t` (tuning) loads model weights only; `-r` (resume) restores the full training state.
- Per-epoch metrics (AP50 / AP75 / AP50_95 / L_AAR / L_dir / L_det) are appended to `outputs/<run>/convergence.csv`.
- Windows: use `-d cuda` or `cuda:0` (both supported); large checkpoint saves retry on transient file locks.

## Evaluation

```bash
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml \
    -r outputs/rtdetrv2_obb_m_diorr_adqm/best_stg1.pth --test-only -d cuda:0
# -> OBB_METRICS AP50=... AP75=... AP50_95=...
```

## Inference

```bash
python tools/inference/torch_inf_obb.py \
    -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml \
    -r outputs/rtdetrv2_obb_m_diorr_adqm/best_stg1.pth \
    -i datasets/DIOR-R/JPEGImages-test -d cuda:0 \
    -o outputs/infer_diorr_adqm -t 0.35 -b 8
```

Outputs (per image):
- `vis/` — detection visualization; each box labeled `class p_box|p_parallel|p_perp`
- `axis/` — per-detection axis info `(cx, cy, w, h, theta, p_box, p_parallel, p_perp, s_parallel, s_perp)`
- `dota/Task1_<class>.txt` — DOTA evaluation server submission format

Baseline models (ADQM disabled) automatically fall back to `p_cls` confidence.

**Latency benchmark** (AxisQualityHead on vs off, expectation < 0.5 ms):

```bash
python tools/inference/torch_inf_obb.py -c ... -r ... --benchmark -d cuda:0
```

## Calibration & Error Mode Analysis

```bash
# 1) Match predictions to GT, compute AAR, dump JSON (per method)
python tools/analysis/generate_calibration_json.py \
    -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml \
    -r outputs/rtdetrv2_obb_m_diorr_adqm/best_stg1.pth \
    -i datasets/DIOR-R/JPEGImages-test \
    -a "datasets/DIOR-R/Annotations/Oriented Bounding Boxes" \
    -o outputs/calibration/adqm_results.json -d cuda:0 -b 8

# 2) LaECE + axis-level ECE + reliability diagram (PDF, Times New Roman, 300 DPI)
python tools/analysis/calibration_analysis.py \
    --baseline-json outputs/calibration/baseline_results.json \
    --adqm-json outputs/calibration/adqm_results.json --output-dir outputs/calibration

# 3) Error mode disentanglement (translation- vs angular-error-dominant samples)
python tools/analysis/error_mode_disentanglement.py \
    --baseline-json outputs/calibration/baseline_results.json \
    --adqm-json outputs/calibration/adqm_results.json --output-dir outputs/calibration
```

## ONNX Export

```bash
python tools/deployment/export_onnx.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml \
    -r outputs/rtdetrv2_obb_m_diorr_adqm/best_stg1.pth --check
```

## Results (DIOR-R test, 800x800)

| Model | AP50 | AP75 | AP50:95 |
|---|---|---|---|
| RT-DETRv2-OBB-M baseline (subset) | — | — | — |
| + ADQM (subset, 102 ep) | 64.2 | 44.0 | 42.1 |
| + ADQM (full-data fine-tune, best so far) | 72.1 | 50.5 | — |

See `model_zoo/` for manifests and per-class metrics. Full-data fine-tuning results will be updated on completion.


## Acknowledgements

This implementation is built on [RT-DETR](https://github.com/lyuwenyu/RT-DETR) / [DEIM](https://github.com/Intelligent-Perception-AI/DEIM). Thanks to the original authors for the open-source baseline.

## License

Apache 2.0 (inherited from RT-DETR).
