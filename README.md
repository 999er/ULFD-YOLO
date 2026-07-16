# ULFD-YOLO

**An ultra-lightweight fish detection model for real-time aquatic animal monitoring on embedded platforms**

ULFD-YOLO is a deployment-oriented object detector developed from YOLOv11n for fish detection in resource-constrained aquatic monitoring systems. The model jointly redesigns the backbone, feature-fusion neck, and detection head to reduce computational cost while retaining useful detection performance in underwater scenes containing small targets, occlusion, uneven illumination, turbidity, and background clutter.

> **Release scope**
>
> This repository is a **partial research-code release** associated with the manuscript *“An Ultra-Lightweight Fish Detection Model for Real-Time Aquatic Animal Monitoring on Embedded Platforms.”* It currently provides the model configuration, training entry point, prediction-export script, bootstrap-analysis scripts, and Grad-CAM visualization code. Dataset images, trained weights, and the complete custom Ultralytics module implementation are not included in the present package.

## Highlights

- **MobileNetV4-tiny backbone** for low-cost hierarchical feature extraction.
- **Hyper-YOLO neck** for higher-order, cross-scale feature fusion, intended to recover information from small and partially occluded fish.
- **Detect_MBConv head** combining MBConv-style depthwise convolution and channel attention for efficient prediction.
- **1.3 M parameters**, **2.6 GFLOPs**, and a **3.0 MB** model file in the reported configuration.
- Real-time deployment at **19–24 FPS** on an NVIDIA Jetson Orin Nano under its 15 W device power mode, using FP32 inference and batch size 1.

## Model Overview

ULFD-YOLO coordinates three components with different roles:

1. **Backbone — MobileNetV4-tiny**  
   Reduces feature-extraction cost and provides the lightweight foundation of the detector.

2. **Neck — Hyper-YOLO**  
   Uses hypergraph-based computation to aggregate semantically related features across spatial positions and feature scales. This component is intended to compensate for representation loss caused by aggressive backbone compression.

3. **Head — Detect_MBConv**  
   Replaces the standard prediction head with an MBConv-based structure containing depthwise convolution and Squeeze-and-Excitation channel attention, reducing prediction-stage computation while preserving discriminative features.

## Reported Results

### Fish-BJ

| Metric | ULFD-YOLO |
|---|---:|
| Precision | 0.920 |
| Recall | 0.910 |
| F1-score | 0.915 |
| mAP@0.5 | 0.960 |
| mAP@0.5:0.95 | 0.732 |
| Parameters | 1.3 M |
| FLOPs | 2.6 G |
| Model size | 3.0 MB |

The primary Fish-BJ experiment used 2,721 training images and 681 held-out evaluation images from 21 aquarium fish species. Image-level bootstrap resampling with 1,000 replicates produced a 95% confidence interval of 0.946–0.973 for mAP@0.5 and 0.638–0.821 for mAP@0.5:0.95.

Across three training runs, ULFD-YOLO obtained:

- mAP@0.5: **0.969 ± 0.008**
- mAP@0.5:0.95: **0.754 ± 0.019**

### WildFish subset

After independent training on a deliberately difficult 10-species subset of WildFish, ULFD-YOLO achieved:

| Metric | Value |
|---|---:|
| Precision | 0.887 |
| Recall | 0.698 |
| F1-score | 0.781 |
| mAP@0.5 | 0.803 |

This experiment is a second-dataset replication after dataset-specific training; it is **not** a zero-shot cross-domain evaluation.

### Embedded deployment

| Item | Setting |
|---|---|
| Platform | NVIDIA Jetson Orin Nano Developer Kit |
| Precision | FP32 |
| Batch size | 1 |
| Input size | 448 × 640 |
| Device power mode | 15 W |
| Latency | approximately 42–53 ms/frame |
| Throughput | approximately 19–24 FPS |

The 15 W value refers to the configured Jetson power mode rather than externally measured total system power.

## Repository Structure

```text
.
├── models/
│   └── yolov11-M4+Hyper.yaml
├── train.py
└── utils/
    ├── generate_preds.py
    ├── paired_bootstrap_delta.py
    ├── real_bootstrap.py
    └── xin_heat.py
```

File descriptions:

- `models/yolov11-M4+Hyper.yaml`  
  ULFD-YOLO architecture configuration with the MobileNetV4-tiny backbone, Hyper-YOLO neck, and Detect_MBConv head.

- `train.py`  
  Training entry point based on the Ultralytics interface.

- `utils/generate_preds.py`  
  Exports validation predictions in YOLO TXT format with confidence scores.

- `utils/paired_bootstrap_delta.py`  
  Performs paired image-level bootstrap comparison between ULFD-YOLO and a reference detector for mAP@0.5 and mAP@0.5:0.95.

- `utils/real_bootstrap.py`  
  Legacy exploratory script based on an image-level F1 proxy. It does **not** directly recompute standard dataset-level mAP and should not be used as the primary script for reporting manuscript confidence intervals.

- `utils/xin_heat.py`  
  Generates Grad-CAM visualizations for selected model layers.

## Requirements

The manuscript experiments were conducted with a Linux-based environment using Python, PyTorch, CUDA, and the Ultralytics framework. The visualization scripts additionally require OpenCV and `pytorch-grad-cam`.

A minimal environment may be created with:

```bash
conda create -n ulfd-yolo python=3.9 -y
conda activate ulfd-yolo

pip install torch torchvision
pip install numpy opencv-python pillow tqdm matplotlib pyyaml grad-cam
```

### Custom Ultralytics modules

The model YAML references the following non-standard modules:

```text
MobileNetV4ConvTiny
HyperComputeModule
MANet
Detect_MBConv
```

These modules must be registered in a compatible modified Ultralytics codebase before the YAML can be parsed. Installing the standard `ultralytics` package alone is not sufficient for the current partial release.

Add the URL and installation instructions for the modified Ultralytics source here after it is uploaded:

```text
CUSTOM_ULTRALYTICS_REPOSITORY_URL
```

## Dataset Preparation

The training script expects an Ultralytics-compatible dataset YAML file:

```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: class_0
  1: class_1
  # ...
```

Bounding-box labels must follow the standard YOLO format:

```text
class_id center_x center_y width height
```

All coordinates are normalized to the range `[0, 1]`.

For prediction-export and bootstrap analysis, prediction files must contain:

```text
class_id center_x center_y width height confidence
```

## Training

1. Update the model, dataset, and output paths in `train.py`.
2. Confirm that the custom modules are available in the modified Ultralytics installation.
3. Start training:

```bash
python train.py
```

The provided script uses:

```text
image size:       640
epochs:           300
optimizer:        SGD
close_mosaic:     10
seed:             42
deterministic:    True
```

The current script sets `batch=256`. The manuscript reports a batch size of 512, so this setting should be reconciled before claiming exact reproduction of the published experiment.

## Exporting Predictions

Edit the weight and dataset YAML paths in:

```text
utils/generate_preds.py
```

Then run:

```bash
python utils/generate_preds.py
```

The script calls Ultralytics validation with `save_txt=True` and `save_conf=True`. The generated label directory is used by the bootstrap scripts.

## Paired Bootstrap Comparison

Edit the following variables in `utils/paired_bootstrap_delta.py`:

```python
GT_DIR = "/path/to/ground_truth/labels"
PRED_A = "/path/to/ulfd_yolo/predictions"
PRED_B = "/path/to/reference_model/predictions"
N_CLASSES = 21
N_BOOTSTRAP = 1000
SEED = 42
```

Run:

```bash
python utils/paired_bootstrap_delta.py
```

The script reports the metric for each model, the paired difference, a percentile 95% confidence interval, and an approximate two-sided bootstrap p-value.

## Grad-CAM Visualization

Install the additional dependency:

```bash
pip install grad-cam
```

Edit the weight path, device, target layers, input path, and output path in `utils/xin_heat.py`, then run:

```bash
python utils/xin_heat.py
```

Target-layer indices depend on the exact model implementation and should be checked whenever the architecture is changed.

## Data and Model Availability

The current partial package does not include the Fish-BJ source images or trained model weights.

According to the accompanying manuscript:

- Fish-BJ images and trained weights may be requested from the corresponding author, subject to institutional data-sharing conditions.
- The public release is intended to include dataset split lists, bounding-box annotations, class information, stored predictions, additional random-seed configurations and outputs, and statistical-analysis scripts.
- WildFish source images remain subject to the original dataset provider’s terms.
- Detection annotations created for the selected WildFish subset are intended for public release.

Please update this section with permanent repository or archive links when those resources are available.

## Reproducibility Notes

- Replace all hard-coded absolute paths before running the scripts.
- Record the exact commit of the modified Ultralytics codebase.
- Add the exact package versions used for the released results.
- Keep the dataset split and class order fixed across all compared models.
- Use the same confidence and NMS settings when exporting predictions for paired comparison.
- The held-out Fish-BJ subset was used for checkpoint selection and final reporting in the manuscript; it should not be described as a fully independent test set.
- The reported contribution is an accuracy–efficiency trade-off rather than an accuracy improvement over YOLOv11n.

## Citation

Citation information will be updated after publication.

```bibtex
@article{zhang_ulfd_yolo,
  title   = {An Ultra-Lightweight Fish Detection Model for Real-Time Aquatic Animal Monitoring on Embedded Platforms},
  author  = {Zhang, Hanyu and Zhang, Zhongde and Liu, Weiping},
  journal = {Animals},
  year    = {YEAR},
  doi     = {DOI}
}
```

Please also cite the original works associated with YOLO, MobileNetV4, Hyper-YOLO, MBConv/Squeeze-and-Excitation, and any third-party Grad-CAM implementation used in your experiments.

## License

A license file is not included in the current partial package. Because this project is built on and modifies Ultralytics components, the repository license must be compatible with the applicable upstream license and the licenses of all incorporated third-party code. Add a clear `LICENSE` file before public release.

## Contact

For questions about the paper, dataset access, or trained weights, contact the corresponding author:

**Weiping Liu**  
Beijing Forestry University  
Email: `lwp123@bjfu.edu.cn`
