# ULFD-YOLO

**An ultra-lightweight fish detection model for real-time aquatic animal monitoring on embedded platforms**

ULFD-YOLO is a deployment-oriented fish detector derived from YOLOv11n. It combines a custom MobileNetV4-tiny backbone, a hypergraph-based feature-fusion neck, and a Detect_MBConv prediction head to reduce computational cost while retaining useful detection performance in challenging aquatic scenes.

> **Release status**
>
> This repository contains the model configuration, source code for the added modules, training and validation entry points, trained checkpoints from three model-training seeds, a seed-0 ONNX export, the exact Fish-BJ image split list, prediction-export and statistical-analysis scripts, and a set of WildFish prediction-format TXT files.
>
> The package is **not a fully self-contained reproduction environment**. Fish-BJ images and ground-truth annotation folders, the dataset YAML files, Fish-BJ held-out prediction files, and a patched Ultralytics package with the custom modules already registered are not included. Local paths in the supplied scripts must be updated before use.

## Highlights

- **MobileNetV4-tiny backbone** for efficient hierarchical feature extraction.
- **Hyper-YOLO neck** for higher-order cross-position and cross-scale feature aggregation.
- **Detect_MBConv head** using MBConv-style efficient convolution and channel attention.
- **1.3 M parameters**, **2.6 GFLOPs**, and an approximately **3.0 MB** PyTorch checkpoint.
- **0.960 mAP@0.5** and **0.732 mAP@0.5:0.95** on Fish-BJ in the primary run.
- Approximately **19–24 FPS model inference** on an NVIDIA Jetson Orin Nano at 448 × 640, FP32, batch size 1, under the platform's 15 W `nvpmodel` power mode.

## Model Overview

ULFD-YOLO coordinates three components with different roles:

1. **Backbone — MobileNetV4-tiny**  
   Provides the primary reduction in parameter count and feature-extraction cost.

2. **Neck — Hyper-YOLO**  
   Aggregates semantically related information across positions and feature scales to compensate for representation loss caused by aggressive backbone compression.

3. **Head — Detect_MBConv**  
   Reduces prediction-stage computation while preserving channel discrimination through MBConv-style operations and Squeeze-and-Excitation attention.

The contribution is a task-oriented backbone–neck–head co-design rather than a claim of a newly invented atomic module. The complete configuration was selected as a Pareto-efficient accuracy–computation operating point: it has the lowest FLOPs among the tested component combinations while retaining 0.960 mAP@0.5.

## Reported Results

### Fish-BJ primary run

| Metric | ULFD-YOLO | YOLOv11n |
|---|---:|---:|
| Precision | 0.920 | 0.972 |
| Recall | 0.910 | 0.940 |
| F1-score | 0.915 | 0.956 |
| mAP@0.5 | 0.960 | 0.980 |
| mAP@0.5:0.95 | 0.732 | 0.848 |
| Parameters | 1.3 M | 2.6 M |
| FLOPs at 640 × 640 | 2.6 G | 6.3 G |
| Model size | 3.0 MB | 5.5 MB |

Relative to YOLOv11n, ULFD-YOLO reduces the parameter count by **50.0%** and FLOPs by **58.7%**, with an absolute mAP@0.5 decrease of 0.020. The reported contribution is therefore an accuracy–efficiency trade-off rather than an accuracy improvement over YOLOv11n.

### Fish-BJ uncertainty and repeated training

The manuscript reports the following image-level bootstrap results for the 681-image held-out subset:

| Metric | Point estimate | 95% bootstrap CI |
|---|---:|---:|
| mAP@0.5 | 0.960 | 0.946–0.973 |
| mAP@0.5:0.95 | 0.732 | 0.638–0.821 |

Across model-training seeds **0, 1, and 42**:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| ULFD-YOLO | 0.969 ± 0.008 | 0.754 ± 0.019 |
| YOLOv11n | 0.979 ± 0.001 | 0.847 ± 0.001 |

The paired image-level bootstrap reported:

```text
Delta mAP@0.5 = mAP@0.5_ULFD-YOLO - mAP@0.5_YOLOv11n
               = -0.018

95% bootstrap percentile CI: -0.026 to -0.010
```

All 1,000 paired differences were negative. A two-sided Wilcoxon signed-rank test across the 21 category-wise mAP@0.5:0.95 values gave `W = 0, p < 0.001`.

The repository includes PyTorch checkpoints for seeds 0, 1, and 42. It does **not** currently include the corresponding `results.csv` files or full training logs.

### WildFish subset

After dataset-specific training on a deliberately difficult 10-category WildFish subset, the manuscript reports:

| Metric | Value |
|---|---:|
| Precision | 0.887 |
| Recall | 0.698 |
| F1-score | 0.781 |
| mAP@0.5 | 0.803 |
| mAP@0.5:0.95 | 0.553 |

This is an evaluation on a second dataset after dataset-specific training. It is not a zero-shot transfer test from Fish-BJ.

### Embedded deployment

| Item | Setting |
|---|---|
| Platform | NVIDIA Jetson Orin Nano Developer Kit |
| Precision | FP32 |
| Batch size | 1 |
| Input size | 448 × 640 |
| Power mode | 15 W `nvpmodel` mode |
| Model-inference latency after warm-up | approximately 42–53 ms/frame |
| Throughput | approximately 19–24 FPS |

The 15 W value denotes the configured Jetson power mode, not externally measured total-system power. The reported timing is model-inference latency and does not represent a separately measured full camera-acquisition-to-display pipeline.

## Actual Repository Structure

The current release package contains:

```text
.
├── Fish-BJ_split_list.csv
├── README.md
├── models/
│   ├── yolov11_ULFD-YOLO.yaml
│   └── AddModels/
│       ├── Detect_MBConv.py
│       ├── Hyper_YOLO.py
│       └── MobileNetV4.py
├── result/
│   ├── weights_seed0/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── weights_seed1/
│   │   ├── best.pt
│   │   └── last.pt
│   └── weights_seed42/
│       ├── best.pt
│       └── last.pt
├── train.py
├── val.py
├── utils/
│   ├── generate_preds.py
│   ├── heat.py
│   ├── paired_bootstrap_delta.py
│   ├── real_bootstrap.py
│   └── wilcoxon_per_class.py
└── WildFish annotations/
    └── labels/
        └── 236 TXT files
```

### Included-file summary

| Path | Description |
|---|---|
| `Fish-BJ_split_list.csv` | Exact Fish-BJ training/held-out membership list containing 3,402 rows |
| `models/yolov11_ULFD-YOLO.yaml` | ULFD-YOLO model definition for 21 Fish-BJ categories |
| `models/AddModels/MobileNetV4.py` | MobileNetV4 implementations, including `MobileNetV4ConvTiny` |
| `models/AddModels/Hyper_YOLO.py` | Hypergraph-related modules, including `HyperComputeModule` and `MANet` |
| `models/AddModels/Detect_MBConv.py` | Detect_MBConv implementation and supporting MBConv/SE components |
| `result/weights_seed0/` | Seed-0 `best.pt`, `last.pt` |
| `result/weights_seed1/` | Seed-1 `best.pt` and `last.pt` |
| `result/weights_seed42/` | Seed-42 `best.pt` and `last.pt` |
| `train.py` | Ultralytics-based training entry point |
| `val.py` | Ultralytics-based validation entry point |
| `utils/generate_preds.py` | Exports YOLO TXT predictions with confidence values |
| `utils/paired_bootstrap_delta.py` | Recomputes paired image-level bootstrap differences from ground truth and two prediction directories |
| `utils/wilcoxon_per_class.py` | Performs the category-level paired Wilcoxon signed-rank test |
| `utils/real_bootstrap.py` | Legacy exploratory F1-proxy bootstrap script; not the primary manuscript analysis |
| `utils/heat.py` | Grad-CAM/feature-response visualization script |
| `WildFish annotations/labels/` | 236 six-column YOLO prediction-format TXT files |

## Requirements

The code was developed around a modified Ultralytics environment.

A minimal environment may be created with:

```bash
conda create -n ulfd-yolo python=3.10 -y
conda activate ulfd-yolo

pip install torch torchvision
pip install ultralytics numpy opencv-python pillow tqdm matplotlib pyyaml scipy grad-cam
```

Exact compatibility depends on the Ultralytics version used by the authors. For full reproducibility, add a pinned `requirements.txt` or `environment.yml` and record the exact Ultralytics commit.

## Registering the Custom Modules

The model YAML references the following non-standard names:

```text
MobileNetV4ConvTiny
HyperComputeModule
MANet
Detect_MBConv
```

Their source implementations are included under:

```text
models/AddModels/
```

They still need to be imported and exposed to the Ultralytics model parser. Installing the unmodified `ultralytics` package alone is not sufficient.

The exact registration procedure depends on the Ultralytics version. In general:

1. Add the module files to the modified Ultralytics source tree or another importable Python package.
2. Export the custom classes from the relevant `ultralytics.nn.modules` namespace.
3. Import the class names in the model-parsing module used by the selected Ultralytics version.
4. Confirm that `YOLO("models/yolov11_ULFD-YOLO.yaml")` can resolve all custom class names.

## Preparing the Dataset YAML

The current package does not include a dataset YAML. Create one locally, for example:

```yaml
path: /absolute/path/to/Fish-BJ
train: images/train
val: images/val

names:
  0: category_0
  1: category_1
  # ...
  20: category_20
```

Ground-truth labels must use normalized YOLO format:

```text
class_id center_x center_y width height
```

## Training

The provided `train.py` uses:

```text
image size:           640
epochs:               300
batch size:           512
optimizer:            SGD
close_mosaic:         10
primary seed:         0
deterministic mode:   enabled
```

Before running, edit the hard-coded paths in `train.py`:

```python
model = YOLO("models/yolov11_ULFD-YOLO.yaml")
model.load("/path/to/yolo11n.pt")

model.train(
    data="/path/to/fish_bj.yaml",
    project="/path/to/output",
    ...
)
```

The supplied script currently references author-local `/home/featurize/...` paths and an older YAML filename. Batch size 512 may exceed the memory available on many systems; reduce it when necessary, while noting that this changes the exact training configuration.

Run:

```bash
python train.py
```

To reproduce the repeated runs, set:

```python
seed=0
seed=1
seed=42
```

while keeping all other settings fixed.

## Validation

Before running `val.py`, edit the checkpoint and dataset paths.

The supplied script currently contains:

```python
model = YOLO("/home/.../weights/bast.pt")
```

Change `bast.pt` to `best.pt` and use a valid local path, for example:

```python
from ultralytics import YOLO

model = YOLO("result/weights_seed0/best.pt")
metrics = model.val(
    data="/path/to/fish_bj.yaml",
    imgsz=640,
)
```

Run:

```bash
python val.py
```

## Prediction Export

Edit the paths in:

```text
utils/generate_preds.py
```

Then run:

```bash
python utils/generate_preds.py
```

The script saves prediction TXT files with confidence scores:

```text
class_id center_x center_y width height confidence
```

These files can be used by the paired-bootstrap script after the corresponding ground-truth labels and reference-model predictions are prepared.

## Paired Bootstrap Analysis

Edit the following variables in:

```text
utils/paired_bootstrap_delta.py
```

```python
GT_DIR = "/path/to/Fish-BJ/labels/val"
PRED_A = "/path/to/ULFD-YOLO/predictions"
PRED_B = "/path/to/YOLOv11n/predictions"

N_CLASSES = 21
N_BOOTSTRAP = 1000
SEED = 42
```

Run:

```bash
python utils/paired_bootstrap_delta.py
```

The script recomputes mAP@0.5 and mAP@0.5:0.95 for both detectors over identical bootstrap image-index samples and reports the paired difference and percentile confidence interval.

## Category-Level Wilcoxon Test

Run:

```bash
python utils/wilcoxon_per_class.py
```

The included script compares the 21 Fish-BJ category-wise mAP@0.5:0.95 values for ULFD-YOLO and YOLOv11n.

## Grad-CAM Visualization

Edit the checkpoint, input, output, device, and target-layer settings in:

```text
utils/heat.py
```

Then run:

```bash
python utils/heat.py
```

The manuscript's Grad-CAM comparison concerns the baseline and Detect_MBConv detection heads. It should not be interpreted as independent evidence for the Hyper-YOLO neck.

## Using the Released Checkpoints

After registering the custom modules in a compatible Ultralytics installation:

```python
from ultralytics import YOLO

model = YOLO("result/weights_seed0/best.pt")
results = model.predict(
    source="/path/to/image_or_video",
    imgsz=640,
)
```

Available checkpoint directories:

```text
result/weights_seed0/
result/weights_seed1/
result/weights_seed42/
```

## Included and Missing Materials

### Included

- exact Fish-BJ split membership CSV;
- ULFD-YOLO YAML configuration;
- source files for MobileNetV4-tiny, Hyper-YOLO/MANet, and Detect_MBConv;
- training and validation entry points;
- prediction-export utility;
- paired-bootstrap and Wilcoxon scripts;
- legacy exploratory bootstrap script;
- Grad-CAM utility;
- `best.pt` and `last.pt` checkpoints for seeds 0, 1, and 42;
- 236 WildFish six-column prediction-format TXT files.

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

## Contact

For questions about the manuscript, Fish-BJ access, or the released implementation, contact:

**Weiping Liu**  
Beijing Forestry University  
Email: `lwp123@bjfu.edu.cn`
