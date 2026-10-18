# GestaltGrouping

Implementation of **Differentiable Gestalt Grouping for Occlusion-Robust Salient Object Detection with Controlled Cue Interventions**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate synthetic shapes and binary saliency masks on the configured image grid.

```bash
gestalt-grouping demo --output data/generated.npz --count 96 --config config.json --seed 13
gestalt-grouping prepare --input data/generated.npz --output data/prepared
```

`x[N,3,H,W]` contains images and binary `y[N,1,H,W]` saliency masks. DUTS training and disjoint salient-object test sets can be packed from paired files. Optional occlusion masks remain evaluator metadata; compositing must preserve the original object mask as the amodal target.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
gestalt-grouping train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
gestalt-grouping evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_proximity`, `no_similarity`, `no_continuity`, `no_closure`. Reuse the prepared data and training seed:

```bash
gestalt-grouping train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
gestalt-grouping evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
gestalt-grouping plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
