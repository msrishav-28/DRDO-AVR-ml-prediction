# AVR-PHM: Physics-Informed Digital Twin for Military AVR Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.14.0-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DeepXDE](https://img.shields.io/badge/DeepXDE-1.11.2-blue?style=flat-square)](https://deepxde.readthedocs.io/)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45.1-purple?style=flat-square)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-DRDO--Internal-red?style=flat-square)]()

A multi-task Physics-Informed Neural Network (PINN) framework for fault prognostics and remaining useful life estimation of military-grade Automatic Voltage Regulators. Built on a high-fidelity synchronous machine DAE simulator with MIL-STD-1275E/810H compliance.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Physics Simulator](#physics-simulator)
- [Models](#models)
- [Evaluation](#evaluation)
- [Experiments](#experiments)
- [Configuration](#configuration)
- [Research Notebook](#research-notebook)
- [Citation](#citation)

---

## Overview

AVR-PHM addresses the challenge of predictive maintenance for military vehicle power systems operating under extreme environmental and combat conditions. The system combines:

1. **High-fidelity physics simulation** -- 8th-order synchronous machine DAE with IEEE Type I AVR
2. **MIL-STD compliance** -- Transient waveform overlays per MIL-STD-1275E and vibration/shock per MIL-STD-810H
3. **Physics-informed deep learning** -- Multi-task PINN with physics-residual loss from d-q axis generator equations
4. **Rare-fault augmentation** -- Conditional WGAN-GP with VVA-validated synthetic data
5. **PHM-grade evaluation** -- 17+ metrics including NASA RUL scoring, calibration analysis, and adversarial robustness

### Key Contributions

| Contribution | Description |
|-------------|-------------|
| Physics-Informed Multi-Task Learning | Composite loss function integrating voltage dynamics constraints with multi-horizon fault prediction |
| MIL-STD-Compliant Simulation | First open DAE-based AVR simulator with full MIL-STD-1275E/810H transient waveform overlays |
| Conditional WGAN-GP Augmentation | GRU-based generator with 14-dim condition encoding for rare military fault scenario augmentation |
| Comprehensive Evaluation | PHM-grade metrics with multi-seed reproducibility, significance testing, and adversarial robustness |

---

## Architecture

```
                    +---------------------+
                    |   Physics Simulator  |
                    |  (8th-order DAE +    |
                    |   MIL-STD overlays)  |
                    +---------+-----------+
                              |
                    +---------v-----------+
                    |  Fault Mechanisms    |
                    |  (Coffin-Manson,     |
                    |   Arrhenius, Miner)  |
                    +---------+-----------+
                              |
               +--------------v--------------+
               |     Data Generation         |
               |  (18 runs, 7 scenarios,     |
               |   350K+ samples)            |
               +--------------+--------------+
                              |
               +--------------v--------------+
               |   WGAN-GP Augmentation      |
               |  (GRU Generator + BiGRU     |
               |   Critic, VVA validation)   |
               +--------------+--------------+
                              |
               +--------------v--------------+
               |   Feature Engineering       |
               |  (9 physics features,       |
               |   lags, rolling stats)      |
               +--------------+--------------+
                              |
          +-------------------v-------------------+
          |                                       |
    +-----v------+                        +-------v-------+
    | Multi-task  |                        |   Baselines   |
    |    PINN     |                        | (Threshold,   |
    | (8 heads,   |                        |  RF, RAE,     |
    |  ~180K      |                        |  PatchTST)    |
    |  params)    |                        +-------+-------+
    +-----+------+                                |
          |                                       |
          +-------------------v-------------------+
                              |
               +--------------v--------------+
               |    PHM-Grade Evaluation     |
               |  (17+ metrics, Wilcoxon,    |
               |   calibration, SHAP,        |
               |   adversarial robustness)   |
               +-----------------------------+
```

### PINN Task Heads

| Head | Task | Output | Loss Function |
|------|------|--------|---------------|
| 1 | Fault warning (1s) | Binary | Focal Loss (gamma=2.0) |
| 2 | Fault warning (5s) | Binary | Focal Loss |
| 3 | Fault warning (10s) | Binary | Focal Loss |
| 4 | Fault warning (30s) | Binary | Focal Loss |
| 5 | Mechanism classification | 4-class | Cross-Entropy |
| 6 | Voltage forecast | 10-step | MSE |
| 7 | Severity estimation | 4-class | Cross-Entropy |
| 8 | RUL estimation | Scalar | Asymmetric MSE |

**Composite Loss:**

```
L_total = 0.5 * L_data + 0.3 * L_physics + 0.2 * L_fault
```

---

## Repository Structure

```
avr_phm/
|-- config/
|   |-- __init__.py          # YAML loader, seed setting, device detection
|   |-- scenarios.yaml       # 7 scenario definitions
|   |-- milstd.yaml          # MIL-STD-1275E/810H parameters
|   |-- model.yaml           # Model hyperparameters
|   +-- paths.yaml           # Centralized path configuration
|
|-- simulator/
|   |-- __init__.py
|   |-- constants.py         # Physical constants (Ra, Xd, H, etc.)
|   |-- dae_model.py         # 8th-order DAE + IEEE Type I AVR (Radau)
|   |-- mil_std_1275e.py     # IES, cranking, spike, load dump waveforms
|   |-- mil_std_810h.py      # PSD vibration, ballistic shock
|   |-- fault_mechanisms.py  # 3 degradation models
|   |-- scenario_engine.py   # Scenario assembly orchestrator
|   +-- validator.py         # Output sanity checks (8+ checks)
|
|-- data_gen/
|   |-- __init__.py
|   |-- pipeline.py          # 18-run dataset generation with resume
|   |-- cgan.py              # Conditional WGAN-GP (GRU/BiGRU)
|   +-- vva.py               # VVA suite (MMD, propensity, TSTR, ACF)
|
|-- features/
|   |-- __init__.py
|   +-- engineer.py          # Feature engineering + time-aware splits
|
|-- models/
|   |-- __init__.py
|   |-- pinn.py              # Multi-task PINN (8 heads, ~180K params)
|   |-- baseline_threshold.py # Rule-based detector (MIL-STD limits)
|   |-- baseline_rf.py       # Random Forest + GBM forecaster
|   |-- recurrent_ae.py      # Bidirectional GRU autoencoder
|   +-- patchtst.py          # PatchTST transformer forecaster
|
|-- eval/
|   |-- __init__.py
|   |-- phm_metrics.py       # 17+ PHM metrics + RUL + complexity
|   |-- calibration.py       # ECE/MCE + reliability diagrams
|   |-- xai.py               # SHAP GradientExplainer
|   +-- adversarial.py       # FGSM/PGD with physical constraints
|
|-- experiments/
|   |-- __init__.py
|   |-- train.py             # Multi-seed training harness (wandb)
|   |-- evaluate.py          # Unified evaluation + significance tests
|   |-- ablation.py          # 5 ablation studies + 3 HP sweeps
|   +-- figures.py           # 8 publication figures (IEEE format)
|
|-- notebooks/
|   +-- avr_phm_research_demo.py  # Full research demonstration
|
|-- data/                    # Generated datasets (gitignored)
|-- outputs/                 # Results, checkpoints, figures
|-- tests/                   # Unit tests
|
|-- requirements.txt         # Pinned dependency versions
|-- setup.py                 # Package installation
|-- pyproject.toml           # Build configuration
+-- .env                     # Environment variables template
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd DRDO

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r avr_phm/requirements.txt

# Install package in development mode
pip install -e avr_phm/

# Configure environment
cp avr_phm/.env avr_phm/.env.local
# Edit .env.local with your WANDB_API_KEY
```

### Dependency Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.5.0 | Neural network training and inference |
| `deepxde` | 1.11.2 | Physics-informed neural network utilities |
| `scipy` | 1.14.0 | DAE solver (Radau IIA) |
| `numpy` | 1.26.4 | Numerical computation |
| `pandas` | 2.2.3 | Data manipulation |
| `scikit-learn` | 1.5.2 | Baselines, metrics, cross-validation |
| `shap` | 0.45.1 | Model explainability |
| `wandb` | 0.18.5 | Experiment tracking |
| `matplotlib` | 3.9.2 | Publication figures |
| `seaborn` | 0.13.2 | Statistical visualization |
| `pyyaml` | 6.0.2 | Configuration loading |

---

## Quick Start

### 1. Generate Dataset

```bash
cd avr_phm
python -m data_gen.pipeline
```

This generates 18 simulation runs across 7 scenarios (~350K samples total). Resume is automatic -- if interrupted, rerun the same command.

### 2. Train PINN

```bash
python -m experiments.train
```

Training uses:
- 5 random seeds: `[42, 123, 456, 789, 2026]`
- Early stopping with patience=100
- Checkpoint saving every 50 epochs
- Mixed precision (AMP) on CUDA
- Gradient clipping (max norm=1.0)
- W&B experiment logging

### 3. Evaluate All Models

```bash
python -m experiments.evaluate
```

### 4. Run Ablation Studies

```bash
python -m experiments.ablation
```

### 5. Generate Publication Figures

```bash
python -m experiments.figures
```

---

## Physics Simulator

### DAE Model

The core simulator solves an 8th-order differential-algebraic equation system modeling a synchronous generator with IEEE Type I AVR:

**State vector:**

```
x = [delta, omega, Eq', Ed', Eq'', Ed'', Vf, Vr]^T
```

**Solver:** SciPy `solve_ivp` with `method='Radau'` (stiff-aware, implicit Runge-Kutta).

### MIL-STD-1275E Transients

| Waveform | Peak | Duration | Trigger |
|----------|------|----------|---------|
| Initial Engagement Surge | -20V dip | 50ms | Engine start |
| Cranking Depression | Down to 16V | 5-20s | Cold cranking |
| Voltage Spike | +250V | 70us | Inductive kickback |
| Load Dump Surge | +100V | 40ms | Sudden load disconnect |

### Fault Mechanisms

| Mechanism | Physics Model | Observable Effect |
|-----------|--------------|-------------------|
| Thyristor Thermal Fatigue | Coffin-Manson cycle counting | Increased forward voltage drop |
| Capacitor Degradation | Arrhenius lifetime acceleration | Capacitance loss, ripple increase |
| Terminal Loosening | Miner's rule cumulative damage | Contact impedance noise |

---

## Models

### PINN (Primary)

- **Encoder:** 3-layer 1D-CNN (kernel=5, 64 filters) + Global Average Pooling
- **Shared:** Dense(128) + Dropout(0.15) + LayerNorm + GELU
- **Heads:** 8 parallel task heads
- **Parameters:** ~180,000 (edge-deployable)
- **UQ:** MC Dropout (T=50 forward passes)

### Baselines

| Model | Type | Key Parameters |
|-------|------|---------------|
| Threshold | Rule-based | V_under=23.5V, V_over=32.5V |
| Random Forest | Ensemble | 500 trees, max_depth=20, balanced |
| 1D-CNN | Deep Learning | 3-layer CNN (kernel=5, 64 filters) + GAP (Physics Ablation) |
| LSTM | Deep Learning | 2-layer LSTM (hidden=64, dropout=0.15) |
| Recurrent AE | Unsupervised | BiGRU [128,64], latent=32, threshold=mean+2.5*std |
| PatchTST | Transformer | patch=16, stride=8, 4 heads, 2 layers |

---

## Evaluation

### Metrics Suite

**Classification:** Precision, Recall, F1 (macro/binary), AUC-ROC, AUC-PR, Specificity, FAR, MDR

**Forecasting:** MAE, RMSE, R-squared (overall and per-step)

**RUL:** NASA Scoring Function, alpha-lambda accuracy (alpha=0.2)

**Calibration:** ECE (15 bins), MCE, reliability diagrams

**Robustness:** FGSM and PGD attacks with +/-2V physical plausibility constraint

**Complexity:** Parameter count, inference latency (ms), throughput (samples/sec), memory (MB)

### Statistical Significance

All comparisons use the **Wilcoxon Signed-Rank Test** across 5 seeds at alpha=0.05.

---

## Experiments

### Ablation Studies

| Experiment | Modification | Purpose |
|-----------|-------------|---------|
| `full_pinn` | No changes (control) | Baseline performance |
| `no_physics` | lambda_physics = 0 | Isolate physics loss contribution |
| `no_cgan` | No augmented data | Isolate WGAN-GP contribution |
| `no_milstd` | No MIL-STD overlays | Isolate transient overlay contribution |
| `single_task` | Only fault_10s head | Isolate multi-task benefit |
| `no_curriculum` | All scenarios at once | Isolate curriculum learning |

### Hyperparameter Sensitivity

| Parameter | Sweep Values |
|-----------|-------------|
| Physics loss weight | 0.0, 0.1, 0.3, 0.5, 0.7, 1.0 |
| cGAN augmentation ratio | 0.0, 0.25, 0.5, 1.0, 2.0 |
| Dropout rate | 0.0, 0.05, 0.10, 0.15, 0.20, 0.30 |

---

## Configuration

All parameters are centralized in YAML configuration files under `config/`:

- **`scenarios.yaml`** -- 7 operating scenarios with temperature, duration, fault probabilities
- **`milstd.yaml`** -- MIL-STD-1275E voltage transient and MIL-STD-810H vibration parameters
- **`model.yaml`** -- PINN, RAE, RF, and WGAN-GP hyperparameters
- **`paths.yaml`** -- All file paths (no hardcoded paths in source code)

---

## Research Notebook

A comprehensive research demonstration notebook is provided:

```
avr_phm/notebooks/avr_phm_research_demo.py
```

The notebook uses Jupytext `percent` format. To open in Jupyter:

```bash
pip install jupytext
jupyter notebook notebooks/avr_phm_research_demo.py
```

Or convert to `.ipynb`:

```bash
jupytext --to notebook notebooks/avr_phm_research_demo.py
```

The notebook covers all 12 pipeline stages with live simulation outputs, feature engineering visualizations, model architecture summaries, and evaluation metric demonstrations.

---

## Reproducibility

- All random seeds are pinned: `numpy`, `torch`, `random`, `sklearn`
- Multi-seed evaluation: `[42, 123, 456, 789, 2026]`
- Exact dependency versions in `requirements.txt`
- Deterministic CUDA: `torch.backends.cudnn.deterministic = True`
- Resume logic for interrupted training and data generation

---

## Citation

```bibtex
@article{avr_phm_2026,
  title={Physics-Informed Digital Twin for Military AVR Predictive Maintenance},
  author={},
  journal={},
  year={2026},
  note={Under review}
}
```

---

## License

This project is developed under DRDO research guidelines. All rights reserved.
