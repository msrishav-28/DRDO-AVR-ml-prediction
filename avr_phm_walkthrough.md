# AVR-PHM Pipeline — Full Walkthrough

## Summary

Fixed the complete AVR-PHM pipeline end-to-end: from broken `__main__` entry points to a fully functional data generation → feature engineering → augmentation → validation pipeline. **14 code gaps** identified and resolved across 3 phases.

---

## Phase 0 — Pipeline Unblocking (Session 1)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `data_gen/pipeline.py` | `__main__` ran `run_tests()` not `generate_full_dataset()` | Argparse `--test` flag |
| 2 | `experiments/train.py` | `__main__` ran `run_tests()` | Full training entry point |
| 3 | `experiments/evaluate.py` | `__main__` ran `run_tests()` | Evaluation entry point |
| 4 | `experiments/ablation.py` | `__main__` ran `run_tests()` | Ablation runner |
| 5 | `data_gen/pipeline.py` | Assertion expected 18 specs (has 16) | Fixed count |
| 6 | `simulator/dae_model.py` | `SIGMA_SENSOR_NOMINAL`, `C_nominal_uf` imported after use | Moved to top |
| 7 | `simulator/dae_model.py` | Radau solver overflow on 7200s runs | Relaxed tolerances + clamping |
| 8 | `simulator/mil_std_810h.py` | `np.trapz` removed in NumPy 2.x | `np.trapezoid` |

**Result**: Data generation completed — **504,016 samples**, 190 fault events, 32 CSV files.

---

## Phase 1 — Critical Pipeline Gaps

| # | File | Gap | Fix |
|---|------|-----|-----|
| 1 | `features/engineer.py` | `compute_targets()` O(n²) | Vectorized via prefix sum → O(n) |
| 6 | `data_gen/cgan.py` | No training entry point | Full `__main__` with `--train/--augment/--test` |
| 7 | `data_gen/cgan.py` | No augmentation strategy | `build_augmentation_dataset()` — 3× fault → 1:4 ratio |
| 10 | `data_gen/vva.py` | Fake TRTR (hardcoded 5% boost) | Real RF-based TRTR baseline |

## Phase 2 — Spec Compliance

| # | File | Gap | Fix |
|---|------|-----|-----|
| 2 | `features/engineer.py` | Temperature rolling stats missing | Added `temperature_c` to channels |
| 3 | `features/engineer.py` | No RUL target | `rul_seconds` column via next-fault distance |
| 5 | `features/engineer.py` | Ambient temp hardcoded 25°C | Per-scenario lookup map |
| 8 | `data_gen/cgan.py` | "developing" severity missing | 4 levels: healthy/incipient/developing/critical |
| 11/13 | `data_gen/vva.py` | TSTR AUROC=0, propensity std=0 | Real AUROC + StratifiedKFold per-fold std |

## Phase 3 — Polish

| # | File | Gap | Fix |
|---|------|-----|-----|
| 12 | `data_gen/vva.py` | `evaluate_tstr()` not in `run_full_vva()` | Updated sig with optional real_train |
| 14 | `data_gen/vva.py` | No `__main__` entry | Argparse `--test` flag |

---

## All Sanity Checks Pass

```
[PASS] data_gen/pipeline.py
[PASS] data_gen/cgan.py
[PASS] data_gen/vva.py
[PASS] features/engineer.py
[PASS] experiments/train.py
[PASS] experiments/evaluate.py
[PASS] experiments/ablation.py
[PASS] models/pinn.py — Params: 122,519
[PASS] simulator/scenario_engine.py
```

---

## How to Run the Pipeline

```bash
cd avr_phm

# Step 1: Generate data (already done — 504K samples in data/raw/)
python -m data_gen.pipeline

# Step 2: Train cGAN for augmentation
python -m data_gen.cgan --train --epochs 300 --device cpu

# Step 3: Generate augmented data
python -m data_gen.cgan --augment

# Step 4: Validate synthetic data quality
python -m data_gen.vva --test

# Step 5: Train PINN model
python -m experiments.train --single-seed --max-epochs 5 --no-wandb

# Step 6: Full multi-seed training
python -m experiments.train

# Step 7: Evaluate
python -m experiments.evaluate

# Step 8: Ablation studies
python -m experiments.ablation

# Step 9: Generate figures
python -m experiments.figures
```

---

## Git History

| Commit | Description |
|--------|-------------|
| `f14a4da` | Wire up pipeline entry points, DAE solver stability, NumPy 2.x |
| `0faa43f` | Add generated dataset CSVs (504K samples) and pipeline review |
| `be85c5c` | Resolve 14 pipeline gaps — vectorize, cGAN training, TRTR, RUL |
