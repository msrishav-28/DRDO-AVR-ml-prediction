# Features & Augmentation Pipeline — Deep Review

Thorough code review of `features/engineer.py`, `data_gen/cgan.py`, and `data_gen/vva.py` cross-referenced against the master plan specification (Sections 5–7).

---

## Inventory of All Modules

| Module | File | Lines | Status |
|--------|------|-------|--------|
| Feature Engineering | `features/engineer.py` | 504 | ✅ Implemented |
| WGAN-GP Augmentation | `data_gen/cgan.py` | 533 | ✅ Implemented |
| VVA Suite | `data_gen/vva.py` | 412 | ⚠️ Partial |
| PINN Model | `models/pinn.py` | 536 | ✅ Implemented |
| Threshold Baseline | `models/baseline_threshold.py` | — | ✅ Exists |
| RF Baseline | `models/baseline_rf.py` | — | ✅ Exists |
| Recurrent AE | `models/recurrent_ae.py` | — | ✅ Exists |
| PatchTST | `models/patchtst.py` | — | ✅ Exists |
| PHM Metrics | `eval/phm_metrics.py` | — | ✅ Exists |
| Calibration | `eval/calibration.py` | — | ✅ Exists |
| XAI/SHAP | `eval/xai.py` | — | ✅ Exists |
| `tests/` directory | — | — | ❌ Not created |

---

## Gap Analysis — `features/engineer.py`

### ✅ What's Correct
- FEATURE_SPEC matches master plan Section 7 exactly (lags, rolling, physics)
- Physics features (dV/dt, dI/dt, power, impedance, thermal stress, ripple) all present
- Scenario one-hot encoding: 7-dim as specified
- Target variables: multi-horizon (1s/5s/10s/30s), mechanism, severity, voltage forecast — all present
- `create_time_aware_splits()` correctly holds out `emp_simulation` and `desert_heat + artillery_firing`
- NaN handling via bfill/ffill/fillna(0) is correct for lag/rolling edge effects

### ⚠️ Gaps Found

| # | Severity | Gap | Details |
|---|----------|-----|---------|
| 1 | **HIGH** | `compute_targets()` — forward-looking labels use O(n²) loop | Lines 324-335: the reverse cumsum approach is correct in concept but the inner loop iterates O(n) per label, making it O(n²) total. For 504K samples this is ~250 billion ops. Should use vectorized rolling sum. |
| 2 | **MEDIUM** | Rolling features only on voltage and current, not temperature | Master plan says "per channel: V, I, T" for rolling stats, but `compute_rolling_features()` only rolls `voltage_v` and `current_a` (line 129). Temperature rolling stats are missing. |
| 3 | **MEDIUM** | No RUL (Remaining Useful Life) target | Master plan Section 10 mentions "Asymmetric RUL Error" as a metric, but there's no RUL target column generated. The PINN model may have a `rul` output head but no training target is computed. |
| 4 | **LOW** | `create_time_aware_splits()` test split outputs unused | The training harness accesses `splits["train"]` and `splits["val"]` but also returns `test_held_out_scenario` and `test_stress_combo` which go unused in the current pipeline. |
| 5 | **LOW** | `ambient_temp_c` hardcoded to 25.0 in `engineer_all_features()` | Should be extracted from the scenario params per-file (e.g., arctic = -40°C, desert = 65°C). This means thermal_stress_index is wrong for non-baseline scenarios. |

---

## Gap Analysis — `data_gen/cgan.py`

### ✅ What's Correct
- Architecture matches master plan Section 5 exactly (Generator: Dense→GRU→Dense(3), Critic: Dense→BiGRU→Dense(1))
- WGAN-GP training loop with n_critic=5, gradient_penalty λ=10 — correct
- Condition encoding: 14-dim (7 scenario + 4 mechanism + 3 severity) — correct
- Adam with betas=(0.0, 0.9) per WGAN-GP paper — correct
- Checkpoint resume logic — correct
- `generate_augmentation_data()` for inference — correct

### ⚠️ Gaps Found

| # | Severity | Gap | Details |
|---|----------|-----|---------|
| 6 | **HIGH** | No `__main__` entry point for training the cGAN | The file has `run_tests()` in `__main__` but no way to actually train. Need a `--train` mode that loads windowed simulator data and runs `train_cgan()`. |
| 7 | **HIGH** | No augmentation strategy implementation | Master plan Section 5 specifies: "generate 3× as many fault samples as healthy samples, then down-sample to 1:4 ratio for final training set." This logic doesn't exist anywhere. |
| 8 | **MEDIUM** | Missing severity level "developing" | `SEVERITY_LEVELS = ["healthy", "incipient", "critical"]` but the fault mechanisms produce 4 levels: healthy/incipient/developing/critical. The "developing" level is collapsed or lost. |
| 9 | **LOW** | No W&B logging in cGAN training | Master plan requires all experiment tracking via wandb — the cGAN training loop only prints to stdout. |

---

## Gap Analysis — `data_gen/vva.py`

### ✅ What's Correct
- All 4 VVA metrics implemented: MMD, Propensity Score, TSTR, ACF Similarity
- MMD with multi-kernel (σ ∈ {0.1, 0.5, 1.0, 5.0, 10.0}) and median heuristic — correct
- Propensity score with 5-fold cross-validated logistic regression — correct
- ACF computed at lags [1, 5, 10, 20, 50] with Pearson correlation — correct
- Acceptance thresholds match spec (MMD < 0.05, AUC < 0.65, ACF corr > 0.95)
- `run_full_vva()` single entry point — correct

### ⚠️ Gaps Found

| # | Severity | Gap | Details |
|---|----------|-----|---------|
| 10 | **HIGH** | `evaluate_tstr()` has placeholder TRTR | Line 262: `f1_trtr = f1_tstr * 1.05` — this is a fake 5% boost, not an actual Train-Real-Test-Real baseline. For publication, we need actual TRTR training and evaluation. |
| 11 | **MEDIUM** | Missing TSTR AUROC computation | Lines 270-271: `tstr_auroc` and `trtr_auroc` are hardcoded to 0.0. Should compute `roc_auc_score` from predicted probabilities. |
| 12 | **MEDIUM** | `evaluate_tstr()` not called in `run_full_vva()` | The TSTR metric is skipped in the combined entry point because it requires labeled data, but there's no documented alternative path. |
| 13 | **MEDIUM** | Propensity score reports single AUC, not mean ± std | Line 219: `auc_std: 0.0` — should compute per-fold AUCs and report std across folds as spec requires. |
| 14 | **LOW** | No `__main__` entry point for VVA | Cannot run `python -m data_gen.vva` to execute the suite on generated data. |

---

## Missing Modules (from Master Plan Section 1)

| Spec Requirement | Status |
|------------------|--------|
| `tests/test_simulator.py` | ❌ Not created |
| `tests/test_features.py` | ❌ Not created |
| `tests/test_models.py` | ❌ Not created |
| `tests/test_metrics.py` | ❌ Not created |
| `experiments/figures.py` | ✅ Exists and working |
| `eval/adversarial.py` | ✅ Exists (bonus, not in spec) |

---

## Recommended Fix Priority

### Phase 1 — Critical Fixes (blocks pipeline execution)
1. **Gap #1**: Vectorize forward-looking labels in `compute_targets()` — O(n²) → O(n)
2. **Gap #6**: Add cGAN training entry point with data loading
3. **Gap #7**: Implement augmentation strategy (3× fault oversample → 1:4 ratio)
4. **Gap #10**: Implement real TRTR baseline in VVA suite

### Phase 2 — Spec Compliance (blocks publication quality)
5. **Gap #2**: Add temperature rolling stats
6. **Gap #3**: Add RUL target computation
7. **Gap #5**: Per-scenario ambient temperature extraction
8. **Gap #8**: Add "developing" severity level to cGAN condition
9. **Gap #11/13**: Fix TSTR AUROC and propensity std computation

### Phase 3 — Polish (nice to have)
10. **Gap #4**: Wire test splits into evaluation harness
11. **Gap #9/14**: Add W&B logging to cGAN, add VVA entry point
12. Create `tests/` directory with proper pytest files
