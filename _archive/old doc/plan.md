This is going to be our complete battle plan. Everything from first line of code to journal submission. Treat this as your research bible.

***

# Project DRDO-AVR: Complete Research Execution Plan

**Working Title:** *Physics-Informed Simulation and Uncertainty-Aware Fault Prognostics for Military Vehicle Automatic Voltage Regulators Under Combat Stress Conditions*

**Target:** IEEE Transactions on Industrial Electronics (Tier 1) or IEEE Transactions on Reliability (Tier 1/2), with arXiv preprint as fallback floor.

***

## Architecture Overview

The paper has **three independent, interlocking contributions**:

```
CONTRIBUTION 1          CONTRIBUTION 2              CONTRIBUTION 3
─────────────────       ──────────────────────       ───────────────────────
Physics-Informed    →   Multi-Task ML Pipeline   →   Uncertainty-Aware
Simulation Engine       (Anomaly + RUL + Type)        Combat Risk Taxonomy
(Section 3)             (Section 4)                   (Section 5-6)

MIL-STD-1275E           LSTM-AE + TFT + XGB          Conformal Intervals
grounded models         with TimeSeriesSplit           + SHAP + Wilcoxon
```

Any one of these alone is a workshop paper. All three together is a Tier 1 submission.

***

## Phase 1: Physics-Informed Data Generator

### 1.1 Ground the Baseline in MIL-STD-1275E

The actual legally published NATO/DoD standard for 28VDC ground vehicle power  gives you these exact parameters to correct your generator with: [sullivanuv](http://sullivanuv.com/wp-content/uploads/2014/06/MIL-STD-1275D-28V-DC-Electrical-systems-in-Military-Vehicles.pdf)

| Parameter | MIL-STD-1275E Value | Your Current (Wrong) Value |
|---|---|---|
| Steady-state min | **22.0 VDC** | 23.5V |
| Steady-state max | **32.0 VDC** | 32.5V |
| Nominal | 28 VDC | 28V ✓ |
| Cold crank min | 16 VDC @ 30s | Not modeled |
| Jump start max | **48 VDC @ 1s** | Not modeled |
| Load dump spike | **+100V transient, 2ms** | Not modeled |
| Alternator ripple | ≤ 0.5V RMS | Not modeled |

These are real numbers from a real public standard. Cite them directly in Section 3.1.

### 1.2 Baseline Voltage Model — Ornstein-Uhlenbeck Process

Replace your current random noise with the OU process, which is the standard mean-reverting stochastic process for voltage regulators in literature: [ieeexplore.ieee](https://ieeexplore.ieee.org/iel7/6287639/6514899/10375385.pdf)

\[ dV_t = \theta(\mu - V_t)\,dt + \sigma\,dW_t \]

- \(\mu = 28.0\) V (nominal)
- \(\theta = 2.5\) (mean reversion speed — tune so voltage returns to nominal within ~0.4s)
- \(\sigma = 0.15\) V/√s (noise intensity — tune to produce ~0.5V RMS ripple per MIL-STD)
- \(dW_t\) = Wiener process increment = \(\mathcal{N}(0, \sqrt{dt})\)

```python
def generate_ou_voltage(n_steps, dt=0.1, mu=28.0, theta=2.5, sigma=0.15, seed=None):
    rng = np.random.default_rng(seed)
    V = np.zeros(n_steps)
    V[0] = mu
    for t in range(1, n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        V[t] = V[t-1] + theta * (mu - V[t-1]) * dt + sigma * dW
    return np.clip(V, 16.0, 48.0)  # hard physical limits per MIL-STD-1275E
```

### 1.3 Scenario Physical Models — All Six

**Scenario 1: Desert Heat**
Use Arrhenius thermal derating on internal resistance: [rfimmunity](https://rfimmunity.com/wp-content/uploads/2017/04/MIL-STD-1275D.pdf)
\[ R_{hot}(T) = R_{ref} \cdot \exp\!\left[\frac{E_a}{k_B}\left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right] \]
\( E_a = 0.7\,\text{eV} \), \( k_B = 8.617 \times 10^{-5}\,\text{eV/K} \), \( T_{ref} = 298\,\text{K} \), \( T_{desert} = 333\,\text{K} \) (60°C).
This causes voltage sag under load: \( V_{sag} = I_{load} \cdot \Delta R_{hot} \). Physically accurate, citable.

**Scenario 2: Arctic Cold**
Same Arrhenius model, negative ΔT. \( T_{arctic} = 233\,\text{K} \) (−40°C). Cold start effect — battery internal resistance rises 3–4× at −40°C. Add a cold-crank transient at \(t=0\): V drops to 16V for 0.5s then recovers per MIL-STD-1275E Curve 1. [sullivanuv](http://sullivanuv.com/wp-content/uploads/2014/06/MIL-STD-1275D-28V-DC-Electrical-systems-in-Military-Vehicles.pdf)

**Scenario 3: Artillery Firing**
Impulsive load step from high-current actuators:
\[ I_{surge}(t) = I_{peak} \cdot e^{-t/\tau} \cdot u(t) \]
\[ V_{bus}(t) = V_{OC} - I_{surge}(t) \cdot R_{int} - L \cdot \frac{dI_{surge}}{dt} \]
\( I_{peak} = 300\,\text{A} \), \( \tau = 50\,\text{ms} \), \( R_{int} = 0.025\,\Omega \), \( L = 1\,\text{mH} \). Fire events: Poisson-distributed, mean rate 0.5/min.

**Scenario 4: Rough Terrain**
Vibration-induced contact resistance oscillation:
\[ R_{contact}(t) = R_0\left(1 + A\sin(2\pi f_{vib} t + \phi)\right) \]
\( f_{vib} \sim \mathcal{U}(2, 15)\,\text{Hz} \) (vehicle suspension resonance band), \( A = 0.15 \).

**Scenario 5: Weapons Active**
MIL-STD-1275E Section 5.3 — switching transients from high-power loads. Multiple back-to-back load switch events per: [sullivanuv](http://sullivanuv.com/wp-content/uploads/2014/06/MIL-STD-1275D-28V-DC-Electrical-systems-in-Military-Vehicles.pdf)
- Load increases from 10% → 85% of rated current in 5ms
- 5 successive switch events with 1s interval
- Each event: \( V_{spike} = L \cdot \frac{\Delta I}{\Delta t} \) overshoot followed by OU mean reversion

**Scenario 6: EMP Simulation**
MIL-STD-461G RS105 double exponential pulse: [ti](https://www.ti.com/lit/pdf/tidudg1)
\[ V_{EMP}(t) = V_0 \cdot E_0 \cdot \left(e^{-\alpha t} - e^{-\beta t}\right) \]
\( \alpha = 4 \times 10^6\,\text{s}^{-1} \), \( \beta = 4.76 \times 10^8\,\text{s}^{-1} \), \( V_0 = 50{,}000\,\text{V/m} \), \( E_0 \) = coupling factor (~0.001 for shielded vehicle harness). Multiple EMP events per run; also adds wideband noise floor +2dB permanently after each event (representing residual interference).

### 1.4 New Columns You Must Add

Your generator must produce these columns that did not exist before:

```
timestamp           → clean ISO format, 10Hz, no mixed types
voltage             → OU + scenario perturbation
current             → correlated with load profile
temperature         → ambient + self-heating model
load_percent        → drives current and voltage sag
v_ripple            → high-freq component, FFT-extractable
transient_flag      → 1 during active perturbation event
event_id            → unique ID per firing/surge/EMP event
time_to_next_fault  → ← THIS IS NEW. RUL label in seconds
fault_within_5s     → binary, your existing label
fault_within_30s    → binary, new lookahead window
fault_type          → under/over/transient/none
fault_severity      → 0-3 ordinal scale
scenario            → unchanged
```

`time_to_next_fault` is computed backward from fault timestamps during generation — it's trivial to add but transforms your paper from binary classification to **Remaining Useful Life prediction**, which has a dedicated IEEE/PHM community. [arxiv](https://arxiv.org/abs/2212.14612v1)

### 1.5 Simulation Fidelity Validation (Section 3.3)

You cannot run hardware experiments, but you can validate statistically against published data:

1. Extract fault rate ranges from NATO/US Army published maintenance statistics (search: "US Army TACOM vehicle electrical fault rate" — these are public)
2. Show your simulated fault rates fall within ±20% of those ranges
3. Run a **two-sample KS test** between your simulated voltage distribution and any published oscilloscope histogram from a 28VDC system paper:
```python
from scipy.stats import ks_2samp, mannwhitneyu
stat, p = ks_2samp(simulated_voltage, reference_voltage)
# p > 0.05 means distributions are statistically consistent
```
4. Show autocorrelation structure of your OU voltage matches expected 1/f noise of real power systems

***

## Phase 2: ML Pipeline — Complete Technical Spec

### 2.1 Feature Engineering v2.0

Keep your existing lag and rolling features but add three new feature families:

**FFT Frequency Domain Features** — extracts ripple and transient signatures invisible to time-domain:
```python
from scipy.fft import rfft, rfftfreq

def extract_fft_features(window, fs=10):  # fs=10Hz sampling
    spectrum = np.abs(rfft(window))
    freqs = rfftfreq(len(window), 1/fs)
    return {
        'fft_dominant_freq': freqs[np.argmax(spectrum[1:]) + 1],
        'fft_power_2_5hz': np.sum(spectrum[(freqs >= 2) & (freqs <= 5)]),
        'fft_power_5_10hz': np.sum(spectrum[(freqs >= 5) & (freqs <= 10)]),
        'spectral_entropy': -np.sum(p * np.log(p + 1e-10) 
                             for p in spectrum/spectrum.sum())
    }
```

**Rate of Change Features:**
```python
df['dV_dt'] = df.groupby('scenario')['voltage'].diff() / dt
df['d2V_dt2'] = df.groupby('scenario')['dV_dt'].diff() / dt
df['dI_dt'] = df.groupby('scenario')['current'].diff() / dt
```

**Cross-variable Interaction Features:**
```python
df['power'] = df['voltage'] * df['current']
df['V_I_ratio'] = df['voltage'] / (df['current'] + 1e-6)
df['thermal_load'] = df['power'] * df['temperature']
df['impedance_proxy'] = df['dV_dt'] / (df['dI_dt'] + 1e-6)
```

### 2.2 Model 1: LSTM Autoencoder (Anomaly Detection)

Train **only on baseline normal data**. Flag combat faults by reconstruction error exceeding a threshold. This sidesteps class imbalance entirely: [ijaseit.insightsociety](https://ijaseit.insightsociety.org/index.php/ijaseit/article/download/20451/4479)

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2):
        super().__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                batch_first=True, dropout=0.2)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, 
                                batch_first=True, dropout=0.2)
    
    def forward(self, x):
        # Encode
        enc_out, (h, c) = self.encoder(x)
        latent = self.encoder_fc(h[-1])
        # Decode (repeat latent across sequence length)
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        return dec_out

# Training: only on baseline
# Anomaly score: MSE reconstruction error per window
# Threshold: 95th percentile of reconstruction error on validation baseline
```

**Evaluation**: Do NOT use accuracy. Report:
- AUC-ROC and AUC-PR curves
- F1 at threshold = {90th, 95th, 99th} percentile
- False Negative Rate at operating threshold
- **Mean early warning time** = mean(fault_time − alert_time) across all true positives

### 2.3 Model 2: Temporal Fusion Transformer (RUL Prediction)

TFT is the current state-of-the-art for multi-horizon time series forecasting with interpretability. Use the `pytorch-forecasting` library: [arxiv](https://arxiv.org/pdf/1912.09363.pdf)

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Dataset setup — TFT requires specific format
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="time_to_next_fault",     # RUL label
    group_ids=["scenario", "run_id"],
    min_encoder_length=60,            # 6 seconds of history
    max_encoder_length=120,           # 12 seconds max
    min_prediction_length=1,
    max_prediction_length=50,         # predict 5 seconds ahead
    static_categoricals=["scenario"],
    time_varying_known_reals=["time_idx", "load_percent"],
    time_varying_unknown_reals=["voltage", "current", "temperature",
                                  "v_ripple", "dV_dt", "power"],
    target_normalizer=GroupNormalizer(groups=["scenario"]),
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=3e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=7,                    # 7 quantiles
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
```

TFT gives you **three layers of interpretability for free**: [towardsdatascience](https://towardsdatascience.com/temporal-fusion-transformer-time-series-forecasting-with-deep-learning-complete-tutorial-d32c1e51cd91/)
- Variable importance (which features matter most per scenario)
- Attention weights (which past timesteps were most predictive)
- Quantile predictions (built-in uncertainty without conformal prediction)

### 2.4 Model 3: XGBoost + SMOTE (Fault Type Classification)

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

# SMOTE only on training fold, never on test
smote_xgb = ImbPipeline([
    ('smote', SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)),
    ('xgb', xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,      # SMOTE handles balance
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    ))
])

# Calibrate probability outputs
calibrated_model = CalibratedClassifierCV(smote_xgb, cv=5, method='isotonic')
```

### 2.5 Conformal Prediction for RUL (The Tier 1 Addition)

This gives **statistically guaranteed prediction intervals** — the exact formulation from the published framework: [epub.ub.uni-muenchen](https://epub.ub.uni-muenchen.de/107522/1/3417-Full-Length_Manuscripts-12659-1-10-20230724__1_.pdf)

```python
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

# Wrap any RUL point predictor (e.g., GBR or TFT point output)
mapie_rul = MapieRegressor(
    estimator=base_rul_model,
    method="plus",      # split conformal
    cv=10,
    random_state=42
)
mapie_rul.fit(X_calib, y_rul_calib)
y_pred, y_intervals = mapie_rul.predict(X_test, alpha=[0.05, 0.10, 0.20])

# Report coverage
coverage_90 = regression_coverage_score(y_test, y_intervals[:, 0, 1], 
                                         y_intervals[:, 1, 1])
# Should be ≥ 0.90 for alpha=0.10
```

Report in your paper: coverage scores at 80%, 90%, 95% nominal confidence. If coverage ≥ nominal level, your intervals are valid. This is a formal statistical guarantee that no other paper in your space likely has.

### 2.6 SHAP Explainability

```python
import shap

# For XGBoost fault type classifier
explainer = shap.TreeExplainer(calibrated_model.estimator.named_steps['xgb'])
shap_values = explainer.shap_values(X_test_scaled)

# Key plots for paper:
# 1. Summary plot (all features, all scenarios)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, 
                  class_names=['baseline','arctic','desert',
                               'artillery','terrain','weapons','emp'])

# 2. Per-scenario dependence plot
shap.dependence_plot('voltage_rolling_std_10', shap_values [ti](https://www.ti.com/lit/pdf/tidudg1), X_test)
# Shows: how voltage variance drives fault risk in EMP scenario specifically
```

The SHAP analysis will almost certainly show `dV_dt`, `voltage_rolling_std_10`, and `impedance_proxy` as top features — which maps directly to physical intuition (rate of change + volatility + electrical impedance = fault predictors). That interpretation goes in Section 6.1 and is what makes defence reviewers trust the model.

### 2.7 Training Protocol — Non-Negotiable

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (average_precision_score, roc_auc_score, 
                              f1_score, precision_recall_curve)

# Purged TimeSeriesSplit: gap prevents any leakage from rolling features
tscv = TimeSeriesSplit(n_splits=5, gap=200, test_size=10000)

results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    
    # SMOTE inside fold only
    model.fit(X_tr, y_tr)
    
    y_prob = model.predict_proba(X_te)[:, 1]
    results.append({
        'fold': fold,
        'auc_roc': roc_auc_score(y_te, y_prob),
        'auc_pr': average_precision_score(y_te, y_prob),
        'f1_t0.3': f1_score(y_te, y_prob > 0.3),
        'f1_t0.5': f1_score(y_te, y_prob > 0.5),
        'fnr': 1 - recall_score(y_te, y_prob > 0.3)
    })

# Report: mean ± std across 5 folds for every metric
```

***

## Phase 3: Statistical Analysis Layer

### 3.1 Scenario Significance Testing

Every fault rate comparison must be backed by a non-parametric test (voltages are non-normal):

```python
from scipy.stats import mannwhitneyu, kruskal, wilcoxon
from statsmodels.stats.multitest import multipletests

# Kruskal-Wallis first: are any scenarios different at all?
stat, p_kruskal = kruskal(*[df[df.scenario==s]['voltage'] 
                             for s in scenarios])

# Pairwise Mann-Whitney U: each scenario vs baseline
p_values = []
for scenario in combat_scenarios:
    baseline_v = df[df.scenario=='baseline']['voltage'].sample(10000)
    combat_v = df[df.scenario==scenario]['voltage']
    _, p = mannwhitneyu(baseline_v, combat_v, alternative='two-sided')
    p_values.append(p)

# Bonferroni correction for multiple comparisons
reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
```

Report Cohen's d as effect size alongside p-values. p-value alone is not enough for Tier 1.

### 3.2 Ablation Study (Section 6.4)

Every paper at this tier needs an ablation study. Test these five configurations and show metric degradation:

| Config | What's Removed | Expected AUC-PR Drop |
|---|---|---|
| Full model | Nothing | Baseline |
| No FFT features | Remove frequency domain | ~3–5% |
| No conformal intervals | Point prediction only | Coverage undefined |
| No SMOTE | Class imbalance unhandled | Large recall drop |
| LSTM-AE only (no TFT) | No RUL component | MAE undefined |
| Rule-based threshold only | No ML at all | Lowest AUC |

***

## Phase 4: Paper Writing — Section by Section

### Section 1: Introduction (Write Last)

Structure as four paragraphs:
1. **Hook**: Operational cost of electrical failure in military vehicles — cite a real incident report or NATO maintenance statistic
2. **Gap**: Existing predictive maintenance work focuses on engines/transmissions; AVR-specific ML under combat stress conditions is unstudied
3. **What we do**: One crisp paragraph stating method
4. **Contributions**: Exactly 4 bullet points, numbered, starting with "We propose..."

### Section 2: Related Work (4 Subsections, 12–15 citations)

```
2.1 Predictive Maintenance for Military Platforms
    → cite: [web:22][web:24][web:30][web:13]
    
2.2 Physics-Informed Synthetic Data for Fault Detection  
    → cite: [web:57][web:79][web:81] + your MIL-STD citations
    
2.3 Deep Anomaly Detection for Imbalanced Time Series
    → cite: LSTM-AE papers [web:110][web:116][web:119]
    
2.4 Uncertainty Quantification in Safety-Critical Prognostics
    → cite: conformal RUL papers [web:111][web:114][web:117]
```

For each subsection: 2–3 sentences per paper, ending with "however, none of these works address X" — where X is your gap.

### Section 3: Physics-Informed Simulation Framework

This is your primary novelty section. Structure:
- 3.1: MIL-STD-1275E parameter basis (table of all parameters)
- 3.2: Mathematical models for each scenario (all 6 equations as laid out in Phase 1)
- 3.3: Dataset statistics (table: scenario, records, fault rate, mean voltage, std voltage)
- 3.4: Simulation fidelity validation (KS test results, comparison to published ranges)

### Section 4: Methodology

- 4.1: Feature engineering (lag + rolling + **FFT + rate-of-change + interaction**)
- 4.2: LSTM Autoencoder architecture (diagram + training protocol)
- 4.3: TFT for RUL (architecture diagram, loss function: QuantileLoss)
- 4.4: XGBoost + SMOTE for fault type (pipeline diagram)
- 4.5: Conformal prediction wrapper (formal coverage guarantee statement)
- 4.6: TimeSeriesSplit evaluation protocol

### Section 5: Results

Table 1: Model comparison across all baselines and proposed models
Table 2: Per-scenario performance breakdown
Table 3: Conformal coverage at 80/90/95% confidence
Figure 1: Precision-Recall curves (ALL models, all scenarios — one subplot grid)
Figure 2: TFT attention weights heatmap (shows which past timesteps matter)
Figure 3: Conformal prediction intervals for RUL on EMP scenario (most dramatic)
Figure 4: SHAP summary plot

### Section 6: Analysis

- 6.1: SHAP — which features drive each scenario's fault risk
- 6.2: Early warning time analysis — histogram of (alert time − fault time) per scenario
- 6.3: Combat risk taxonomy (your existing table, now with p-values and Cohen's d added)
- 6.4: Ablation study results

### Section 7: Discussion

Three mandatory things here:
1. **Simulation limitations**: Honest statement that real hardware may show different coupling coefficients, unknown failure modes
2. **Path to real hardware**: Describe exactly what a hardware validation study would look like — what sensors, what test bench, what data format you'd need. This makes DRDO collaboration concrete.
3. **Deployment architecture**: ONNX export, edge device spec (ARM Cortex-M7 @216MHz can run the RF inference in <1ms), latency analysis

***

## Phase 5: Submission Strategy

### Step 1: arXiv First (Week 8)
Post to `cs.LG` + `eess.SP` (Signal Processing for Electrical/Systems). This:
- Establishes priority date immediately
- Creates a citable preprint for your fellowship/internship applications
- Gets you community feedback before peer review

### Step 2: Primary Submission (Week 9)
**IEEE Transactions on Industrial Electronics** — Impact Factor ~7.5, Tier 1.
- Scope explicitly includes: power electronics, fault detection, vehicle systems, ML for industrial systems
- Average review time: 3–4 months
- They regularly publish simulation-based methodology papers

### Step 3: Backup (If Rejected at Week 20)
**IEEE Access** — Open access, faster review, broad scope, still indexed and respectable. Revise based on reviewer comments, resubmit within 2 weeks. [ieeexplore.ieee](https://ieeexplore.ieee.org/abstract/document/10015520/)

### Step 4: Conference Version in Parallel (Submit Week 10)
**IEEE IECON 2026** or **IEEE ICIT 2026**  — submit a 6-page condensed version of the core ML contribution. Conference acceptance gives you something to present while the journal is under review, and builds your CV. [icit2026.ieee-ies](https://icit2026.ieee-ies.org/SS1_CFP.pdf)

***

## Full 10-Week Timeline

| Week | Deliverable | Hours/Week |
|---|---|---|
| **1** | Rewrite data generator: OU process, MIL-STD-1275E params, all 6 scenario models | 20 |
| **2** | Add `time_to_next_fault`, `event_id`, `v_ripple` columns; run generation; validate with KS test | 15 |
| **3** | Feature engineering v2.0: FFT + rate-of-change + interaction features | 10 |
| **4** | LSTM Autoencoder: build, train on baseline only, threshold selection, AUC-PR evaluation | 20 |
| **5** | TFT for RUL: dataset setup, training, quantile loss, attention weight extraction | 20 |
| **6** | XGBoost + SMOTE, conformal prediction wrapper, SHAP analysis, ablation study | 15 |
| **7** | All experiments: 5-fold TimeSeriesSplit, statistical tests, generate all figures/tables | 15 |
| **8** | Write Sections 3+4+5 (technical core) in LaTeX (use IEEE template on Overleaf) | 20 |
| **9** | Write Sections 1+2+6+7; internal review; upload to arXiv | 20 |
| **10** | Format for IEEE TIE; write cover letter; submit | 8 |

**Total: ~163 hours over 10 weeks = ~16 hours/week** — manageable alongside coursework.

***

## Tools and Libraries Locked In

```
Data Generation:    numpy, scipy.stats (OU process, distributions)
ML Core:            pytorch, pytorch-forecasting (TFT), xgboost
Imbalance:          imbalanced-learn (SMOTE)
Uncertainty:        MAPIE (conformal prediction)
Explainability:     shap
Statistics:         scipy.stats (Mann-Whitney, KS, Kruskal-Wallis)
                    statsmodels (multiple testing correction)
Visualization:      matplotlib, seaborn, plotly (interactive for dashboard)
Deployment:         onnxruntime, skl2onnx
Writing:            Overleaf (IEEE template), Zotero (citations)
Version Control:    GitHub (public repo — linked in paper for reproducibility)
```

***

## The Hardware Collaboration Pitch (After arXiv)

Once your arXiv paper is live, you have a concrete artifact to show. Your DRDO collaboration email will say:

> *"We have developed and published a physics-informed simulation framework for 28VDC military AVR fault prediction, grounded in MIL-STD-1275E, achieving [X]% AUC-PR with conformal-guaranteed RUL intervals. The framework is designed for direct calibration with real telemetry data. We would like to validate our simulation parameters and model generalization using test bench data from [Lab/Unit]."*

That is a completely fundable, credible research collaboration proposal. The arXiv paper is your proof of seriousness — no lab will say no to free ML collaboration on their hardware problem.