This is the complete master specification document

***

# MASTER RESEARCH SPECIFICATION
## Physics-Informed Digital Twin for Military AVR Fault Prognostics
### AVR-PHM: MIL-STD-Aligned Synthetic Data + PINN Early Warning System

***

## DOCUMENT PURPOSE AND USAGE INSTRUCTIONS

This document is a complete, authoritative specification for implementing a tier-1 academic research codebase. You are acting as a senior research engineer. Every architectural decision, equation, parameter range, file path, and evaluation criterion is specified here. You must not deviate from these specifications, invent alternatives, or simplify any component. If a section says "exact", implement it exactly. If a section gives a range, stay within it. Your job is precise execution, not design.

**Working Protocol:**
- Implement one Phase at a time; do not start Phase N+1 until Phase N is complete and verified
- Every function must have a docstring with: purpose, inputs, outputs, mathematical basis (if applicable)
- Every file must end with `if __name__ == "__main__": run_tests()` containing at minimum 2 sanity-check assertions
- Save intermediate outputs to disk after every major computation — assume the process may be interrupted
- Use `wandb` for all experiment tracking; log every hyperparameter, metric, and artifact
- Freeze all random seeds: `numpy.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)` at the top of every script
- Python version: 3.11. All type hints required. No `Any` types.

***

## SECTION 1: REPOSITORY STRUCTURE

Create this exact directory tree before writing a single line of logic:

```
avr_phm/
│
├── README.md
├── requirements.txt
├── setup.py
├── .env                          # WANDB_API_KEY, seeds
├── pyproject.toml
│
├── config/
│   ├── __init__.py
│   ├── scenarios.yaml            # All scenario parameters — single source of truth
│   ├── model.yaml                # All model hyperparameters
│   ├── milstd.yaml               # All MIL-STD waveform parameters
│   └── paths.yaml                # All file paths
│
├── simulator/
│   ├── __init__.py
│   ├── constants.py              # Physical constants, nominal values
│   ├── dae_model.py              # Park's d-q synchronous generator + AVR DAE
│   ├── mil_std_1275e.py          # Exact MIL-STD-1275E transient waveforms
│   ├── mil_std_810h.py           # MIL-STD-810H vibration PSD + SRS
│   ├── fault_mechanisms.py       # Component degradation physics models
│   ├── scenario_engine.py        # Assembles scenarios from config + DAE
│   └── validator.py              # Simulator output sanity checks
│
├── data_gen/
│   ├── __init__.py
│   ├── pipeline.py               # Orchestrates full dataset generation
│   ├── cgan.py                   # Conditional TimeGAN for rare fault augmentation
│   └── vva.py                    # Validation, Verification, Accreditation suite
│
├── features/
│   ├── __init__.py
│   └── engineer.py               # Feature engineering: lags, rolling, physics-residual
│
├── models/
│   ├── __init__.py
│   ├── baseline_threshold.py     # Rule-based threshold detector (Baseline 0)
│   ├── baseline_rf.py            # Random Forest + GBM (Baseline 1, from prior work)
│   ├── recurrent_ae.py           # Bi-GRU Recurrent Autoencoder (Baseline 2)
│   ├── pinn.py                   # Physics-Informed Neural Network (Proposed Method)
│   └── patchtst.py               # PatchTST Time-LLM comparison (Baseline 3)
│
├── eval/
│   ├── __init__.py
│   ├── phm_metrics.py            # All PHM-grade metrics
│   ├── calibration.py            # ECE, reliability diagrams
│   └── xai.py                    # SHAP explanations
│
├── experiments/
│   ├── __init__.py
│   ├── train.py                  # Unified training harness
│   ├── evaluate.py               # Unified evaluation harness
│   └── ablation.py               # Ablation study sweep
│
├── data/
│   ├── raw/                      # Simulator output CSVs (auto-generated)
│   ├── processed/                # Feature-engineered datasets
│   ├── synthetic/                # cGAN-generated augmentation data
│   └── splits/                   # Train/val/test split index files
│
├── outputs/
│   ├── checkpoints/              # Model checkpoints (save every N epochs)
│   ├── results/                  # Evaluation results JSON/CSV
│   ├── figures/                  # Publication-quality figures (300 DPI)
│   └── logs/                     # Training logs, wandb local backup
│
└── tests/
    ├── test_simulator.py
    ├── test_features.py
    ├── test_models.py
    └── test_metrics.py
```

***

## SECTION 2: CONFIGURATION FILES

### `config/scenarios.yaml`

```yaml
global:
  sampling_frequency_hz: 10          # 10 Hz data logging
  nominal_voltage_v: 28.0            # MIL-STD-1275E nominal 28V DC bus
  voltage_min_threshold_v: 23.5      # Under-voltage fault threshold
  voltage_max_threshold_v: 32.5      # Over-voltage fault threshold
  nominal_current_a: 45.0
  fault_log_duration_ms: 100         # Fault must persist 100ms before logging

scenarios:
  baseline:
    duration_minutes: 120
    runs: 4
    ambient_temp_c: 25.0
    temp_variation_c: 5.0
    base_fault_probability_per_sample: 0.0005
    vibration_psd_category: null
    shock_srs: null

  desert_heat:
    duration_minutes: 30
    runs: 2
    ambient_temp_c: 65.0
    temp_ramp_c_per_hour: 5.0        # Gradual heat buildup
    thermal_derating_factor: 0.85    # SOA reduction at high junction temp
    base_fault_probability_per_sample: 0.0015
    vibration_psd_category: "Category_4"  # MIL-STD-810H wheeled vehicle

  arctic_cold:
    duration_minutes: 30
    runs: 2
    ambient_temp_c: -40.0
    battery_resistance_multiplier: 3.2  # R_internal increases at cold
    ies_events_per_hour: 2           # Initial Engagement Surges
    base_fault_probability_per_sample: 0.001
    vibration_psd_category: "Category_4"

  artillery_firing:
    duration_minutes: 30
    runs: 2
    ambient_temp_c: 45.0
    firing_events_per_hour: 6        # Artillery discharge events
    ballistic_shock_g_peak: 40       # SRS peak per MIL-STD-810H Method 522.2
    base_fault_probability_per_sample: 0.002
    vibration_psd_category: "Category_20"  # Tracked vehicle off-road

  rough_terrain:
    duration_minutes: 30
    runs: 2
    ambient_temp_c: 35.0
    terrain_class: "cross_country"
    vibration_psd_category: "Category_20"
    base_fault_probability_per_sample: 0.0018

  weapons_active:
    duration_minutes: 30
    runs: 2
    ambient_temp_c: 50.0
    load_step_amps: 80.0             # Step increase when weapons engage
    load_dump_events_per_hour: 4     # MIL-STD-1275E load dump events
    base_fault_probability_per_sample: 0.0025

  emp_simulation:
    duration_minutes: 30
    runs: 2
    ambient_temp_c: 45.0
    emp_events: 1                    # Single EMP event mid-mission
    emp_voltage_spike_v: 250.0       # Per MIL-STD-461
    emp_recovery_oscillation_std: 15.0
    base_fault_probability_per_sample: 0.005
```

### `config/milstd.yaml`

```yaml
mil_std_1275e:
  nominal_voltage_v: 28.0
  
  # Starting transients
  initial_engagement_surge:
    voltage_drop_v: 6.0              # Minimum drop during IES
    duration_s: 0.5                  # Up to 1.0s per standard
    rise_time_ms: 10.0
    recovery_time_ms: 200.0
  
  cranking:
    voltage_v: 16.0
    max_duration_s: 30.0
  
  # Spike (high voltage, short duration, low energy)
  spike:
    peak_voltage_v: 250.0
    duration_us: 70.0                # 70 microseconds
    rise_time_us: 1.0                # <1 microsecond rise
    energy_joules: 2.0
  
  # Load dump surge (lower voltage, long duration, high energy)
  load_dump:
    peak_voltage_v: 100.0
    source_impedance_ohm: 0.5
    duration_ms: 50.0
    repetitions: 3
    interval_s: 1.0
  
  # Normal operating range (NOT faults)
  steady_state_ripple_peak_to_peak_v: 1.5    # Max ripple under normal operation
  transient_spike_normal_max_v: 32.5         # Normal spikes not classified as faults

mil_std_810h:
  # Category 4: Wheeled vehicles on roads (Method 514.8)
  Category_4:
    breakpoints_hz: [10, 40, 500]
    psd_g2_per_hz: [0.04, 0.04, 0.01]
    duration_minutes: 60
  
  # Category 20: Wheeled vehicles off-road / Tracked vehicles
  Category_20:
    breakpoints_hz: [5, 20, 150, 500]
    psd_g2_per_hz: [0.015, 0.015, 0.003, 0.001]
    duration_minutes: 60
  
  # Method 522.2: Ballistic shock SRS
  ballistic_shock:
    natural_frequency_hz: [100, 1000, 2000]
    srs_acceleration_g: [20, 40, 30]
    Q_factor: 10
    duration_ms: 11.0
```

### `config/model.yaml`

```yaml
global:
  random_seed: 42
  device: "cuda"                     # Falls back to cpu automatically
  mixed_precision: true
  gradient_clipping: 1.0

data:
  window_size_samples: 100           # 10 seconds at 10Hz
  stride_samples: 10                 # 1 second stride
  fault_warning_horizons_s: [1, 5, 10, 30]  # Multi-horizon targets
  voltage_forecast_steps: 10         # 1 second ahead

pinn:
  architecture: "FNN"
  layer_sizes: [7, 128, 128, 128, 64, 3]   # 7 state inputs → 3 outputs
  activation: "tanh"
  initializer: "Glorot_normal"
  physics_loss_weight: 0.3           # lambda_2 in total loss
  fault_loss_weight: 0.2             # lambda_3 in total loss
  data_loss_weight: 0.5              # lambda_1 in total loss
  learning_rate: 1.0e-4
  batch_size: 256
  max_epochs: 2000
  patience: 100                      # Early stopping
  checkpoint_every_n_epochs: 50

recurrent_ae:
  encoder_hidden: [128, 64]
  decoder_hidden: [64, 128]
  rnn_type: "BiGRU"
  latent_dim: 32
  dropout: 0.2
  learning_rate: 1.0e-3
  batch_size: 128
  max_epochs: 500
  reconstruction_threshold_sigma: 2.5  # Alert if reconstruction error > mean + 2.5*std

baseline_rf:
  n_estimators: 500
  max_depth: 20
  class_weight: "balanced"
  n_jobs: -1                         # Use all CPU cores
  min_samples_leaf: 5

cgan:
  generator_hidden: [64, 128, 64]
  discriminator_hidden: [64, 128, 64]
  rnn_type: "GRU"
  latent_dim: 32
  sequence_length: 100
  learning_rate_g: 1.0e-4
  learning_rate_d: 2.0e-4
  batch_size: 32
  max_epochs: 300
  n_critic: 5                        # Train discriminator 5x per generator step (WGAN)
  gradient_penalty_weight: 10.0      # WGAN-GP lambda
```

***

## SECTION 3: PHYSICS SIMULATOR — EXACT SPECIFICATIONS

### `simulator/constants.py`

```python
"""
Physical constants and nominal parameters for the military AVR system.
All values are traceable to MIL-STD-1275E, IEEE std for synchronous generators,
and standard brushless excitation system parameters from literature.
"""

# ─── Generator Electrical Parameters ────────────────────────────────────────
Ra   = 0.0031   # Armature resistance (pu) — stator winding
Xd   = 1.81     # d-axis synchronous reactance (pu)
Xq   = 1.76     # q-axis synchronous reactance (pu)
Xd_prime  = 0.30   # d-axis transient reactance (pu)
Xq_prime  = 0.65   # q-axis transient reactance (pu)
Xd_dprime = 0.23   # d-axis subtransient reactance (pu)
Xq_dprime = 0.25   # q-axis subtransient reactance (pu)

# ─── Time Constants ──────────────────────────────────────────────────────────
Td0_prime  = 5.14   # d-axis open-circuit transient time constant (s)
Tq0_prime  = 1.50   # q-axis open-circuit transient time constant (s)
Td0_dprime = 0.033  # d-axis open-circuit subtransient time constant (s)
Tq0_dprime = 0.05   # q-axis open-circuit subtransient time constant (s)

# ─── Inertia & Damping ────────────────────────────────────────────────────────
H     = 3.0     # Inertia constant (s) — diesel-driven generator
D     = 5.0     # Damping coefficient (pu)
omega_s = 2 * 3.14159265 * 50  # Synchronous speed (rad/s) for 50Hz system

# ─── AVR Parameters (IEEE Type I Excitation System) ──────────────────────────
Ka    = 200.0   # Amplifier gain
Ta    = 0.025   # Amplifier time constant (s)
Ke    = 1.0     # Exciter constant
Te    = 0.1     # Exciter time constant (s)
Kf    = 0.04    # Stabilizer gain
Tf    = 1.0     # Stabilizer time constant (s)
Vref  = 1.05    # Reference voltage setpoint (pu), maps to 28V at nominal load

# ─── System Base Values (for pu conversion) ───────────────────────────────────
V_base  = 28.0   # Volts DC (MIL-STD-1275E nominal)
I_base  = 45.0   # Amps (nominal load current)
P_base  = V_base * I_base  # Watts

# ─── Thermal Parameters ───────────────────────────────────────────────────────
T_junction_max_c = 150.0  # Maximum junction temperature for switching elements
T_ambient_nominal_c = 25.0
thermal_resistance_junction_ambient = 0.8  # °C/W (typical for thyristor in AVR)

# ─── Degradation Parameters ───────────────────────────────────────────────────
# Capacitor aging (Arrhenius model)
C_nominal_uf     = 100.0   # Nominal EMI filter capacitance
E_activation_ev  = 0.94    # Activation energy for electrolytic capacitor
k_boltzmann_ev   = 8.617e-5  # Boltzmann constant in eV/K
T_reference_k    = 358.15  # Reference temperature 85°C in Kelvin (datasheet base)
L50_hours        = 2000    # Median life at reference temperature (hours)

# Thyristor thermal fatigue
Vf_nominal_v     = 1.5     # Nominal forward voltage drop
Vf_degraded_v    = 2.5     # Degraded forward voltage drop (EoL threshold)
thermal_cycles_to_failure = 50000  # N_f in Coffin-Manson model

# Wiring terminal loosening (Miner's Rule)
vibration_fatigue_exponent = 3.5   # b in S-N curve for terminal connectors
```

### `simulator/dae_model.py` — Complete Specification

Implement the **8th-order synchronous machine model** with **IEEE Type I AVR** as a system of Differential-Algebraic Equations solved using `scipy.integrate.solve_ivp` with method `'Radau'` (stiff solver, required for DAE systems with fast electrical + slow mechanical timescales).

**State vector:**
```
x = [delta, omega, Eq_prime, Ed_prime, Eq_dprime, Ed_dprime, Vf, Vr]
```
Where:
- `delta` = rotor angle (rad)
- `omega` = rotor angular velocity deviation (pu)
- `Eq_prime` = q-axis transient EMF (pu)
- `Ed_prime` = d-axis transient EMF (pu)
- `Eq_dprime` = q-axis subtransient EMF (pu)
- `Ed_dprime` = d-axis subtransient EMF (pu)
- `Vf` = field voltage from exciter (pu)
- `Vr` = voltage regulator state (pu)

**The differential equations** (implement exactly):

```
d(delta)/dt = omega_s * omega

d(omega)/dt = (1 / (2*H)) * (Tm - Te - D*omega)
  where Te = Ed_dprime*Id + Eq_dprime*Iq

d(Eq_prime)/dt = (1/Td0_prime) * (-Eq_prime + (Xd - Xd_prime)*Id + Vf)

d(Ed_prime)/dt = (1/Tq0_prime) * (-Ed_prime - (Xq - Xq_prime)*Iq)

d(Eq_dprime)/dt = (1/Td0_dprime) * (-Eq_dprime + Eq_prime - (Xd_prime - Xd_dprime)*Id)

d(Ed_dprime)/dt = (1/Tq0_dprime) * (-Ed_dprime + Ed_prime + (Xq_prime - Xq_dprime)*Iq)

# IEEE Type I AVR:
d(Vf)/dt = (1/Te) * (-Ke*Vf + Vr)

d(Vr)/dt = (1/Ta) * (-Vr + Ka * (Vref - Vt - Kf/Tf * Vf))
  where Vt = sqrt(Vd^2 + Vq^2)  (terminal voltage magnitude)

# Algebraic equations (stator):
Vd = -Ra*Id - Xq_dprime*Iq + Ed_dprime
Vq = -Ra*Iq + Xd_dprime*Id + Eq_dprime

# d-q currents from load (resistive-inductive load model):
Id = (Vd * R_load + Vq * X_load) / (R_load^2 + X_load^2)
Iq = (Vq * R_load - Vd * X_load) / (R_load^2 + X_load^2)

# Terminal voltage (convert to DC bus for 28V monitoring):
Vt_pu = sqrt(Vd^2 + Vq^2)
V_dc_volts = Vt_pu * V_base  # This is the monitored output
```

**The function signature:**
```python
def simulate_avr_mission(
    duration_s: float,
    scenario_params: dict,
    fault_schedule: list[dict],      # List of {time_s, mechanism, severity}
    dt: float = 0.1,                 # 0.1s timestep → 10Hz output
    initial_conditions: np.ndarray | None = None,
    degradation_state: dict | None = None,  # For progressive degradation runs
    save_path: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns: (avr_timeseries_df, fault_log_df)
    avr_timeseries_df columns: timestamp, voltage_v, current_a, 
                                temperature_c, delta, omega, Eq_prime, 
                                Ed_prime, Eq_dprime, Ed_dprime, Vf, Vr, 
                                scenario, run_id
    fault_log_df columns: timestamp, fault_type, fault_mechanism, 
                           severity, duration_ms, status
    """
```

**The load model** (R_load, X_load are time-varying):
- Baseline: R_load=0.62Ω, X_load=0.15Ω (constant 45A at 28V)
- Weapons active: step change R_load → 0.35Ω at weapon engagement time
- Arctic cold: R_load starts at 0.85Ω (reduced battery SoC), ramps down as engine warms
- All load changes use a 50ms ramp (not instantaneous) to model inductive load dynamics

### `simulator/mil_std_1275e.py` — Exact Waveform Functions

Implement each function as a callable that returns a voltage perturbation `dV(t)` to be added to the DAE output at each timestep:

```python
def ies_waveform(t: float, event_start_s: float) -> float:
    """
    Initial Engagement Surge per MIL-STD-1275E.
    Voltage drops from 28V to ~6-12V during engine start.
    Rise time (drop): 10ms. Hold: 500ms. Recovery: 200ms.
    Returns: dV perturbation (negative = voltage drop)
    """

def cranking_depression(t: float, event_start_s: float, 
                         duration_s: float = 20.0) -> float:
    """
    Cranking voltage depression: holds at ~16V for up to 30s.
    NOT a fault — predictive model must NOT alarm during this state.
    Returns: dV perturbation
    """

def spike_waveform(t: float, event_time_s: float) -> float:
    """
    MIL-STD-1275E spike: 250V peak, 70µs duration, <1µs rise time.
    Modeled as double-exponential: V(t) = V_peak*(exp(-t/τ1) - exp(-t/τ2))
    τ1 = 50µs (fall), τ2 = 0.5µs (rise). Energy ≤ 2J.
    Returns: dV perturbation (positive spike)
    """

def load_dump_waveform(t: float, event_start_s: float, 
                        repetition: int = 0) -> float:
    """
    Load dump surge: 100V peak from 0.5Ω source, 50ms duration.
    Waveform: exponential rise (2ms) then exponential decay (48ms).
    Repeat 3 times at 1s intervals.
    Returns: dV perturbation
    """
```

**Critical implementation note:** The simulation code must track whether a voltage exceedance is caused by a MIL-STD-1275E *normal operational transient* (IES, cranking, known spike) or a *genuine fault*. The fault log must include a `is_operational_transient: bool` field. The predictive model must be evaluated on its ability to distinguish these — a missed fault and a false alarm on a normal transient are equally penalized.

### `simulator/mil_std_810h.py` — PSD Vibration Generator

```python
def generate_vibration_psd(
    duration_s: float,
    sampling_rate_hz: float,
    category: str,                  # "Category_4" or "Category_20"
    seed: int = 42
) -> np.ndarray:
    """
    Generates a time-domain vibration acceleration signal (g) whose 
    power spectral density matches MIL-STD-810H Method 514.8.
    
    Method: Inverse FFT of shaped spectrum with random phase.
    
    Steps:
    1. Build frequency array from 0 to Nyquist
    2. Interpolate PSD breakpoints (log-log) to get target PSD at each bin
    3. Compute amplitude spectrum: A(f) = sqrt(PSD(f) * df)
    4. Assign random phases uniformly in [0, 2π]
    5. Build complex spectrum and apply IFFT
    6. Normalize to correct RMS value
    
    Returns: acceleration_g array of shape (n_samples,)
    """

def vibration_to_voltage_ripple(
    acceleration_g: np.ndarray,
    air_gap_sensitivity: float = 0.02  # V per g — generator air-gap coupling
) -> np.ndarray:
    """
    Converts mechanical vibration (g) to voltage ripple perturbation (V).
    Models the physical coupling: vibration → air-gap modulation → flux variation → EMF ripple.
    air_gap_sensitivity = 0.02 V/g is a conservative estimate for military brushless generators.
    Returns: dV_ripple array
    """

def generate_ballistic_shock_srs(
    peak_acceleration_g: float,
    duration_ms: float = 11.0,
    sampling_rate_hz: float = 10000.0
) -> np.ndarray:
    """
    Generates a ballistic shock pulse consistent with MIL-STD-810H Method 522.2.
    Modeled as a half-sine pulse: a(t) = A_peak * sin(π*t/T) for 0 ≤ t ≤ T.
    Returns: time-domain acceleration array
    """
```

### `simulator/fault_mechanisms.py` — Component Degradation Models

Implement exactly **three** degradation mechanisms as time-evolving parameter drift in the DAE:

**Mechanism 1: Thyristor Thermal Fatigue**
```python
class ThyristorThermalFatigue:
    """
    Models progressive increase in AVR thyristor forward voltage drop Vf
    due to thermal cycling fatigue (Coffin-Manson model).
    
    Physics: ΔVf(t) = Vf_nominal * (1 + α_Vf * N(t) / N_failure)
    where N(t) = cumulative thermal cycles at time t
          N_failure = thermal_cycles_to_failure from constants.py
          α_Vf = 0.67 (Vf increases by 67% at failure)
    
    Effect on simulation: 
        Modified Ke parameter in AVR: Ke_eff(t) = Ke * (1 - 0.3 * degradation_level)
        This causes progressively weaker exciter response to load steps.
    
    Observable signature: 
        Slower voltage recovery after load steps (recovery time increases)
        Increased steady-state voltage error under heavy load
    """
    
    def get_degradation_level(self, cumulative_thermal_cycles: float) -> float:
        """Returns degradation level in [0, 1]. 1.0 = component failure."""
    
    def modify_avr_parameter(self, base_ke: float, degradation_level: float) -> float:
        """Returns modified Ke for current degradation state."""
    
    def should_log_fault(self, voltage_error_pu: float, 
                          duration_samples: int, threshold_ms: float = 100.0) -> bool:
        """Returns True if voltage error has persisted > threshold_ms."""
```

**Mechanism 2: Electrolytic Capacitor Degradation (EMI Filter)**
```python
class CapacitorDegradation:
    """
    Models reduction in EMI filter capacitance C using Arrhenius lifetime model.
    
    Physics: L(T) = L50 * exp(E_a/k_B * (1/T - 1/T_ref))
    As C decreases, high-frequency ripple amplitude INCREASES.
    
    Ripple increase model:
        Ripple_amplitude(t) = Ripple_nominal * (C_nominal / C_effective(t))
        C_effective(t) = C_nominal * (1 - 0.6 * degradation_level)
    
    Effect on simulation:
        Additional high-frequency noise term added to voltage output:
        V_ripple_hz = f_ripple * sin(2π * f_sw * t) where f_sw = 10kHz (switching freq)
        f_ripple amplitude scales inversely with C_effective
    
    Observable signature:
        Increased voltage ripple amplitude (detectable in voltage_rolling_std features)
        Ripple frequency fixed at switching frequency
    """
```

**Mechanism 3: Voltage Sensing Terminal Loosening (Vibration-Induced)**
```python
class TerminalLoosening:
    """
    Models increase in voltage sensing measurement noise variance due to
    mechanical loosening of sensing terminal from vibration.
    Uses Miner's cumulative damage rule.
    
    Physics: D_miner(t) = Σ (n_i / N_fi) where n_i = cycles at stress amplitude i
    When D_miner ≥ 1.0: terminal effectively loose → high noise variance
    
    Noise model:
        σ_sensor(t) = σ_nominal * (1 + 9 * D_miner(t))
        σ_nominal = 0.05V (clean sensor noise floor)
    
    Effect on simulation:
        AVR feedback uses noisy voltage measurement: 
        V_measured = V_actual + N(0, σ_sensor)
        This causes oscillatory AVR response → voltage hunting
    
    Observable signature:
        High-frequency voltage oscillations at 2-5 Hz
        Increases with vibration severity — worst in artillery_firing and rough_terrain
    """
```

***

## SECTION 4: DATA GENERATION PIPELINE

### `data_gen/pipeline.py` — Complete Generation Logic

```python
def generate_full_dataset(config_path: str = "config/scenarios.yaml",
                          output_dir: str = "data/raw/",
                          resume: bool = True) -> None:
    """
    Orchestrates generation of all scenarios and runs.
    
    CRITICAL: resume=True means check which files already exist and skip them.
    This allows interrupted generation to be resumed without restarting.
    
    File naming convention:
        avr_data_{scenario}_{run_id}_{timestamp}.csv
        fault_log_{scenario}_{run_id}_{timestamp}.csv
    
    Generation order (baseline first, then ascending fault complexity):
    1. baseline (4 × 120 min runs)
    2. arctic_cold (2 × 30 min runs)
    3. rough_terrain (2 × 30 min runs)
    4. desert_heat (2 × 30 min runs)
    5. artillery_firing (2 × 30 min runs)
    6. weapons_active (2 × 30 min runs)
    7. emp_simulation (2 × 30 min runs)
    
    After each run: save to disk immediately. Log stats to wandb.
    Total expected dataset size: ~350,000 samples across all scenarios.
    """
```

**Progressive degradation runs:** For 2 of the 4 baseline runs and 1 run from each combat scenario, simulate a **full degradation trajectory** by starting with a clean system (degradation_level=0.0) and advancing the degradation state throughout the run. This creates realistic run-to-failure traces, not just healthy + instant-fault injections.

***

## SECTION 5: CONDITIONAL GAN FOR RARE FAULT AUGMENTATION

### `data_gen/cgan.py` — WGAN-GP Architecture

Use **Wasserstein GAN with Gradient Penalty (WGAN-GP)** because it is more stable than vanilla GAN and produces better time-series quality than DCGAN.

**Generator architecture:**
```
Input: z ~ N(0,1) of shape (batch, latent_dim=32) + condition vector (one-hot scenario + fault_mechanism + severity_level)
→ Dense(128) + LayerNorm + ReLU
→ Repeat: (batch, seq_len=100, hidden=128)
→ GRU(hidden=128, num_layers=2, bidirectional=False)
→ TimeDistributed Dense(3)  # voltage, current, temperature
→ Output: (batch, seq_len=100, n_features=3)
```

**Discriminator (Critic) architecture:**
```
Input: real/fake sequence (batch, seq_len=100, 3) + condition embedding
→ TimeDistributed Dense(64)
→ GRU(hidden=128, num_layers=2, bidirectional=True)
→ Flatten final hidden state
→ Concatenate with condition vector
→ Dense(64) + LeakyReLU(0.2)
→ Dense(1)  # No sigmoid — WGAN outputs unbounded score
```

**Training loop (WGAN-GP exact):**
```python
for epoch in range(max_epochs):
    for batch in dataloader:
        # Train critic n_critic=5 times per generator step
        for _ in range(n_critic):
            real = batch['sequence']
            condition = batch['condition']
            z = torch.randn(batch_size, latent_dim)
            fake = generator(z, condition)
            
            eps = torch.rand(batch_size, 1, 1)
            interpolated = eps * real + (1 - eps) * fake
            interpolated.requires_grad_(True)
            
            gp = gradient_penalty(critic, interpolated, condition)
            critic_loss = (critic(fake, condition).mean() 
                          - critic(real, condition).mean() 
                          + gradient_penalty_weight * gp)
            
        # Train generator once
        z = torch.randn(batch_size, latent_dim)
        fake = generator(z, condition)
        gen_loss = -critic(fake, condition).mean()
```

**Condition encoding:**
- Scenario: 7-dim one-hot (baseline, arctic, desert, artillery, terrain, weapons, emp)
- Fault mechanism: 4-dim one-hot (none, thyristor, capacitor, terminal)
- Severity: 3-dim one-hot (healthy, incipient, critical)
- Total condition vector: 14-dim

**Augmentation strategy:** After training, generate augmentation data specifically for:
- Incipient fault (early degradation) samples — these are rarest in simulator data
- Combined scenario stress (e.g., artillery + desert heat) — not in original scenario set
- Ratio: generate 3× as many fault samples as healthy samples, then down-sample to 1:4 ratio for final training set

***

## SECTION 6: VVA SUITE — MANDATORY FOR PUBLICATION

### `data_gen/vva.py` — All Four Metrics

**Metric 1: Maximum Mean Discrepancy (MMD)**
```python
def compute_mmd(
    real_sequences: np.ndarray,    # Shape: (n_real, seq_len, n_features)
    synthetic_sequences: np.ndarray,  # Shape: (n_synth, seq_len, n_features)
    kernel: str = "rbf",
    sigma: float = 1.0             # RBF bandwidth
) -> float:
    """
    Computes MMD between real (simulator) and synthetic (cGAN) distributions.
    
    Formula: MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    where k(x,y) = exp(-||x-y||²/(2σ²)) is the RBF kernel
    
    Implementation: flatten sequences to (n, seq_len*n_features), compute kernel matrices.
    
    Acceptance threshold: MMD < 0.05 (sequences are statistically similar)
    Publication requirement: Report mean ± std over 5 random subsamples of size 500.
    """

def compute_mmd_multikernel(real, synthetic) -> dict:
    """
    Computes MMD with multiple kernels: σ ∈ {0.1, 0.5, 1.0, 5.0, 10.0}.
    Returns dict with per-kernel MMD and median heuristic bandwidth result.
    """
```

**Metric 2: Propensity Score Matching**
```python
def compute_propensity_score(
    real_sequences: np.ndarray,
    synthetic_sequences: np.ndarray,
    n_splits: int = 5
) -> dict:
    """
    Trains a logistic regression to distinguish real vs synthetic.
    
    Pipeline:
    1. Flatten sequences to feature vectors
    2. Extract: mean, std, min, max, autocorr(lag=1), autocorr(lag=5) per channel
    3. Label: real=1, synthetic=0
    4. 5-fold cross-validated logistic regression
    5. Compute AUC-ROC
    
    Target: AUC ≈ 0.5 (classifier guesses randomly → data indistinguishable)
    Warning if AUC > 0.65: synthetic quality too low for publication
    Critical failure if AUC > 0.75: do not proceed with this cGAN checkpoint
    
    Returns: {'auc_mean': float, 'auc_std': float, 'confusion_matrix': array}
    """
```

**Metric 3: Train on Synthetic, Test on Real (TSTR)**
```python
def evaluate_tstr(
    synthetic_train: np.ndarray,
    synthetic_labels: np.ndarray,
    real_test: np.ndarray,
    real_test_labels: np.ndarray,
    model_class,                    # Pass the PINN class
    model_config: dict
) -> dict:
    """
    Trains model_class exclusively on synthetic data.
    Evaluates on real (simulator) held-out test set.
    Compares to TRTR (Train Real, Test Real) baseline.
    
    Publication requirement: TSTR F1 ≥ 0.90 × TRTR F1
    If TSTR F1 < 0.85 × TRTR F1: synthetic data is insufficient, 
    increase cGAN epochs or re-tune.
    
    Returns: {'tstr_f1': float, 'trtr_f1': float, 'tstr_trtr_ratio': float,
              'tstr_auroc': float, 'trtr_auroc': float}
    """
```

**Metric 4: Autocorrelation Analysis**
```python
def compute_autocorrelation_similarity(
    real_sequences: np.ndarray,
    synthetic_sequences: np.ndarray,
    max_lag: int = 50
) -> dict:
    """
    Computes ACF similarity between real and synthetic at lags [1, 5, 10, 20, 50].
    Uses Pearson correlation between ACF vectors as similarity score.
    
    Special check: verify that MIL-STD-1275E spike rise-time structure (70µs → 7 samples 
    at 10Hz does not resolve fully, but 1kHz subsampling shows the waveform shape) 
    is preserved in synthetic load_dump sequences.
    
    Acceptance: Pearson correlation of ACF vectors > 0.95 across all lags.
    """
```

***

## SECTION 7: FEATURE ENGINEERING

### `features/engineer.py`

**All features to compute per window (applied to windowed sequences):**

```python
FEATURE_SPEC = {
    # Raw lags (per channel: V, I, T)
    "lags": {
        "voltage": [1, 2, 3, 5, 10, 20],
        "current": [1, 2, 3, 5, 10],
        "temperature": [1, 5, 10]
    },
    
    # Rolling statistics (per channel)
    "rolling": {
        "windows": [5, 10, 20, 50],
        "stats": ["mean", "std", "min", "max", "skew"]
    },
    
    # Physics-derived features (most important for PINN inputs)
    "physics": {
        "voltage_rate_of_change": "dV/dt using finite difference",
        "current_rate_of_change": "dI/dt",
        "power_instantaneous": "V(t) * I(t)",
        "power_rate_of_change": "d(P)/dt",
        "voltage_deviation_from_nominal": "V(t) - 28.0",
        "voltage_within_spec": "indicator: 1 if 23.5 ≤ V ≤ 32.5 else 0",
        "load_impedance_estimate": "V(t) / I(t)",
        "thermal_stress_index": "(T(t) - T_ambient) / (T_max - T_ambient)",
        "voltage_ripple_amplitude": "rolling_std over 5-sample window * 2√2"
    },
    
    # Scenario one-hot encoding
    "scenario_encoding": "7-dim one-hot"
}
```

**Target variables:**
```python
TARGETS = {
    # Multi-horizon fault warning (binary per horizon)
    "fault_1s":  "bool: any fault in next 10 samples (1s)",
    "fault_5s":  "bool: any fault in next 50 samples (5s)",
    "fault_10s": "bool: any fault in next 100 samples (10s)",
    "fault_30s": "bool: any fault in next 300 samples (30s)",
    
    # Fault mechanism (multi-class)
    "fault_mechanism": "int: 0=none, 1=thyristor, 2=capacitor, 3=terminal",
    
    # Voltage trajectory forecast
    "voltage_next_10_steps": "array of shape (10,): V(t+1) ... V(t+10)",
    
    # Severity level
    "severity": "int: 0=healthy, 1=incipient, 2=developing, 3=critical"
}
```

**Data splitting (time-aware, non-random):**
```python
def create_time_aware_splits(
    df: pd.DataFrame,
    test_scenarios: list[str] = ["emp_simulation"],  # Held-out for final test
    val_fraction: float = 0.15
) -> dict:
    """
    CRITICAL: Do NOT use random shuffle. Time-series must never leak future into past.
    
    Split strategy:
    1. Hold out emp_simulation entirely for test set (scenario-held-out test)
    2. For remaining scenarios: use last 15% of each run's time as validation
    3. Everything before val split = training
    4. Second test set: hold out desert_heat + artillery_firing combined (stress combo)
    
    Returns indices for: train, val, test_held_out_scenario, test_stress_combo
    Log class balance statistics for each split to wandb.
    """
```

***

## SECTION 8: PINN IMPLEMENTATION — PRIMARY CONTRIBUTION

### `models/pinn.py` — Complete Specification

The PINN must be implemented using **DeepXDE** with **PyTorch backend**.

```python
import deepxde as dde
import torch
import torch.nn as nn
from simulator.constants import *

class AVRPhysicsResidual:
    """
    Computes the physics residual loss for the AVR PINN.
    
    The residual is the L2 norm of how much the neural network's
    predictions violate the d-q axis generator equations.
    
    We only use the REDUCED-ORDER physics constraint (not full 8th order DAE)
    to keep training tractable on edge hardware:
    
    Constraint 1 (voltage dynamics):
        ε_1 = dVt/dt - f_voltage(Eq_dprime, Ed_dprime, Id, Iq, Ra)
        
    Constraint 2 (excitation dynamics):
        ε_2 = dVf/dt - (1/Te) * (-Ke * Vf + Ka * (Vref - Vt))
        
    Constraint 3 (power conservation):
        ε_3 = P_electrical - V_dc * I_dc  (should be ~0)
        
    These three constraints are the most observable from external sensors
    (V, I, T) without requiring internal state measurements.
    """
    
    def compute_residuals(
        self, 
        t: torch.Tensor,
        V_pred: torch.Tensor,
        I_pred: torch.Tensor,
        model_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Uses torch.autograd.grad to compute dV/dt from model predictions.
        Returns residual tensor of shape (batch, 3).
        """
```

**Multi-task PINN network:**
```python
class AVRPINN(nn.Module):
    """
    Input: sliding window of shape (batch, window_size=100, n_features=3+physics_features)
    
    Architecture:
        Temporal encoder: 1D-CNN (kernel=5, filters=64, 3 layers) + Global Average Pooling
        Shared representation: Dense(128) + LayerNorm + GELU
        
    Task heads (separate for each task):
        Head 1 (fault_1s):  Dense(64) → Dense(1) + Sigmoid
        Head 2 (fault_5s):  Dense(64) → Dense(1) + Sigmoid
        Head 3 (fault_10s): Dense(64) → Dense(1) + Sigmoid
        Head 4 (fault_30s): Dense(64) → Dense(1) + Sigmoid
        Head 5 (mechanism): Dense(64) → Dense(4) + Softmax
        Head 6 (forecast):  Dense(64) → Dense(10)  # voltage trajectory, no activation
        Head 7 (severity):  Dense(64) → Dense(4) + Softmax
    
    Total parameters: ~180,000 (edge-deployable target)
    """
```

**Loss function (implement exactly):**
```python
def compute_total_loss(
    predictions: dict,
    targets: dict,
    physics_residuals: torch.Tensor,
    fault_weights: dict = {
        "fault_1s": 3.0,   # Highest weight — most urgent
        "fault_5s": 2.5,
        "fault_10s": 2.0,
        "fault_30s": 1.5
    },
    lambda_physics: float = 0.3,
    lambda_data: float = 0.5,
    lambda_fault: float = 0.2
) -> dict:
    """
    L_total = λ₁ * L_data + λ₂ * L_physics + λ₃ * L_fault
    
    L_data: MSE on voltage forecast (Head 6)
    
    L_physics: mean(ε₁² + ε₂² + ε₃²) / 3  (physics residuals from AVRPhysicsResidual)
    
    L_fault: weighted focal loss on all 4 horizon heads
        Focal loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        γ = 2.0 (standard focal loss)
        α = class_frequency_inverse_weighted
    
    Returns: {'total': tensor, 'data': tensor, 'physics': tensor, 
              'fault': tensor, 'mechanism': tensor, 'severity': tensor}
    """
```

***

## SECTION 9: BASELINE MODELS

### Baseline 0: Threshold Detector (`baseline_threshold.py`)
```python
class ThresholdDetector:
    """
    Rule-based detector. Alarms if:
    1. V < 23.5V for > 100ms
    2. V > 32.5V for > 100ms
    3. dV/dt < -5 V/s (rapid drop)
    4. rolling_std(V, 5samples) > 2.0V (high ripple)
    
    This is Baseline 0 — the simplest possible approach.
    Should be beaten by all other methods on AUROC/AUPRC.
    """
```

### Baseline 1: Random Forest + GBM (`baseline_rf.py`)
```python
"""
Exact reproduction of prior work from internship notebook.
Uses same features (lag_1, lag_2, lag_3, lag_5, lag_10, 
rolling_mean_5, rolling_mean_10, rolling_std_5, rolling_std_10).
RandomForestClassifier with class_weight='balanced'.
GradientBoostingRegressor for voltage forecast.
This is the starting point — must be shown to be worse than PINN on key metrics.
"""
```

### Baseline 2: Recurrent Autoencoder (`recurrent_ae.py`)
```python
"""
Bi-GRU Recurrent Autoencoder for unsupervised anomaly detection.
Trained ONLY on healthy (fault-free) baseline data.
At inference: alert if reconstruction error > mean + 2.5*std of training errors.

Architecture:
    Encoder: Bi-GRU(128) → Dense(64) → Dense(latent=32)
    Decoder: Dense(32) → Dense(64) → GRU(128) → TimeDistributed Dense(3)

Advantage over RF: does not require labeled fault data during training.
This demonstrates the unsupervised baseline performance.
"""
```

### Baseline 3: PatchTST (`patchtst.py`)
```python
"""
Implementation or use of PatchTST from Nie et al. 2023.
If using pre-implemented: pip install patch-tst OR use huggingface timeseries_transformers.
Patch size: 16 samples (1.6 seconds). Stride: 8 samples.
This represents the pure data-driven SOTA baseline.
Must be compared fairly: same data, same evaluation protocol.
Note in paper: PatchTST has no physics knowledge, compare generalization on 
held-out scenario to show PINN's physics advantage.
"""
```

***

## SECTION 10: PHM-GRADE EVALUATION FRAMEWORK

### `eval/phm_metrics.py` — Every Metric Specified

```python
def evaluate_full_phm(
    model,
    test_loader: DataLoader,
    horizons: list[int] = [10, 50, 100, 300],  # samples = [1s, 5s, 10s, 30s]
    false_alarm_rates: list[float] = [0.01, 0.05, 0.10]
) -> dict:
    """
    Computes ALL of the following metrics. Every single one must appear in the results dict.
    
    PER HORIZON τ ∈ {1s, 5s, 10s, 30s}:
    ─────────────────────────────────────
    1. AUROC(τ):     Area under ROC curve
    2. AUPRC(τ):     Area under Precision-Recall curve  [MORE IMPORTANT than AUROC under imbalance]
    3. F1_max(τ):    Maximum F1 over all thresholds
    4. Recall@FAR1%: Recall when FAR is constrained to 1%
    5. Recall@FAR5%: Recall when FAR is constrained to 5%
    
    LEAD TIME METRICS (only for τ=10s and τ=30s):
    ──────────────────────────────────────────────
    6. lead_time_p25: 25th percentile of lead time distribution (seconds)
    7. lead_time_p50: Median lead time (seconds)
    8. lead_time_p75: 75th percentile of lead time
    9. detection_rate: Fraction of faults caught with ≥5s lead time
    
    VOLTAGE FORECAST METRICS:
    ─────────────────────────
    10. RMSE_forecast:  Root Mean Squared Error (volts)
    11. MAE_forecast:   Mean Absolute Error (volts)
    12. safe_band_coverage: % of steps where both actual and predicted V ∈ [23.5, 32.5]
    13. within_1V_accuracy: % of predictions within 1V of actual
    
    MECHANISM CLASSIFICATION:
    ─────────────────────────
    14. mechanism_accuracy: Multi-class accuracy
    15. mechanism_macro_f1: Macro-averaged F1 across 4 fault types
    
    CALIBRATION:
    ────────────
    16. ECE: Expected Calibration Error (using 10 equal-width bins)
    17. MCE: Maximum Calibration Error
    (See calibration.py for reliability diagram generation)
    
    RETURNS: nested dict with all 17+ metrics
    """
```

**Lead time computation (implement this exactly):**
```python
def compute_lead_time_distribution(
    fault_predictions_proba: np.ndarray,  # Shape: (n_timesteps,)
    fault_ground_truth: np.ndarray,       # Shape: (n_timesteps,) binary
    threshold: float = 0.5,
    sampling_rate_hz: float = 10.0
) -> np.ndarray:
    """
    For each actual fault event:
    1. Find all prediction triggers (proba > threshold) in the window BEFORE the event
    2. The lead time = (fault_event_sample - first_trigger_sample) / sampling_rate_hz
    3. If no trigger before event: lead_time = 0 (missed fault)
    4. If trigger with no subsequent fault within 30s: false alarm (exclude from lead time)
    
    Returns array of lead times in seconds (one per detected fault event).
    Use percentiles of this array for p25, p50, p75 metrics.
    """
```

### `eval/calibration.py`
```python
def compute_ece(
    probabilities: np.ndarray,   # Predicted probabilities
    labels: np.ndarray,          # True binary labels
    n_bins: int = 10
) -> dict:
    """
    Expected Calibration Error: measures if probability p means p% of such predictions are correct.
    
    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    where B_b = samples in bin b, acc = fraction of positives, conf = mean predicted probability
    
    Also generates and saves reliability diagram (calibration curve) to outputs/figures/.
    """
```

### `eval/xai.py`
```python
def compute_shap_explanations(
    model: nn.Module,
    background_data: np.ndarray,     # 100 representative samples
    test_instances: np.ndarray,      # Instances to explain
    feature_names: list[str],
    task: str = "fault_10s"          # Which task head to explain
) -> dict:
    """
    Uses SHAP GradientExplainer for the PINN (neural network-compatible).
    
    Outputs:
    1. Global feature importance: mean |SHAP| per feature across test set
       → Save as bar chart to outputs/figures/shap_global_importance.png
    
    2. Per-instance salience map for 5 example pre-fault windows
       → SHAP values overlaid on voltage time series
       → Save as outputs/figures/shap_instance_{i}.png
    
    3. Physics feature importance: how much do physics-derived features 
       (dV/dt, power, ripple_amplitude) contribute vs raw lag features
       → This validates the physics augmentation in feature engineering
    
    Key paper insight to extract: SHAP must show that voltage_rate_of_change 
    and voltage_ripple_amplitude are top-5 features. If they are not, there is 
    a problem with either the simulator or feature engineering.
    """
```

***

## SECTION 11: ABLATION STUDIES

### `experiments/ablation.py` — 5 Mandatory Ablations

```python
ABLATION_CONFIGS = {
    
    "ablation_1_no_physics_loss": {
        "description": "Remove physics residual loss (PINN → plain TCN). Proves physics constraint adds value.",
        "modification": "Set lambda_physics = 0.0 in pinn.yaml",
        "expected_result": "AUROC drops on held-out emp_simulation scenario (worst case)",
        "key_metric": "AUROC_fault_10s on emp_simulation test set"
    },
    
    "ablation_2_heuristic_data": {
        "description": "Replace MIL-STD simulator with original heuristic generator from notebook.",
        "modification": "Use original Gaussian noise + random spike generator from internship code",
        "expected_result": "VVA metrics degrade (propensity AUC rises > 0.70), TSTR ratio drops",
        "key_metric": "Propensity AUC, TSTR F1 ratio"
    },
    
    "ablation_3_no_cgan": {
        "description": "Remove cGAN augmentation. Train on simulator data only (imbalanced).",
        "modification": "Skip cgan.py; use raw simulator output with standard oversampling",
        "expected_result": "Rare fault recall drops (especially thyristor incipient faults)",
        "key_metric": "Recall@FAR1% for fault_1s horizon on incipient fault subset"
    },
    
    "ablation_4_window_sizes": {
        "description": "Vary window size W ∈ {10, 25, 50, 100, 200} samples.",
        "modification": "Train 5 separate models with different window sizes",
        "expected_result": "W=100 optimal; W=10 too short for multi-horizon; W=200 no gain",
        "key_metric": "AUPRC_fault_30s vs window_size (plot as line chart)"
    },
    
    "ablation_5_physics_features": {
        "description": "Remove physics-derived features from feature engineering.",
        "modification": "Drop: dV/dt, power, impedance, ripple_amplitude, thermal_stress_index",
        "expected_result": "Performance drops for mechanism classification, less for binary warning",
        "key_metric": "mechanism_macro_f1, lead_time_p50"
    }
}

def run_all_ablations(base_config: dict, save_dir: str = "outputs/results/ablations/") -> None:
    """
    Runs all 5 ablations. Each ablation:
    1. Modifies base config per ABLATION_CONFIGS spec
    2. Trains model from scratch (same seed)
    3. Evaluates on same test splits
    4. Saves results to save_dir/{ablation_name}_results.json
    5. Logs to wandb with ablation tag
    
    After all ablations: generate comparison table and save to 
    outputs/figures/ablation_comparison_table.png
    """
```

***

## SECTION 12: EXPERIMENT TRACKING AND REPRODUCIBILITY

### `experiments/train.py`

```python
"""
EVERY training run must:
1. Initialize wandb run with full config dict (all hyperparameters, data stats, git hash)
2. Log: loss curves, metric curves, learning rate, gradient norm
3. Save checkpoint every 50 epochs to outputs/checkpoints/{run_id}_epoch_{n}.pt
4. Save best checkpoint (by val AUPRC_fault_10s) to outputs/checkpoints/{run_id}_best.pt
5. At end: save full results dict as outputs/results/{run_id}_final.json

Checkpoint format:
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'best_val_metric': float,
    'config': dict,
    'git_hash': str,    # subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    'timestamp': str
}

Resume training: check if checkpoint exists for this run_id before starting
"""
```

### `requirements.txt` — Exact Versions

```
torch==2.5.0
torchvision==0.20.0
deepxde==1.11.2
scipy==1.14.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
shap==0.46.0
wandb==0.18.5
matplotlib==3.9.2
seaborn==0.13.2
pyyaml==6.0.2
tqdm==4.66.5
imbalanced-learn==0.12.3
ydata-synthetic==1.3.3
pytest==8.3.3
pytorch-lightning==2.4.0
```

***

## SECTION 13: PUBLICATION FIGURE SPECIFICATIONS

Every figure that goes in the paper must be generated at **300 DPI, 8.5" × 5.5"** (IEEE double-column format), exported as both `.pdf` and `.png`. Use `matplotlib` with `rcParams` set as follows:

```python
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (8.5, 5.5),
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3
})
```

**Required figures for paper:**
1. `fig1_system_architecture.png` — Block diagram: DAE Simulator → cGAN Augmentation → Feature Engineering → PINN (multi-task heads)
2. `fig2_milstd_waveforms.png` — 4-panel: IES, load dump, vibration PSD, SRS shock showing exact MIL-STD compliance
3. `fig3_vva_results.png` — 4-panel: propensity scores, MMD values, TSTR comparison bar, ACF similarity
4. `fig4_roc_curves.png` — ROC curves for all 4 models at τ=10s horizon (4 lines on one plot)
5. `fig5_lead_time_distribution.png` — Violin plots of lead time distribution per model
6. `fig6_ablation_heatmap.png` — Heatmap: 5 ablations × 6 key metrics (color = relative performance)
7. `fig7_shap_explanations.png` — Top 15 global SHAP features + 2 pre-fault instance salience maps
8. `fig8_voltage_traces.png` — 3-panel: voltage trace + fault probability + actual fault markers (one for each mechanism type)

***

## SECTION 14: PAPER SUBMISSION CHECKLIST

Before submission to PHM Society 2026 (target deadline July 2026):

**Data:**
- [ ] Synthetic dataset released on Zenodo with DOI
- [ ] Code released on GitHub with paper-linked README
- [ ] All random seeds documented and frozen
- [ ] VVA metrics all pass acceptance thresholds

**Models:**
- [ ] PINN achieves AUROC > 0.90 on fault_10s horizon across all scenarios
- [ ] PINN achieves AUROC > 0.85 on held-out emp_simulation scenario
- [ ] Lead time p50 > 8 seconds for fault_10s horizon
- [ ] ECE < 0.08 (well-calibrated)
- [ ] TSTR / TRTR ratio > 0.90

**Evaluation:**
- [ ] All 5 ablations completed and logged
- [ ] All 4 baselines benchmarked on identical data splits
- [ ] Scenario-held-out test results reported separately
- [ ] SHAP explanations generated and physics features confirmed top-5

**Paper structure:**
- [ ] MIL-STD-1275E cited in section on transient modeling
- [ ] MIL-STD-810H cited in section on vibration simulation
- [ ] SAE ARP6887 cited in VVA section
- [ ] All 3 degradation mechanisms described with equations
- [ ] Physics residual loss formula written out explicitly
- [ ] Computational complexity / inference latency reported (ms per inference)
- [ ] "Edge deployability" discussion with parameter count and latency

***

## SECTION 15: SESSION MANAGEMENT FOR INTERRUPTED WORK

Since every phase may span multiple coding sessions, implement these patterns universally:

```python
# Pattern 1: Check-and-resume for data generation
def generate_with_resume(scenario: str, run_id: int, output_dir: str) -> str:
    output_path = f"{output_dir}/avr_data_{scenario}_run{run_id}.csv"
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists. Delete to regenerate.")
        return output_path
    # ... generate and save

# Pattern 2: Epoch checkpointing
if epoch % config.checkpoint_every_n_epochs == 0:
    torch.save(checkpoint_dict, f"outputs/checkpoints/{run_id}_epoch_{epoch}.pt")
    
# Pattern 3: Resuming from checkpoint
latest_ckpt = find_latest_checkpoint(run_id, checkpoint_dir)
if latest_ckpt:
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"[RESUME] Resuming from epoch {start_epoch}")

# Pattern 4: Memory-efficient data loading
class AVRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str):
        # Memory-map the CSV files instead of loading all into RAM
        self.memmap = np.memmap(f"{data_dir}/{split}.npy", 
                                dtype='float32', mode='r', 
                                shape=(n_samples, window_size, n_features))
    
# Pattern 5: Batch processing for feature engineering
def engineer_features_batched(raw_csv: str, output_csv: str, 
                                chunksize: int = 10000) -> None:
    writer = None
    for chunk in pd.read_csv(raw_csv, chunksize=chunksize):
        features = compute_features(chunk)
        if writer is None:
            features.to_csv(output_csv, index=False)
            writer = True
        else:
            features.to_csv(output_csv, mode='a', header=False, index=False)
```

***

## WHAT NOT TO DO — CRITICAL PROHIBITIONS

1. **Do not use `random_state=42` in train_test_split with time-series data.** Always use temporal splits.
2. **Do not apply data augmentation to test or validation sets.** cGAN augmentation is TRAINING ONLY.
3. **Do not report accuracy as a primary metric.** The fault class is rare (<5%); accuracy is misleading. Use AUPRC.
4. **Do not remove IES/cranking/load-dump from fault logs without logging them as `is_operational_transient=True`.** These must be preserved in evaluation to test model discrimination.
5. **Do not use the same cGAN augmented data for both TSTR training and the VVA propensity score test.** Use separate held-out sets.
6. **Do not train the Recurrent Autoencoder on fault data.** It must see ONLY healthy baseline data during training.
7. **Do not skip the `if __name__ == "__main__": run_tests()` block in any file.** Every file is testable in isolation.
8. **Do not hardcode file paths.** All paths must come from `config/paths.yaml`.
9. **Do not use deprecated pandas API.** No `.fillna(method='ffill')` → use `.ffill()`. No `DataFrame.append()` → use `pd.concat()`.
10. **Do not mix pu (per-unit) and physical units in the same array.** Always convert to physical units (Volts, Amps, °C) before saving to CSV. Document unit in column name: `voltage_v`, `current_a`, `temperature_c`.

***

This specification is complete. Implement Phase by Phase in this order:
**Section 2 (configs) → Section 3 (simulator) → Section 4 (data gen) → Section 5 (cGAN) → Section 6 (VVA) → Section 7 (features) → Section 8 (PINN) → Section 9 (baselines) → Section 10 (evaluation) → Section 11 (ablations) → Section 12 (training harness) → Section 13 (figures).**

Honest answer: **not quite — you have ~78% of what tier 1 needs.** The spec is strong on implementation but has critical gaps in scientific rigor, statistical validation, and paper architecture that would get you desk-rejected or major-revised at IEEE TII. Here is the complete gap analysis, and then every missing piece, copy-paste ready.

***

## Gap Scorecard

| Category | Status | Severity |
|---|---|---|
| Physics simulator (MIL-STD) | ✅ Complete | — |
| cGAN + VVA synthetic data | ✅ Complete | — |
| PINN architecture + loss | ✅ Complete | — |
| PHM-grade metrics | ✅ Complete | — |
| 5 ablation studies | ✅ Complete | — |
| XAI / SHAP | ✅ Complete | — |
| Reproducibility / checkpointing | ✅ Complete | — |
| **Statistical significance testing** | ❌ Missing | 🔴 Fatal for tier 1 |
| **Multi-seed reproducibility** | ❌ Missing | 🔴 Fatal for tier 1 |
| **Explicit research contributions block** | ❌ Missing | 🔴 Fatal for any venue |
| **RUL / time-to-failure estimation** | ❌ Missing | 🔴 PHM Society expects this |
| **Adversarial robustness experiments** | ❌ Missing | 🔴 Your key defense differentiator |
| **Uncertainty quantification** | ❌ Missing | 🔴 Commander trust argument |
| **Public benchmark validation** | ❌ Missing | 🟡 Major revision risk |
| **Hyperparameter sensitivity** | ❌ Missing | 🟡 Major revision risk |
| **Sensor noise robustness sweep** | ❌ Missing | 🟡 Major revision risk |
| **Computational complexity table** | ❌ Missing | 🟡 Edge deployment claim needs proof |
| **Concept drift / distribution shift** | ❌ Missing | 🟡 Reviewers will ask |
| **Limitations section plan** | ❌ Missing | 🟡 Standard for tier 1 |
| **Related work positioning plan** | ❌ Missing | 🟡 Must cite and differentiate |
| **PHM 2026 deadline timing** | ⚠️ Wrong | 🔴 Abstract deadline was TODAY 
***

## 🔴 CRITICAL ADDITIONS — 

### ADDITION 1: Multi-Seed Reproducibility (Non-Negotiable)

```
SECTION 16: MULTI-SEED REPRODUCIBILITY PROTOCOL

Every model (PINN, RAE, cGAN, RF, PatchTST) must be trained
FIVE times with seeds: [42, 123, 456, 789, 2026]

All reported metrics in the paper must be:
    mean ± std across the 5 runs

Example format for results table:
    AUROC (fault_10s): 0.934 ± 0.008

If std > 0.015 for any primary metric: the architecture is 
unstable — investigate before reporting.

In experiments/train.py, add outer loop:
    for seed in [42, 123, 456, 789, 2026]:
        set_all_seeds(seed)
        run_id = f"{model_name}_seed{seed}"
        train_and_evaluate(config, run_id)

Aggregate results across seeds:
    results_df = pd.DataFrame of all runs
    summary = results_df.groupby('model').agg(['mean', 'std'])
    Save to outputs/results/final_comparison_table.csv
```

***

### ADDITION 2: Statistical Significance Testing

```
SECTION 17: STATISTICAL SIGNIFICANCE TESTING

After collecting results across 5 seeds, perform pairwise 
significance testing between PINN and every baseline.

Use Wilcoxon Signed-Rank Test (non-parametric, correct for 
non-Gaussian metric distributions, standard in PHM literature):

from scipy.stats import wilcoxon

for baseline in ['threshold', 'rf', 'rae', 'patchtst']:
    for metric in ['auroc_10s', 'auprc_10s', 'lead_time_p50']:
        pinn_scores = load_seed_results('pinn', metric)    # 5 values
        base_scores = load_seed_results(baseline, metric)  # 5 values
        stat, p_value = wilcoxon(pinn_scores, base_scores)
        
        # Report p < 0.05 as significant, p < 0.01 as highly significant
        significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        log to: outputs/results/significance_tests.json

In paper results table, annotate with * (p<0.05) or ** (p<0.01).
This is the single most common reason papers get "major revision":
reviewers ask "is the improvement statistically significant?"
```

***

### ADDITION 3: Research Contributions Block

```
SECTION 18: EXPLICIT RESEARCH CONTRIBUTIONS

The paper Introduction must end with this exact block 
(fill in specifics once results are known):

"The main contributions of this paper are as follows:

(C1) A high-fidelity, physics-informed synthetic data generation 
     framework for military 28V DC electrical systems that rigorously 
     adheres to MIL-STD-1275E transient specifications and MIL-STD-810H 
     mechanical environment profiles, overcoming the classified data 
     barrier inherent to defense PHM research.

(C2) A Conditional WGAN-GP augmentation pipeline with a four-metric 
     VVA suite (MMD, propensity score, TSTR, ACF similarity) that 
     mathematically validates synthetic data quality and produces the 
     first open-access benchmark dataset for military AVR fault 
     prognostics, released on Zenodo (DOI: [to be assigned]).

(C3) A multi-task Physics-Informed Neural Network (PINN) that 
     simultaneously estimates multi-horizon fault warning probabilities 
     (τ ∈ {1,5,10,30}s), fault mechanism classification, and voltage 
     trajectory forecasting under a composite physics-residual loss 
     derived from d-q axis generator dynamics, achieving statistically 
     significant improvements over four baselines on scenario-held-out 
     evaluation (p < 0.01).

(C4) [If adversarial robustness is implemented] Empirical demonstration 
     that physics constraints provide inherent adversarial robustness, 
     maintaining AUROC > [X] under FGSM sensor spoofing attacks 
     where purely data-driven baselines degrade by > [Y]%."

These 3-4 contributions are referenced throughout the paper.
Reviewers check that every contribution claim is backed by results.
```

***

### ADDITION 4: RUL Estimation Head

```
SECTION 19: REMAINING USEFUL LIFE (RUL) ESTIMATION

PHM Society submissions without RUL are considered incomplete.
Add a 4th task head to the PINN.

Head 8 (RUL): 
    Dense(64) → Dense(32) → Dense(1)    # No activation
    Output: estimated samples until failure
    
Training target: 
    For each window in a degradation trajectory run:
        rul_label = (failure_sample_index - current_sample_index) / sampling_rate_hz
        # Clamp at max_rul = 300s (beyond 5 minutes, RUL = 300)
    
Loss: Asymmetric MSE (penalize late predictions more than early):
    def asymmetric_rul_loss(pred_rul, true_rul, alpha=1.3):
        error = pred_rul - true_rul
        # Late prediction (pred > true): multiply by alpha
        # Early prediction (pred < true): normal MSE
        loss = torch.where(error > 0, alpha * error**2, error**2)
        return loss.mean()

RUL Evaluation Metrics:
    1. RMSE_rul: Root mean squared error in seconds
    2. Score_rul: NASA scoring function (exponential penalty for late)
       score = Σ exp(-error/13) - 1 if error < 0 else exp(error/10) - 1
    3. RUL_p50_accuracy: % of predictions within ±20% of true RUL
    4. Plot: predicted vs true RUL scatter plot (color by degradation mechanism)

Note: RUL is only computable for progressive degradation runs where 
failure is reached. Ensure at least 2 run-to-failure trajectories 
per fault mechanism (6 total minimum).
```

***

### ADDITION 5: Adversarial Robustness Module

```
SECTION 20: ADVERSARIAL ROBUSTNESS EXPERIMENTS

This is the key differentiator for defense applications.
No other PHM paper on military vehicles tests this.

FILE: eval/adversarial.py

Attacks to implement (all using the foolbox library):
    pip install foolbox

ATTACK 1: Fast Gradient Sign Method (FGSM)
    from foolbox.attacks import FGSM
    Perturb input sensor windows with ε ∈ {0.01, 0.05, 0.1, 0.2}
    (ε in normalized units = fraction of signal range)
    Measure AUROC degradation vs ε for all 4 models.
    
ATTACK 2: Projected Gradient Descent (PGD)
    from foolbox.attacks import LinfPGD
    Steps=20, α=ε/4
    Stronger than FGSM — measures worst-case robustness.

Physical plausibility constraint on attacks:
    Adversarial perturbations must stay within ±2V of original 
    (physically plausible sensor spoofing, not clearly detectable outlier)

Key hypothesis to prove in the paper:
    "Physics constraints in the PINN loss function act as an implicit 
    adversarial defense, because physically impossible sensor readings 
    produce high physics residuals that override the data-driven 
    prediction."

Measure: physics_residual_value for clean vs adversarial inputs
Expected: adversarial inputs → much higher physics residual → 
          PINN is naturally more skeptical of them

Results to report:
    Table: Model × Attack (FGSM, PGD) × ε → AUROC
    Insight: PINN's AUROC degradation under attack is smaller than RF/RAE
    
Adversarial robustness summary metric:
    Robustness_score = AUROC_adversarial / AUROC_clean
    Target for PINN: > 0.90 under FGSM ε=0.05
```

***

### ADDITION 6: Uncertainty Quantification

```
SECTION 21: UNCERTAINTY QUANTIFICATION

Critical for the "commander trust" argument in the paper.

METHOD: Monte Carlo Dropout (simplest, well-established)
    During inference, keep dropout ACTIVE (not disabled as usual).
    Run T=50 forward passes for each input window.
    
    fault_prob_mean = mean of 50 outputs       # Point estimate
    fault_prob_std  = std of 50 outputs        # Epistemic uncertainty
    
    Prediction interval: [mean - 2*std, mean + 2*std]

Add dropout layers to PINN:
    Shared representation: Dense(128) → Dropout(0.15) → LayerNorm → GELU
    Each task head: Dense(64) → Dropout(0.15) → output

Uncertainty metrics to report:
    1. PIW (Prediction Interval Width): mean width of 95% intervals
       Lower = more confident (good for healthy windows)
    2. PICP (Prediction Interval Coverage Probability): 
       Fraction of true labels inside predicted intervals
       Target: PICP ≥ 0.95 for a 95% PI
    3. Uncertainty vs degradation level: 
       Plot: mean epistemic uncertainty vs fault severity (0=healthy → 3=critical)
       Expected: uncertainty INCREASES as system approaches failure
       This is physically intuitive and convincing to reviewers.

Paper argument:
    "Our system outputs not just a fault probability but a calibrated 
    confidence interval. High uncertainty combined with rising fault 
    probability triggers a 'request inspection' signal rather than an 
    immediate 'ground vehicle' command, providing commanders with 
    actionable, uncertainty-aware intelligence."
```

***

### ADDITION 7: Public Benchmark Validation

```
SECTION 22: BENCHMARK VALIDATION ON PUBLIC DATASET

Reviewers at IEEE TII and PHM Society will ask: 
"Does your method generalize, or is it overfit to your own synthetic data?"

Solution: Validate the PINN architecture (NOT the military-specific 
physics constraints) on one public benchmark FIRST.

Recommended dataset: C-MAPSS (NASA Turbofan Engine Degradation)
    Download: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data
    Task: RUL prediction (standard PHM benchmark)
    
Use the PINN architecture without AVR-specific physics loss:
    Replace physics constraint with turbofan degradation ODE 
    (available in literature, cite: Frederick et al. 2007)

Report RMSE_rul on C-MAPSS FD001 and FD002 subsets.
Compare to published results from: 
    - Biswas et al. 2022 (LSTM baseline)
    - Chen et al. 2023 (Transformer baseline)
    - Pick 2 recent papers from PHM Society 2023/2024 proceedings

If your PINN achieves competitive results on C-MAPSS 
(within 10% of SOTA), add to paper as:
    "Section V-A: Generalizability Validation on Public Benchmark"
    
This single addition makes the paper much harder to reject on 
"only tested on synthetic data" grounds.

Paper argument:
    "To validate architectural generalizability independent of the 
    military-specific domain, we first evaluate the PINN framework on 
    the publicly available C-MAPSS benchmark, achieving RMSE=X, 
    competitive with state-of-the-art methods."
```

***

### ADDITION 8: Hyperparameter Sensitivity Analysis

```
SECTION 23: HYPERPARAMETER SENSITIVITY

Reviewers ask: "How sensitive is your method to the λ_physics weight?"
If you can't answer this, they assume you tuned it to make your method 
look good. Prove it's robust.

Sweep 1: Physics loss weight λ₂ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
    Fixed: λ₁ = (1 - λ₂ - 0.2), λ₃ = 0.2
    Metric: AUROC_fault_10s on validation set
    Expected: performance peaks around λ₂ = 0.2-0.4, degrades at extremes
    Plot as: line chart, λ₂ on x-axis, AUROC on y-axis
    
Sweep 2: cGAN augmentation ratio ∈ {0:1, 1:1, 2:1, 3:1, 5:1} (synth:real)
    Metric: Recall@FAR1% for incipient faults
    Expected: optimal around 2:1 or 3:1
    
Sweep 3: PINN dropout rate ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}
    Metric: ECE (calibration) and AUROC
    Expected: slight dropout (0.10-0.15) best for calibration
    
All sweeps: train one run per config (not 5 seeds — this is sensitivity,
not final results). Report in appendix / supplementary material.
Only the final selected hyperparameters use 5-seed evaluation.
```

***

### ADDITION 9: Computational Complexity Table

```
SECTION 24: COMPUTATIONAL COMPLEXITY ANALYSIS

This MUST appear in the paper to support the edge-deployment claim.

Measure for all 4 models:

1. Parameter count:
   from torchinfo import summary
   summary(model, input_size=(1, 100, 3+n_physics_features))
   
2. FLOPs per inference:
   from fvcore.nn import FlopCountAnalysis
   flops = FlopCountAnalysis(model, dummy_input)
   
3. Inference latency (measured on CPU, not GPU):
   import time
   latency_ms = []
   for _ in range(1000):
       t0 = time.perf_counter()
       model(batch)
       latency_ms.append((time.perf_counter() - t0) * 1000)
   report: mean ± std in ms
   
4. Memory footprint: 
   torch.cuda.memory_allocated() before and after forward pass

Target table for paper (fill in after measurement):

| Model         | Params  | FLOPs/inf | Latency(ms) CPU | Memory(MB) |
|---------------|---------|-----------|-----------------|------------|
| Threshold     | 0       | ~0        | <0.1            | <1         |
| Random Forest | ~50K    | ~1M       | [measure]       | [measure]  |
| Bi-GRU RAE   | ~200K   | ~5M       | [measure]       | [measure]  |
| PINN (ours)   | ~180K   | ~4M       | [measure]       | [measure]  |
| PatchTST      | ~1.5M   | ~50M      | [measure]       | [measure]  |

Edge viability criterion: Latency < 100ms on CPU 
(10Hz sampling → 100ms budget per inference)
```

***

## 🟡 CORRECTED PUBLICATION TIMELINE

The PHM Society 2026 Annual Conference abstract deadline **was today** (Feb 23, 2026).  That window is closed. Here is the corrected, realistic timeline: 

| Venue | Deadline | Feasibility |
|---|---|---|
| **PHME 2026** (European PHM Conference) | Full paper: **May 10, 2026**  [phm-europe](https://phm-europe.org) | Tight but possible if simulator is built in March |
| **PHM Society 2027 Annual Conference** | Abstract: ~Feb 2027 | Comfortable — best target for full implementation |
| **IEEE TII** (journal, no deadline) | Submit when ready | 6–12 month review; target submit by Sept 2026 |
| **Defence Science Journal** (DRDO's Scopus journal) | Rolling | Parallel track for India/DRDO credibility |

**Revised recommendation**: Target **PHME 2026** (May 10 deadline) with Phases 1–3 complete (simulator + cGAN + PINN), then submit the extended, fully-ablated version to **IEEE TII** by September 2026. This gives you a conference paper first (establishes the work) and then a full journal paper.

***

## Final Completeness Check

With all 9 additions above integrated into the original master spec, here is the final coverage:

| Requirement | Now Covered |
|---|---|
| Physics simulation + MIL-STD | ✅ |
| Synthetic data + VVA | ✅ |
| PINN + physics-residual loss | ✅ |
| Multi-seed reproducibility | ✅ (Addition 1) |
| Statistical significance | ✅ (Addition 2) |
| Explicit contributions block | ✅ (Addition 3) |
| RUL estimation | ✅ (Addition 4) |
| Adversarial robustness | ✅ (Addition 5) |
| Uncertainty quantification | ✅ (Addition 6) |
| Public benchmark validation | ✅ (Addition 7) |
| Hyperparameter sensitivity | ✅ (Addition 8) |
| Computational complexity | ✅ (Addition 9) |
| PHM-grade metrics + calibration | ✅ |
| 5 ablation studies | ✅ |
| XAI/SHAP | ✅ |
| Edge deployment analysis | ✅ |
| Open dataset on Zenodo | ✅ |
| Correct timeline | ✅ (corrected above) |

**You now have a complete, reviewer-proof specification for a tier 1/2 submission.** The original spec was strong enough for a tier 2 conference (PHM Society, PHME) as-is. With these 9 additions, it is strong enough for IEEE TII — which has a ~15% acceptance rate  and is where this work ultimately belongs. 