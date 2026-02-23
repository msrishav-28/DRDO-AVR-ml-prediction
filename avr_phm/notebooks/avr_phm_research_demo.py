# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AVR-PHM: Physics-Informed Digital Twin for Military Predictive Maintenance
#
# **Research Demonstration Notebook**
#
# This notebook demonstrates the complete AVR-PHM pipeline from physics simulation through
# model training, evaluation, and publication-quality figure generation. Each section maps
# to a specific phase of the master plan.
#
# ---
#
# ## Table of Contents
#
# 1. [Environment Setup](#1-environment-setup)
# 2. [Physics Simulator Verification](#2-physics-simulator)
# 3. [MIL-STD Waveform Generation](#3-milstd-waveforms)
# 4. [Fault Mechanism Simulation](#4-fault-mechanisms)
# 5. [Scenario Assembly and Data Generation](#5-data-generation)
# 6. [Feature Engineering](#6-feature-engineering)
# 7. [WGAN-GP Augmentation and VVA](#7-wgan-gp)
# 8. [PINN Architecture and Training](#8-pinn)
# 9. [Baseline Model Comparison](#9-baselines)
# 10. [Evaluation Framework](#10-evaluation)
# 11. [Ablation Studies](#11-ablation)
# 12. [Publication Figures](#12-figures)
#
# ---

# %% [markdown]
# ## 1. Environment Setup <a id="1-environment-setup"></a>
#
# We begin by importing all required modules and verifying the compute environment.

# %%
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add package root to path
PACKAGE_ROOT = Path(os.path.abspath("")).parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
    sys.path.insert(0, str(PACKAGE_ROOT / "avr_phm"))

# Reproducibility
SEED = 42
np.random.seed(SEED)

print(f"NumPy version:  {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# %%
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:     {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device:    {DEVICE}")

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# %% [markdown]
# ## 2. Physics Simulator Verification <a id="2-physics-simulator"></a>
#
# The core of AVR-PHM is an 8th-order synchronous machine DAE model with
# an IEEE Type I Automatic Voltage Regulator (AVR). The state vector is:
#
# $$\mathbf{x} = [\delta, \omega, E'_q, E'_d, E''_q, E''_d, V_f, V_r]^T$$
#
# The system is solved using a stiff-aware Radau IIA integrator from SciPy.

# %%
from simulator.constants import (
    H,
    Ka,
    Ke,
    Omega_base,
    Ra,
    Ta,
    Te,
    Tf,
    V_base,
    Vref,
    Xd,
    Xd_prime,
    Xd_dprime,
    Xq,
    Xq_prime,
    Xq_dprime,
    x0,
)

print("=== Synchronous Machine Parameters ===")
print(f"V_base  = {V_base:.1f} V (DC bus nominal)")
print(f"Ra      = {Ra:.5f} pu (armature resistance)")
print(f"Xd      = {Xd:.4f} pu (d-axis synchronous reactance)")
print(f"Xd'     = {Xd_prime:.4f} pu (d-axis transient reactance)")
print(f"Xd''    = {Xd_dprime:.4f} pu (d-axis sub-transient reactance)")
print(f"Xq      = {Xq:.4f} pu (q-axis synchronous reactance)")
print(f"H       = {H:.2f} s (inertia constant)")
print()
print("=== IEEE Type I AVR Parameters ===")
print(f"Ka      = {Ka:.1f} (amplifier gain)")
print(f"Ta      = {Ta:.3f} s (amplifier time constant)")
print(f"Ke      = {Ke:.2f} (exciter constant)")
print(f"Te      = {Te:.3f} s (exciter time constant)")
print(f"Tf      = {Tf:.3f} s (feedback time constant)")
print(f"Vref    = {Vref:.2f} pu (reference voltage)")
print()
print(f"Initial state x0 = {x0}")

# %%
from simulator.dae_model import solve_avr_dae

print("Running baseline DAE simulation (30s at 10Hz)...")
t_start = time.perf_counter()

result = solve_avr_dae(
    duration_s=30.0,
    dt=0.1,
    x0=x0,
    load_impedance_pu=0.5,
    temperature_c=25.0,
    seed=SEED,
)

t_elapsed = time.perf_counter() - t_start
print(f"Simulation completed in {t_elapsed:.2f}s")
print(f"Output shape: {result.shape}")
print(f"Columns: timestamp, Vt(V), It(A), 8 state variables")

# %%
# Plot the baseline voltage and current traces
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

t = result[:, 0]
v = result[:, 1]
i = result[:, 2]

axes[0].plot(t, v, color="#1976D2", linewidth=0.8)
axes[0].set_ylabel("Terminal Voltage (V)")
axes[0].axhline(y=28.0, color="gray", linestyle="--", linewidth=0.5, label="Nominal 28V")
axes[0].axhline(y=23.5, color="red", linestyle=":", linewidth=0.5, label="Under-V limit")
axes[0].axhline(y=32.5, color="red", linestyle=":", linewidth=0.5, label="Over-V limit")
axes[0].legend(fontsize=8)
axes[0].set_title("DAE Simulation: Baseline Scenario (30s)")

axes[1].plot(t, i, color="#388E3C", linewidth=0.8)
axes[1].set_ylabel("Current (A)")

axes[2].plot(t, result[:, 3], color="#E64A19", linewidth=0.8, label="delta")
axes[2].plot(t, result[:, 4], color="#7B1FA2", linewidth=0.8, label="omega")
axes[2].set_ylabel("State Variables")
axes[2].set_xlabel("Time (s)")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"\nVoltage statistics:")
print(f"  Mean:   {np.mean(v):.3f} V")
print(f"  Std:    {np.std(v):.4f} V")
print(f"  Min:    {np.min(v):.3f} V")
print(f"  Max:    {np.max(v):.3f} V")
print(f"  Within MIL-STD limits: {np.all((v >= 23.5) & (v <= 32.5))}")

# %% [markdown]
# ## 3. MIL-STD Waveform Generation <a id="3-milstd-waveforms"></a>
#
# The simulator overlays standardized military transient waveforms:
# - **MIL-STD-1275E**: Voltage transients (IES, cranking, spike, load dump)
# - **MIL-STD-810H**: Vibration-induced voltage ripple and ballistic shock

# %%
from simulator.mil_std_1275e import (
    cranking_depression,
    ies_waveform,
    load_dump_waveform,
    spike_waveform,
)

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle("MIL-STD-1275E Transient Waveforms", fontsize=14, fontweight="bold")

# (a) IES
t_ies = np.linspace(0, 1.0, 2000)
v_ies = np.array([28.0 + ies_waveform(t, 0.0) for t in t_ies])
axes[0, 0].plot(t_ies * 1000, v_ies, color="#1976D2", linewidth=1.2)
axes[0, 0].fill_between(t_ies * 1000, 23.5, v_ies, where=v_ies < 23.5,
                          alpha=0.3, color="red", label="Below spec")
axes[0, 0].axhline(y=28, color="gray", linestyle="--", linewidth=0.5)
axes[0, 0].set_title("(a) Initial Engagement Surge")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Voltage (V)")
axes[0, 0].legend(fontsize=7)

# (b) Cranking
t_crank = np.linspace(0, 30, 3000)
v_crank = np.array([28.0 + cranking_depression(t, 0.0, 20.0) for t in t_crank])
axes[0, 1].plot(t_crank, v_crank, color="#388E3C", linewidth=1.2)
axes[0, 1].axhline(y=16, color="orange", linestyle="--", linewidth=0.5)
axes[0, 1].set_title("(b) Cranking Depression")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Voltage (V)")

# (c) Spike
t_sp = np.linspace(0, 300e-6, 3000)
v_sp = np.array([28.0 + spike_waveform(t, 0.0) for t in t_sp])
axes[1, 0].plot(t_sp * 1e6, v_sp, color="#E64A19", linewidth=1.2)
axes[1, 0].set_title(r"(c) Voltage Spike (250V, 70$\mu$s)")
axes[1, 0].set_xlabel(r"Time ($\mu$s)")
axes[1, 0].set_ylabel("Voltage (V)")

# (d) Load dump
t_ld = np.linspace(0, 0.1, 1000)
v_ld = np.array([28.0 + load_dump_waveform(t, 0.0, 0) for t in t_ld])
axes[1, 1].plot(t_ld * 1000, v_ld, color="#7B1FA2", linewidth=1.2)
axes[1, 1].set_title("(d) Load Dump Surge (100V)")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Voltage (V)")

plt.tight_layout()
plt.show()

# %%
from simulator.mil_std_810h import generate_vibration_signal, vibration_to_voltage_ripple

print("=== MIL-STD-810H Vibration Simulation ===")
vib_signal = generate_vibration_signal(
    duration_s=2.0, fs=1000.0,
    freq_bands=[(5, 100, 0.04), (100, 500, 0.02)],
    seed=SEED,
)
print(f"Vibration signal shape: {vib_signal.shape}")
print(f"RMS acceleration: {np.sqrt(np.mean(vib_signal**2)):.4f} g")

ripple = vibration_to_voltage_ripple(
    vib_signal, fs=1000.0, air_gap_sensitivity=0.001
)
print(f"Voltage ripple RMS: {np.std(ripple) * 1000:.2f} mV")

# %% [markdown]
# ## 4. Fault Mechanism Simulation <a id="4-fault-mechanisms"></a>
#
# Three physics-based degradation models:
#
# | Mechanism | Physics Law | Effect |
# |-----------|-----------|--------|
# | **Thyristor Thermal Fatigue** | Coffin-Manson | Increased $V_f$ forward drop |
# | **Capacitor Degradation** | Arrhenius lifetime | Capacitance loss, ripple increase |
# | **Terminal Loosening** | Miner's rule | Increased contact impedance noise |

# %%
from simulator.fault_mechanisms import (
    CapacitorDegradation,
    TerminalLoosening,
    ThyristorThermalFatigue,
)

# Thyristor degradation over time
thyristor = ThyristorThermalFatigue(severity=0.0, temperature_c=85.0)
cap_deg = CapacitorDegradation(severity=0.0, temperature_c=55.0)
terminal = TerminalLoosening(severity=0.0)

severities = np.linspace(0, 1.0, 100)
vf_drops = []
cap_losses = []
noise_vars = []

for sev in severities:
    thyristor.severity = sev
    cap_deg.severity = sev
    terminal.severity = sev
    vf_drops.append(thyristor.compute_vf_increase())
    cap_losses.append(cap_deg.compute_capacitance_fraction())
    noise_vars.append(terminal.compute_noise_variance())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
fig.suptitle("Fault Mechanism Degradation Curves", fontweight="bold")

axes[0].plot(severities, vf_drops, color="#E64A19", linewidth=1.5)
axes[0].set_xlabel("Severity")
axes[0].set_ylabel("V_f Increase (V)")
axes[0].set_title("(a) Thyristor (Coffin-Manson)")
axes[0].grid(alpha=0.3)

axes[1].plot(severities, cap_losses, color="#1976D2", linewidth=1.5)
axes[1].set_xlabel("Severity")
axes[1].set_ylabel("Capacitance Fraction")
axes[1].set_title("(b) Capacitor (Arrhenius)")
axes[1].grid(alpha=0.3)

axes[2].plot(severities, noise_vars, color="#388E3C", linewidth=1.5)
axes[2].set_xlabel("Severity")
axes[2].set_ylabel("Noise Variance")
axes[2].set_title("(c) Terminal (Miner's Rule)")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Scenario Assembly and Data Generation <a id="5-data-generation"></a>
#
# The scenario engine assembles the DAE model, MIL-STD overlays, and fault
# mechanisms into complete simulation runs. The data pipeline generates
# **18 runs** across 7 scenarios.
#
# | Scenario | Runs | Duration | Temperature | Key Stress |
# |----------|------|----------|-------------|------------|
# | Baseline | 4 | 120 min | 25 C | None |
# | Arctic Cold | 2 | 30 min | -40 C | Cold start |
# | Desert Heat | 2 | 30 min | 55 C | Thermal |
# | Artillery Firing | 2 | 30 min | 45 C | Shock + vibration |
# | Rough Terrain | 2 | 30 min | 35 C | Continuous vibration |
# | Weapons Active | 2 | 30 min | 40 C | Heavy load transients |
# | EMP Simulation | 2 | 30 min | 30 C | EM interference |

# %%
from config import load_yaml

scenarios_cfg = load_yaml("scenarios")
print("=== Scenario Configuration Summary ===\n")
for name, params in scenarios_cfg["scenarios"].items():
    dur = params.get("duration_minutes", "?")
    temp = params.get("temperature_c", "?")
    print(f"  {name:20s}  duration={dur:>4}min  temp={temp:>4}C")

print(f"\n  Global sampling rate:   {scenarios_cfg['global']['sampling_rate_hz']} Hz")
print(f"  Global seed:            {scenarios_cfg['global'].get('seed', 42)}")

# %%
from data_gen.pipeline import GENERATION_ORDER

print(f"\n=== Data Generation Order ({len(GENERATION_ORDER)} runs) ===\n")
print(f"{'#':>3s}  {'Scenario':20s}  {'Run':>4s}  {'Progressive':>12s}")
print("-" * 50)
for i, spec in enumerate(GENERATION_ORDER):
    prog_marker = "[PROG]" if spec["progressive"] else ""
    print(f"{i+1:3d}  {spec['scenario']:20s}  {spec['run_id']:4d}  {prog_marker:>12s}")

# %% [markdown]
# ## 6. Feature Engineering <a id="6-feature-engineering"></a>
#
# The feature engineering pipeline computes **9 physics-derived features**, lag features,
# rolling statistics, and scenario encodings. All features include units in column names.

# %%
from features.engineer import (
    FEATURE_SPEC,
    compute_physics_features,
    compute_rolling_features,
    engineer_all_features,
)

# Create sample data from DAE output
sample_df = pd.DataFrame({
    "timestamp": result[:, 0],
    "voltage_v": result[:, 1],
    "current_a": result[:, 2],
    "temperature_c": np.full(len(result), 25.0),
    "scenario": "baseline",
    "run_id": 1,
})

print(f"Raw features: {sample_df.shape[1]} columns, {len(sample_df)} samples")

featured_df = engineer_all_features(sample_df)
print(f"After engineering: {featured_df.shape[1]} columns")
print(f"\nPhysics-derived features:")
physics_cols = [c for c in featured_df.columns if c in [
    "dv_dt", "di_dt", "power_instantaneous_w", "dp_dt",
    "voltage_deviation_v", "voltage_within_spec",
    "load_impedance_ohm", "thermal_stress_index",
    "voltage_ripple_amplitude_v",
]]
for col in physics_cols:
    print(f"  {col:35s}  mean={featured_df[col].mean():>10.4f}  std={featured_df[col].std():>10.4f}")

# %%
# Visualize key physics features
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle("Physics-Derived Features (Baseline Scenario)", fontweight="bold")

t_feat = featured_df["timestamp"].values

axes[0, 0].plot(t_feat, featured_df["dv_dt"], color="#1976D2", linewidth=0.6)
axes[0, 0].set_ylabel("dV/dt (V/s)")
axes[0, 0].set_title("Voltage Rate of Change")

axes[0, 1].plot(t_feat, featured_df["power_instantaneous_w"], color="#E64A19", linewidth=0.6)
axes[0, 1].set_ylabel("Power (W)")
axes[0, 1].set_title("Instantaneous Power")

axes[1, 0].plot(t_feat, featured_df["voltage_deviation_v"], color="#388E3C", linewidth=0.6)
axes[1, 0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[1, 0].set_ylabel("V_dev (V)")
axes[1, 0].set_title("Voltage Deviation from 28V Nominal")

axes[1, 1].plot(t_feat, featured_df["load_impedance_ohm"], color="#7B1FA2", linewidth=0.6)
axes[1, 1].set_ylabel("Z (Ohm)")
axes[1, 1].set_title("Estimated Load Impedance")

axes[2, 0].plot(t_feat, featured_df["thermal_stress_index"], color="#F57F17", linewidth=0.6)
axes[2, 0].set_ylabel("TSI")
axes[2, 0].set_title("Thermal Stress Index")

axes[2, 1].plot(t_feat, featured_df["voltage_ripple_amplitude_v"], color="#C62828", linewidth=0.6)
axes[2, 1].set_ylabel("Ripple (V)")
axes[2, 1].set_title("Voltage Ripple Amplitude")

for ax in axes.flat:
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. WGAN-GP Architecture and VVA <a id="7-wgan-gp"></a>
#
# The rare-fault augmentation system uses a Conditional WGAN-GP:
#
# | Component | Architecture |
# |-----------|-------------|
# | Generator | z(32) + condition(14) -> Dense(128) -> GRU(128, 2L) -> TimeDistributed(3) |
# | Critic | sequence(100,3) + condition(14) -> Dense(64) -> BiGRU(128, 2L) -> Dense(1) |
# | Training | WGAN-GP: n_critic=5, gradient penalty lambda=10 |
#
# Synthetic data quality is validated by the VVA suite (4 metrics).

# %%
from data_gen.cgan import (
    CONDITION_DIM,
    CGANCritic,
    CGANGenerator,
    encode_condition,
)

gen = CGANGenerator(latent_dim=32)
crit = CGANCritic()

n_gen_params = sum(p.numel() for p in gen.parameters())
n_crit_params = sum(p.numel() for p in crit.parameters())

print("=== WGAN-GP Architecture ===\n")
print(f"Generator parameters:  {n_gen_params:>10,}")
print(f"Critic parameters:     {n_crit_params:>10,}")
print(f"Total:                 {n_gen_params + n_crit_params:>10,}")
print(f"Condition dimension:   {CONDITION_DIM}")

# Demo: condition encoding
cond_baseline = encode_condition("baseline", "none", "healthy")
cond_fault = encode_condition("artillery_firing", "thyristor", "critical")
print(f"\nCondition (baseline/healthy):      {cond_baseline}")
print(f"Condition (artillery/thyristor):    {cond_fault}")

# %%
# Generator forward pass demonstration
z = torch.randn(8, 32)
c = torch.tensor(np.tile(cond_fault, (8, 1)), dtype=torch.float32)

with torch.no_grad():
    synthetic = gen(z, c)

print(f"\nSynthetic batch shape: {synthetic.shape}")
print(f"  Channel 0 (Voltage):     mean={synthetic[:,:,0].mean():.4f}, std={synthetic[:,:,0].std():.4f}")
print(f"  Channel 1 (Current):     mean={synthetic[:,:,1].mean():.4f}, std={synthetic[:,:,1].std():.4f}")
print(f"  Channel 2 (Temperature): mean={synthetic[:,:,2].mean():.4f}, std={synthetic[:,:,2].std():.4f}")

# Plot a single generated sequence
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
fig.suptitle("WGAN-GP Generated Sequence (Untrained)", fontweight="bold")
labels = ["Voltage", "Current", "Temperature"]
colors = ["#1976D2", "#388E3C", "#E64A19"]

for i, (label, color) in enumerate(zip(labels, colors)):
    axes[i].plot(synthetic[0, :, i].numpy(), color=color, linewidth=1)
    axes[i].set_title(label)
    axes[i].set_xlabel("Time Step")
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
from data_gen.vva import compute_mmd, compute_mmd_multikernel

# VVA metric demonstration: MMD of identical distributions
rng = np.random.default_rng(SEED)
data_a = rng.normal(0, 1, (200, 50, 3))
data_b = rng.normal(0, 1, (200, 50, 3))
data_c = rng.normal(0.5, 1.5, (200, 50, 3))

mmd_same = compute_mmd(data_a, data_b, sigma=1.0)
mmd_diff = compute_mmd(data_a, data_c, sigma=1.0)

print("=== VVA: MMD Demonstration ===\n")
print(f"MMD(same distribution):     {mmd_same:.6f}  {'PASS' if mmd_same < 0.05 else 'FAIL'}")
print(f"MMD(different distribution): {mmd_diff:.6f}  {'PASS' if mmd_diff < 0.05 else 'FAIL'}")

# Multi-kernel MMD
mk_results = compute_mmd_multikernel(data_a, data_b)
print(f"\nMulti-kernel MMD results:")
for k, v in mk_results.items():
    print(f"  {k:25s}: {v:.6f}")

# %% [markdown]
# ## 8. PINN Architecture and Training <a id="8-pinn"></a>
#
# The AVR-PHM PINN uses a 1D-CNN temporal encoder with 8 parallel task heads:
#
# | Head | Task | Output | Loss |
# |------|------|--------|------|
# | 1-4 | Fault at 1s/5s/10s/30s | Binary | Focal Loss |
# | 5 | Mechanism classification | 4-class | Cross-Entropy |
# | 6 | Voltage forecast | 10 steps | MSE |
# | 7 | Severity | 4-class | Cross-Entropy |
# | 8 | RUL estimation | Scalar | Asymmetric MSE |
#
# **Composite loss function:**
#
# $$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{physics} + \lambda_3 \mathcal{L}_{fault}$$

# %%
from models.pinn import AVRPINN, AVRPhysicsResidual, compute_total_loss, focal_loss

model = AVRPINN(n_input_features=10, window_size=100, dropout_rate=0.15)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=== PINN Architecture Summary ===\n")
print(f"Total parameters:      {n_params:>10,}")
print(f"Trainable parameters:  {n_trainable:>10,}")
print(f"Edge-deployable:       {'Yes' if n_params < 500000 else 'No'} (<500K target)")
print(f"Dropout rate:          0.15")
print(f"Window size:           100 (10s at 10Hz)")
print()

# Verify output shapes
x_test = torch.randn(4, 100, 10)
outputs = model(x_test)

print("Task head output shapes:")
for task_name, tensor in outputs.items():
    print(f"  {task_name:15s}: {str(tensor.shape):>15s}")

# %%
# MC Dropout uncertainty quantification
print("\n=== MC Dropout Uncertainty Quantification ===\n")
print("Running T=50 stochastic forward passes...")

uq_results = model.predict_with_uncertainty(x_test, n_forward_passes=50)

for task_name, stats in uq_results.items():
    mean_val = stats["mean"].mean().item()
    std_val = stats["std"].mean().item()
    print(f"  {task_name:15s}: mean={mean_val:>8.4f}, epistemic_std={std_val:>8.4f}")

# %%
# Loss function demonstration
print("\n=== Composite Loss Function ===\n")

# Create dummy targets
dummy_targets = {
    "forecast": torch.randn(4, 10),
    "fault_1s": torch.randint(0, 2, (4, 1)).float(),
    "fault_5s": torch.randint(0, 2, (4, 1)).float(),
    "fault_10s": torch.randint(0, 2, (4, 1)).float(),
    "fault_30s": torch.randint(0, 2, (4, 1)).float(),
    "mechanism": torch.randint(0, 4, (4,)),
    "severity": torch.randint(0, 4, (4,)),
    "rul": torch.rand(4, 1) * 100,
}

physics_res = torch.randn(4, 3) * 0.1

loss_dict = compute_total_loss(
    predictions=outputs,
    targets=dummy_targets,
    physics_residuals=physics_res,
    lambda_physics=0.3,
    lambda_data=0.5,
    lambda_fault=0.2,
)

print(f"Loss weights: lambda_data=0.5, lambda_physics=0.3, lambda_fault=0.2\n")
for loss_name, loss_val in loss_dict.items():
    print(f"  L_{loss_name:12s} = {loss_val.item():.6f}")

# %% [markdown]
# ## 9. Baseline Model Comparison <a id="9-baselines"></a>
#
# Four baseline models for ablation comparison:
#
# | Model | Type | Trainable Params | Purpose |
# |-------|------|-----------------|---------|
# | Threshold | Rule-based | 0 | Lower bound |
# | Random Forest | Ensemble | N/A | Best non-DL baseline |
# | Recurrent AE | Unsupervised DL | ~68K | Anomaly detection |
# | PatchTST | Transformer | ~50K | SOTA forecasting |

# %%
from models.baseline_rf import RFBaseline
from models.baseline_threshold import ThresholdDetector
from models.patchtst import PatchTST
from models.recurrent_ae import RecurrentAutoencoder

baseline_models = {
    "Threshold": ThresholdDetector(),
    "Random Forest": RFBaseline(n_estimators=10, n_jobs=1),
    "Recurrent AE": RecurrentAutoencoder(n_features=3),
    "PatchTST": PatchTST(n_features=3, seq_len=100),
}

print("=== Baseline Model Comparison ===\n")
print(f"{'Model':20s}  {'Parameters':>12s}  {'Type':15s}")
print("-" * 55)

for name, m in baseline_models.items():
    if hasattr(m, 'parameters'):
        n_p = sum(p.numel() for p in m.parameters())
        print(f"{name:20s}  {n_p:>12,}  {'Deep Learning':15s}")
    else:
        print(f"{name:20s}  {'N/A':>12s}  {'Classical':15s}")

# PatchTST output demonstration
ptst = baseline_models["PatchTST"]
ptst_out = ptst(x_test[:, :, :3])
print(f"\nPatchTST output shapes:")
for k, v in ptst_out.items():
    print(f"  {k}: {v.shape}")

# %% [markdown]
# ## 10. Evaluation Framework <a id="10-evaluation"></a>
#
# The PHM-grade evaluation framework computes 17+ metrics covering:
# - Classification (P, R, F1, AUC-ROC, AUC-PR, Specificity, FAR, MDR)
# - Forecast (MAE, RMSE, R-squared per step)
# - RUL (NASA Scoring Function, alpha-lambda accuracy)
# - Calibration (ECE, MCE)
# - Computational complexity (latency, throughput, memory)

# %%
from eval.calibration import compute_ece, plot_reliability_diagram
from eval.phm_metrics import (
    compute_classification_metrics,
    compute_computational_complexity,
    compute_forecast_metrics,
    compute_rul_metrics,
)

# Classification metrics demonstration
rng = np.random.default_rng(SEED)
y_true = rng.integers(0, 2, 500)
y_pred = y_true.copy()
flip_idx = rng.choice(500, 25, replace=False)
y_pred[flip_idx] = 1 - y_pred[flip_idx]
y_proba = rng.beta(5, 2, 500) * y_true + rng.beta(2, 5, 500) * (1 - y_true)

cls_metrics = compute_classification_metrics(y_true, y_pred, y_proba, horizon_name="10s")

print("=== Classification Metrics (10s Horizon, Simulated) ===\n")
for k, v in cls_metrics.items():
    print(f"  {k:30s}: {v:.4f}")

# %%
# RUL metrics with NASA scoring function
true_rul = rng.uniform(10, 200, 100)
pred_rul = true_rul + rng.normal(0, 15, 100)

rul_metrics = compute_rul_metrics(true_rul, pred_rul)

print("\n=== RUL Estimation Metrics ===\n")
for k, v in rul_metrics.items():
    print(f"  {k:25s}: {v:.4f}")

# %%
# Calibration analysis
cal_data = compute_ece(y_true, y_proba, n_bins=10)
print(f"\n=== Calibration Metrics ===\n")
print(f"  ECE: {cal_data['ece']:.4f}")
print(f"  MCE: {cal_data['mce']:.4f}")

# Reliability diagram
plot_reliability_diagram(cal_data, model_name="PINN (Simulated)", save_path=None)

# %%
# Computational complexity
print("\n=== Computational Complexity (PINN) ===\n")
comp = compute_computational_complexity(
    model, input_shape=(1, 100, 10), device=DEVICE, n_warmup=5, n_runs=50
)
for k, v in comp.items():
    print(f"  {k:30s}: {v:.4f}")

# %% [markdown]
# ## 11. Ablation Studies <a id="11-ablation"></a>
#
# Five mandatory ablation experiments to isolate the contribution of each component:

# %%
from experiments.ablation import ABLATION_CONFIGS, HYPERPARAM_SWEEPS

print("=== Mandatory Ablation Configurations ===\n")
for name, config in ABLATION_CONFIGS.items():
    print(f"  {name:20s}: {config['description']}")
    print(f"    {'physics_weight':20s} = {config['physics_weight']}")
    print(f"    {'use_cgan_data':20s} = {config['use_cgan_data']}")
    print(f"    {'multi_task':20s} = {config['multi_task']}")
    print()

print("=== Hyperparameter Sensitivity Sweeps ===\n")
for param, values in HYPERPARAM_SWEEPS.items():
    print(f"  {param}: {values}")

# %% [markdown]
# ## 12. Publication Figures <a id="12-figures"></a>
#
# Generate all 8 publication-quality figures (IEEE format, 300 DPI).

# %%
from experiments.figures import (
    fig1_system_architecture,
    fig4_cgan_distribution_overlay,
    fig5_training_curves,
    fig6_roc_curves,
    fig7_shap_importance,
    fig8_ablation_bar_chart,
)

import tempfile
fig_dir = tempfile.mkdtemp()

print("Generating publication figures...\n")
fig1_system_architecture(fig_dir)
fig4_cgan_distribution_overlay(fig_dir)
fig5_training_curves(fig_dir)
fig6_roc_curves(fig_dir)
fig7_shap_importance(fig_dir)
fig8_ablation_bar_chart(fig_dir)

generated = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
print(f"\nGenerated {len(generated)} figures:")
for f in sorted(generated):
    size_kb = os.path.getsize(os.path.join(fig_dir, f)) / 1024
    print(f"  {f:45s}  ({size_kb:.0f} KB)")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the complete AVR-PHM pipeline:
#
# | Phase | Module | Status |
# |-------|--------|--------|
# | Physics Simulator | `simulator/dae_model.py` | Verified |
# | MIL-STD Waveforms | `simulator/mil_std_1275e.py` | Verified |
# | Fault Mechanisms | `simulator/fault_mechanisms.py` | Verified |
# | Data Pipeline | `data_gen/pipeline.py` | Configured |
# | Feature Engineering | `features/engineer.py` | Verified |
# | WGAN-GP | `data_gen/cgan.py` | Architecture verified |
# | VVA Suite | `data_gen/vva.py` | Metrics verified |
# | PINN | `models/pinn.py` | Architecture verified |
# | Baselines | `models/*.py` | All 4 verified |
# | Evaluation | `eval/*.py` | Metrics verified |
# | Ablation | `experiments/ablation.py` | Configured |
# | Figures | `experiments/figures.py` | Generated |
#
# **Next steps:**
# 1. Install dependencies: `pip install -r requirements.txt`
# 2. Generate full dataset: `python -m data_gen.pipeline`
# 3. Train PINN: `python -m experiments.train`
# 4. Evaluate all models: `python -m experiments.evaluate`
# 5. Run ablation studies: `python -m experiments.ablation`
