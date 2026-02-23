"""
Physical constants and nominal parameters for the military AVR system.

All values are traceable to MIL-STD-1275E, IEEE standards for synchronous
generators, and standard brushless excitation system parameters from
literature. These constants are the single source of truth for the
entire simulation and must not be duplicated or overridden elsewhere.

References:
    - MIL-STD-1275E: Characteristics of 28 Volt DC Input Power
    - IEEE Std 421.5: IEEE Recommended Practice for Excitation System Models
    - Kundur, P. (1994). Power System Stability and Control. McGraw-Hill.

Mathematical basis:
    All per-unit (pu) values use the machine base:
        V_base = 28.0 V (DC bus nominal)
        I_base = 45.0 A (nominal load current)
        P_base = V_base * I_base = 1260 W
    Per-unit convention: x_pu = x_actual / x_base
"""

import random

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

import math

# ─── Generator Electrical Parameters ────────────────────────────────────────
Ra: float = 0.0031      # Armature resistance (pu) — stator winding
Xd: float = 1.81        # d-axis synchronous reactance (pu)
Xq: float = 1.76        # q-axis synchronous reactance (pu)
Xd_prime: float = 0.30  # d-axis transient reactance (pu)
Xq_prime: float = 0.65  # q-axis transient reactance (pu)
Xd_dprime: float = 0.23 # d-axis subtransient reactance (pu)
Xq_dprime: float = 0.25 # q-axis subtransient reactance (pu)

# ─── Time Constants ──────────────────────────────────────────────────────────
Td0_prime: float = 5.14    # d-axis open-circuit transient time constant (s)
Tq0_prime: float = 1.50    # q-axis open-circuit transient time constant (s)
Td0_dprime: float = 0.033  # d-axis open-circuit subtransient time constant (s)
Tq0_dprime: float = 0.05   # q-axis open-circuit subtransient time constant (s)

# ─── Inertia & Damping ────────────────────────────────────────────────────────
H: float = 3.0              # Inertia constant (s) — diesel-driven generator
D: float = 5.0              # Damping coefficient (pu)
omega_s: float = 2.0 * math.pi * 50.0  # Synchronous speed (rad/s) for 50Hz system

# ─── AVR Parameters (IEEE Type I Excitation System) ──────────────────────────
Ka: float = 200.0    # Amplifier gain
Ta: float = 0.025    # Amplifier time constant (s)
Ke: float = 1.0      # Exciter constant
Te: float = 0.1      # Exciter time constant (s)
Kf: float = 0.04     # Stabilizer gain
Tf: float = 1.0      # Stabilizer time constant (s)
Vref: float = 1.05   # Reference voltage setpoint (pu), maps to 28V at nominal load

# ─── System Base Values (for pu conversion) ───────────────────────────────────
V_base: float = 28.0              # Volts DC (MIL-STD-1275E nominal)
I_base: float = 45.0              # Amps (nominal load current)
P_base: float = V_base * I_base   # Watts (1260W)

# ─── Thermal Parameters ───────────────────────────────────────────────────────
T_junction_max_c: float = 150.0   # Maximum junction temperature for switching elements
T_ambient_nominal_c: float = 25.0
thermal_resistance_junction_ambient: float = 0.8  # °C/W (typical for thyristor in AVR)

# ─── Degradation Parameters ───────────────────────────────────────────────────
# Capacitor aging (Arrhenius model)
C_nominal_uf: float = 100.0       # Nominal EMI filter capacitance
E_activation_ev: float = 0.94     # Activation energy for electrolytic capacitor
k_boltzmann_ev: float = 8.617e-5  # Boltzmann constant in eV/K
T_reference_k: float = 358.15     # Reference temperature 85°C in Kelvin (datasheet base)
L50_hours: float = 2000.0         # Median life at reference temperature (hours)

# Thyristor thermal fatigue
Vf_nominal_v: float = 1.5         # Nominal forward voltage drop
Vf_degraded_v: float = 2.5        # Degraded forward voltage drop (EoL threshold)
thermal_cycles_to_failure: float = 50000.0  # N_f in Coffin-Manson model

# Wiring terminal loosening (Miner's Rule)
vibration_fatigue_exponent: float = 3.5  # b in S-N curve for terminal connectors

# ─── Load Model Defaults ─────────────────────────────────────────────────────
R_load_nominal: float = 0.62      # Ohms — nominal load resistance (45A at 28V)
X_load_nominal: float = 0.15      # Ohms — nominal load reactance
R_load_weapons: float = 0.35      # Ohms — reduced resistance during weapons engagement
R_load_arctic: float = 0.85       # Ohms — increased resistance at cold (reduced battery SoC)
LOAD_RAMP_DURATION_S: float = 0.05  # 50ms ramp for load changes

# ─── Sensor Noise Floor ──────────────────────────────────────────────────────
SIGMA_SENSOR_NOMINAL: float = 0.05  # V — clean sensor noise floor

# ─── State Vector Indices ─────────────────────────────────────────────────────
# x = [delta, omega, Eq_prime, Ed_prime, Eq_dprime, Ed_dprime, Vf, Vr]
IDX_DELTA: int = 0
IDX_OMEGA: int = 1
IDX_EQ_PRIME: int = 2
IDX_ED_PRIME: int = 3
IDX_EQ_DPRIME: int = 4
IDX_ED_DPRIME: int = 5
IDX_VF: int = 6
IDX_VR: int = 7
N_STATES: int = 8

# ─── Initial Conditions ──────────────────────────────────────────────────────
# Nominal steady-state operating point (pre-computed for 28V, 45A)
X0_NOMINAL: list[float] = [
    0.5,    # delta (rad) — typical rotor angle at nominal load
    0.0,    # omega (pu) — synchronous speed deviation = 0 at steady state
    1.05,   # Eq_prime (pu) — matches Vref
    0.0,    # Ed_prime (pu) — zero for cylindrical rotor approximation
    1.05,   # Eq_dprime (pu) — subtransient EMF ≈ Eq_prime at steady state
    0.0,    # Ed_dprime (pu)
    1.05,   # Vf (pu) — field voltage at steady state = Vref
    0.0,    # Vr (pu) — regulator state at steady state = 0
]


def run_tests() -> None:
    """Sanity checks for physical constants."""
    # Test 1: Base power calculation
    assert abs(P_base - 1260.0) < 1e-6, (
        f"P_base should be 1260W, got {P_base}"
    )

    # Test 2: Reactance ordering (physical consistency)
    assert Xd_dprime < Xd_prime < Xd, (
        "Reactance ordering violated: Xd'' < Xd' < Xd required"
    )

    # Test 3: Synchronous speed for 50Hz
    expected_omega_s: float = 2.0 * math.pi * 50.0
    assert abs(omega_s - expected_omega_s) < 1e-6, (
        f"omega_s should be {expected_omega_s}, got {omega_s}"
    )

    # Test 4: State vector size
    assert N_STATES == 8, f"State vector must have 8 states, got {N_STATES}"
    assert len(X0_NOMINAL) == N_STATES, (
        f"Initial conditions must have {N_STATES} elements"
    )

    print("[PASS] simulator/constants.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
