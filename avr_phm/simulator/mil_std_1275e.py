"""
MIL-STD-1275E transient waveform generators.

Implements exact voltage perturbation functions for standard 28V DC bus
transient events defined in MIL-STD-1275E (Characteristics of 28 Volt DC
Electrical Systems in Military Vehicles).

Each function returns a voltage perturbation dV(t) to be added to the
DAE output at each timestep. The simulation code tracks whether a voltage
exceedance is caused by a normal operational transient (IES, cranking,
known spike) or a genuine fault.

References:
    - MIL-STD-1275E (Department of Defense Interface Standard)
    - SAE AS5698: Vehicle Electrical Power Quality Characteristics

Mathematical basis:
    All waveforms are modeled as analytic functions of time relative to the
    event start time. See individual function docstrings for waveform equations.
"""

import random
from typing import Any

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def ies_waveform(t: float, event_start_s: float) -> float:
    """
    Initial Engagement Surge per MIL-STD-1275E.

    Purpose:
        Models the voltage drop that occurs during engine cranking when the
        starter motor engages. The bus voltage drops from nominal 28V to
        approximately 22V (a 6V drop) as the starter draws high current.

    Inputs:
        t: Current simulation time (s).
        event_start_s: Time at which the IES event begins (s).

    Outputs:
        dV perturbation in Volts (negative = voltage drop).

    Mathematical basis:
        Three-phase waveform:
        Phase 1 (drop):     Exponential voltage drop over rise_time (10ms)
                            dV = -V_drop * (1 - exp(-dt/tau_rise))
        Phase 2 (hold):     Constant depression for duration (500ms)
                            dV = -V_drop
        Phase 3 (recovery): Exponential recovery over recovery_time (200ms)
                            dV = -V_drop * exp(-(dt - duration)/tau_recovery)

        Parameters from config/milstd.yaml:
            V_drop = 6.0V, duration = 0.5s, rise_time = 10ms, recovery_time = 200ms
    """
    dt: float = t - event_start_s

    if dt < 0:
        return 0.0

    v_drop: float = 6.0
    rise_time_s: float = 0.010
    hold_duration_s: float = 0.5
    recovery_time_s: float = 0.200

    tau_rise: float = rise_time_s / 3.0
    tau_recovery: float = recovery_time_s / 3.0

    if dt < rise_time_s:
        # Phase 1: rapid voltage drop
        return -v_drop * (1.0 - np.exp(-dt / tau_rise))
    elif dt < rise_time_s + hold_duration_s:
        # Phase 2: held depression
        return -v_drop
    elif dt < rise_time_s + hold_duration_s + recovery_time_s:
        # Phase 3: exponential recovery
        dt_recovery: float = dt - rise_time_s - hold_duration_s
        return -v_drop * np.exp(-dt_recovery / tau_recovery)
    else:
        return 0.0


def cranking_depression(
    t: float,
    event_start_s: float,
    duration_s: float = 20.0,
) -> float:
    """
    Cranking voltage depression per MIL-STD-1275E.

    Purpose:
        Models the sustained low voltage condition during engine cranking.
        The bus holds at approximately 16V for up to 30s.
        CRITICAL: This is NOT a fault — the predictive model must NOT alarm
        during this state.

    Inputs:
        t: Current simulation time (s).
        event_start_s: Time at which cranking begins (s).
        duration_s: Duration of cranking in seconds (default 20s, max 30s).

    Outputs:
        dV perturbation in Volts (negative depression from 28V to ~16V).

    Mathematical basis:
        Modeled as smooth depression using cosine transition:
            During cranking: dV = -(28 - 16) = -12V
            Entry/exit ramps: 100ms cosine transitions for physical realism.

        Parameters from config/milstd.yaml:
            cranking_voltage_v = 16.0V, max_duration_s = 30.0s
    """
    dt: float = t - event_start_s

    if dt < 0 or dt > duration_s:
        return 0.0

    depression_v: float = 28.0 - 16.0  # 12V drop
    ramp_s: float = 0.1  # 100ms transition

    if dt < ramp_s:
        # Entry ramp (cosine transition for smoothness)
        return -depression_v * 0.5 * (1.0 - np.cos(np.pi * dt / ramp_s))
    elif dt > duration_s - ramp_s:
        # Exit ramp
        dt_exit: float = dt - (duration_s - ramp_s)
        return -depression_v * 0.5 * (1.0 + np.cos(np.pi * dt_exit / ramp_s))
    else:
        # Full depression
        return -depression_v


def spike_waveform(t: float, event_time_s: float) -> float:
    """
    MIL-STD-1275E voltage spike: 250V peak, 70µs duration, <1µs rise time.

    Purpose:
        Models high-voltage, short-duration, low-energy spikes caused by
        inductive load switching or relay contact bounce.

    Inputs:
        t: Current simulation time (s).
        event_time_s: Time at which the spike occurs (s).

    Outputs:
        dV perturbation in Volts (positive spike above nominal).

    Mathematical basis:
        Double-exponential pulse (Billington form):
            V(t) = V_peak * (exp(-dt/τ₁) - exp(-dt/τ₂))
        where:
            τ₁ = 50µs (fall time constant)
            τ₂ = 0.5µs (rise time constant)
            V_peak is scaled so the actual peak equals 250V above nominal.

        The peak of the double-exponential occurs at:
            t_peak = (τ₁*τ₂)/(τ₁-τ₂) * ln(τ₁/τ₂)

        Energy constraint: total energy ≤ 2J (per MIL-STD-1275E).

        Parameters from config/milstd.yaml:
            peak_voltage_v = 250.0, duration_us = 70.0, rise_time_us = 1.0
    """
    dt: float = t - event_time_s

    if dt < 0:
        return 0.0

    # Time constants for double-exponential
    tau1: float = 50.0e-6    # 50µs fall
    tau2: float = 0.5e-6     # 0.5µs rise

    # Beyond 500µs, spike is negligible
    if dt > 500.0e-6:
        return 0.0

    # Raw double-exponential
    raw: float = np.exp(-dt / tau1) - np.exp(-dt / tau2)

    # Compute peak normalization factor
    t_peak: float = (tau1 * tau2) / (tau1 - tau2) * np.log(tau1 / tau2)
    raw_peak: float = np.exp(-t_peak / tau1) - np.exp(-t_peak / tau2)

    # Scale to achieve 250V peak
    v_peak: float = 250.0
    if abs(raw_peak) > 1e-15:
        scale: float = v_peak / raw_peak
    else:
        scale = 0.0

    return scale * raw


def load_dump_waveform(
    t: float,
    event_start_s: float,
    repetition: int = 0,
) -> float:
    """
    Load dump surge per MIL-STD-1275E.

    Purpose:
        Models the voltage surge caused by sudden disconnection of a
        large inductive load. 100V peak from 0.5Ω source impedance,
        50ms duration. Repeats 3 times at 1s intervals.

    Inputs:
        t: Current simulation time (s).
        event_start_s: Time at which the first load dump begins (s).
        repetition: Which repetition (0, 1, or 2) to generate.

    Outputs:
        dV perturbation in Volts (positive surge).

    Mathematical basis:
        Waveform per pulse:
            Rise phase (0-2ms): exponential rise
                dV = V_peak * (1 - exp(-dt/τ_rise))
                τ_rise = 0.5ms
            Decay phase (2-50ms): exponential decay
                dV = V_peak * exp(-(dt-2ms)/τ_decay)
                τ_decay = 15ms

        Repetition timing: pulse at event_start + repetition * interval_s

        Parameters from config/milstd.yaml:
            peak_voltage_v = 100.0, source_impedance_ohm = 0.5,
            duration_ms = 50.0, repetitions = 3, interval_s = 1.0
    """
    interval_s: float = 1.0
    effective_start: float = event_start_s + repetition * interval_s
    dt: float = t - effective_start

    if dt < 0:
        return 0.0

    v_peak: float = 100.0
    rise_duration_s: float = 0.002    # 2ms rise
    total_duration_s: float = 0.050   # 50ms total
    tau_rise: float = 0.0005          # 0.5ms rise time constant
    tau_decay: float = 0.015          # 15ms decay time constant

    if dt > total_duration_s:
        return 0.0

    if dt < rise_duration_s:
        return v_peak * (1.0 - np.exp(-dt / tau_rise))
    else:
        dt_decay: float = dt - rise_duration_s
        return v_peak * np.exp(-dt_decay / tau_decay)


def compute_all_transient_perturbations(
    t: float,
    transient_events: list[dict[str, Any]],
) -> tuple[float, bool]:
    """
    Compute the total voltage perturbation from all active MIL-STD-1275E transients.

    Purpose:
        Sums contributions from all transient events active at time t.
        Also returns whether the perturbation is from a known operational
        transient (vs a genuine fault).

    Inputs:
        t: Current simulation time (s).
        transient_events: List of dicts, each with keys:
            {type: str, start_s: float, ...additional params...}
            type ∈ {'ies', 'cranking', 'spike', 'load_dump'}

    Outputs:
        (dV_total, is_operational_transient) tuple.

    Mathematical basis:
        Linear superposition of individual transient waveforms.
    """
    dv_total: float = 0.0
    is_transient: bool = False

    for event in transient_events:
        event_type: str = event["type"]
        start_s: float = event["start_s"]
        dv: float = 0.0

        if event_type == "ies":
            dv = ies_waveform(t, start_s)
        elif event_type == "cranking":
            duration: float = event.get("duration_s", 20.0)
            dv = cranking_depression(t, start_s, duration)
        elif event_type == "spike":
            dv = spike_waveform(t, start_s)
        elif event_type == "load_dump":
            # Sum all 3 repetitions
            for rep in range(3):
                dv += load_dump_waveform(t, start_s, rep)

        if abs(dv) > 0.01:
            is_transient = True

        dv_total += dv

    return dv_total, is_transient


def run_tests() -> None:
    """Sanity checks for MIL-STD-1275E waveform generators."""
    # Test 1: IES produces negative perturbation
    dv_ies: float = ies_waveform(0.25, 0.0)
    assert dv_ies < -5.0, (
        f"IES should produce ~-6V drop during hold, got {dv_ies:.2f}V"
    )

    # Test 2: Spike peak is near 250V
    # Find approximate peak time
    t_peak: float = (50e-6 * 0.5e-6) / (50e-6 - 0.5e-6) * np.log(
        50e-6 / 0.5e-6
    )
    dv_spike: float = spike_waveform(t_peak, 0.0)
    assert 200.0 < dv_spike < 260.0, (
        f"Spike peak should be ~250V, got {dv_spike:.2f}V"
    )

    # Test 3: Cranking depression is ~-12V
    dv_crank: float = cranking_depression(10.0, 0.0, 20.0)
    assert abs(dv_crank + 12.0) < 0.5, (
        f"Cranking depression should be ~-12V, got {dv_crank:.2f}V"
    )

    # Test 4: Load dump peak is near 100V
    dv_ld: float = load_dump_waveform(0.002, 0.0, 0)
    assert 80.0 < dv_ld < 110.0, (
        f"Load dump peak should be ~100V, got {dv_ld:.2f}V"
    )

    # Test 5: Before event, perturbation is zero
    assert ies_waveform(-1.0, 0.0) == 0.0
    assert spike_waveform(-1.0, 0.0) == 0.0
    assert load_dump_waveform(-1.0, 0.0) == 0.0

    print("[PASS] simulator/mil_std_1275e.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
