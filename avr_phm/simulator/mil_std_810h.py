"""
MIL-STD-810H vibration and shock generators.

Implements PSD-shaped vibration signals (Method 514.8) and ballistic shock
pulses (Method 522.2) for military vehicle environments.

The vibration signals are coupled to electrical domain via the air-gap
sensitivity model: mechanical vibration modulates the generator air gap,
causing flux variation that produces EMF ripple.

References:
    - MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests
    - Method 514.8: Vibration
    - Method 522.2: Ballistic Shock

Mathematical basis:
    PSD generation: Inverse FFT of shaped spectrum with random phase.
    Air-gap coupling: Linear model Δv = sensitivity * acceleration_g.
    SRS shock: Half-sine pulse a(t) = A_peak * sin(π*t/T).
"""

import random
from typing import Any

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def generate_vibration_psd(
    duration_s: float,
    sampling_rate_hz: float,
    category: str,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a time-domain vibration acceleration signal (g) whose PSD
    matches MIL-STD-810H Method 514.8.

    Purpose:
        Creates a physically realistic random vibration signal for a
        specified vehicle category. The signal's power spectral density
        matches the breakpoint specification from the standard.

    Inputs:
        duration_s: Duration of the vibration signal in seconds.
        sampling_rate_hz: Sampling rate in Hz (typically 10 Hz for main
            simulation, but can be higher for sub-sampling).
        category: MIL-STD-810H vibration category string.
            "Category_4" = Wheeled vehicles on roads.
            "Category_20" = Wheeled off-road / Tracked vehicles.
        seed: Random seed for reproducibility.

    Outputs:
        acceleration_g: np.ndarray of shape (n_samples,) containing
            acceleration values in g (gravitational units).

    Mathematical basis:
        Steps:
        1. Build frequency array from 0 to Nyquist (sampling_rate_hz / 2).
        2. Interpolate PSD breakpoints (log-log) to get target PSD at each
           frequency bin.
        3. Compute amplitude spectrum: A(f) = sqrt(PSD(f) * df).
        4. Assign random phases uniformly in [0, 2π].
        5. Build complex spectrum and apply inverse FFT.
        6. Normalize output to match target RMS value.
    """
    from config import load_yaml

    milstd_cfg: dict[str, Any] = load_yaml("milstd")

    if category not in milstd_cfg["mil_std_810h"]:
        raise ValueError(
            f"Unknown vibration category '{category}'. "
            f"Valid: {list(milstd_cfg['mil_std_810h'].keys())}"
        )

    cat_cfg: dict[str, Any] = milstd_cfg["mil_std_810h"][category]
    breakpoints_hz: list[float] = cat_cfg["breakpoints_hz"]
    psd_values: list[float] = cat_cfg["psd_g2_per_hz"]

    rng: np.random.Generator = np.random.default_rng(seed)

    n_samples: int = int(duration_s * sampling_rate_hz)
    nyquist: float = sampling_rate_hz / 2.0
    df: float = 1.0 / duration_s
    n_freq_bins: int = n_samples // 2 + 1

    # Build frequency array
    freqs: np.ndarray = np.linspace(0.0, nyquist, n_freq_bins)

    # Interpolate PSD breakpoints in log-log space
    log_bp_hz: np.ndarray = np.log10(
        np.array(breakpoints_hz, dtype=np.float64)
    )
    log_psd: np.ndarray = np.log10(
        np.array(psd_values, dtype=np.float64)
    )

    # Handle DC and frequencies below first breakpoint
    target_psd: np.ndarray = np.zeros(n_freq_bins)
    for i in range(n_freq_bins):
        f: float = freqs[i]
        if f <= 0.0:
            target_psd[i] = 0.0
        elif f < breakpoints_hz[0]:
            target_psd[i] = psd_values[0]
        elif f > breakpoints_hz[-1]:
            target_psd[i] = psd_values[-1]
        else:
            log_f: float = np.log10(f)
            target_psd[i] = 10.0 ** np.interp(log_f, log_bp_hz, log_psd)

    # Compute amplitude spectrum from PSD
    amplitude: np.ndarray = np.sqrt(target_psd * df)

    # Random phases
    phases: np.ndarray = rng.uniform(0.0, 2.0 * np.pi, n_freq_bins)
    phases[0] = 0.0  # DC component has zero phase

    # Build complex spectrum
    spectrum: np.ndarray = amplitude * np.exp(1j * phases)

    # Inverse FFT to get time-domain signal
    signal: np.ndarray = np.fft.irfft(spectrum, n=n_samples)

    # Normalize to correct RMS
    _trapz = getattr(np, "trapezoid", np.trapz)  # NumPy 2.0 compat
    target_rms: float = np.sqrt(_trapz(target_psd, freqs))
    if target_rms > 0.0:
        actual_rms: float = np.sqrt(np.mean(signal**2))
        if actual_rms > 1e-15:
            signal = signal * (target_rms / actual_rms)

    return signal


def vibration_to_voltage_ripple(
    acceleration_g: np.ndarray,
    air_gap_sensitivity: float = 0.02,
) -> np.ndarray:
    """
    Convert mechanical vibration (g) to voltage ripple perturbation (V).

    Purpose:
        Models the physical coupling chain:
        vibration → air-gap modulation → flux variation → EMF ripple.

    Inputs:
        acceleration_g: Array of acceleration values in g.
        air_gap_sensitivity: Coupling coefficient in V per g.
            Default 0.02 V/g is a conservative estimate for military
            brushless generators (per master plan spec).

    Outputs:
        dV_ripple: np.ndarray of voltage perturbation in Volts.

    Mathematical basis:
        Linear coupling model:
            ΔV(t) = air_gap_sensitivity × a(t)
        where a(t) is the acceleration in g at time t.
        This is a first-order approximation; higher-order effects
        (nonlinear flux saturation) are neglected.
    """
    dv_ripple: np.ndarray = air_gap_sensitivity * acceleration_g
    return dv_ripple


def generate_ballistic_shock_srs(
    peak_acceleration_g: float,
    duration_ms: float = 11.0,
    sampling_rate_hz: float = 10000.0,
) -> np.ndarray:
    """
    Generate a ballistic shock pulse consistent with MIL-STD-810H Method 522.2.

    Purpose:
        Creates a single ballistic shock event (e.g., from artillery firing
        or mine blast) modeled as a half-sine pulse.

    Inputs:
        peak_acceleration_g: Peak acceleration in g (e.g., 40g per spec).
        duration_ms: Pulse duration in milliseconds (default 11ms per spec).
        sampling_rate_hz: Sampling rate for the pulse waveform (default 10kHz
            for adequate temporal resolution of the fast pulse).

    Outputs:
        acceleration: np.ndarray of acceleration values in g.

    Mathematical basis:
        Half-sine pulse:
            a(t) = A_peak × sin(π × t / T)    for 0 ≤ t ≤ T
            a(t) = 0                           otherwise
        where T = duration_ms / 1000.

        SRS parameters from config/milstd.yaml:
            natural_frequency_hz = [100, 1000, 2000]
            srs_acceleration_g = [20, 40, 30]
            Q_factor = 10
            duration_ms = 11.0
    """
    duration_s: float = duration_ms / 1000.0
    n_samples: int = int(duration_s * sampling_rate_hz)

    t: np.ndarray = np.linspace(0.0, duration_s, n_samples)
    acceleration: np.ndarray = peak_acceleration_g * np.sin(
        np.pi * t / duration_s
    )

    return acceleration


def shock_to_voltage_perturbation(
    shock_acceleration_g: np.ndarray,
    shock_duration_ms: float,
    simulation_dt: float,
    simulation_sampling_rate_hz: float = 10.0,
    air_gap_sensitivity: float = 0.02,
) -> np.ndarray:
    """
    Downsample a high-rate shock pulse to simulation timestep and convert
    to voltage perturbation.

    Purpose:
        Ballistic shock pulses are generated at high sampling rates (10kHz)
        but the main simulation runs at 10Hz. This function downsamples
        appropriately and converts to voltage domain.

    Inputs:
        shock_acceleration_g: High-rate acceleration array from
            generate_ballistic_shock_srs().
        shock_duration_ms: Duration of the shock pulse in ms.
        simulation_dt: Simulation timestep in seconds (typically 0.1s).
        simulation_sampling_rate_hz: Main simulation rate (10 Hz).
        air_gap_sensitivity: V/g coupling coefficient.

    Outputs:
        dV_shock: np.ndarray of voltage perturbation at simulation rate.

    Mathematical basis:
        RMS averaging of the high-rate shock signal within each simulation
        timestep bin, then conversion via air_gap_sensitivity.
    """
    shock_duration_s: float = shock_duration_ms / 1000.0
    n_sim_steps: int = max(
        1, int(np.ceil(shock_duration_s / simulation_dt))
    )

    # Compute RMS acceleration per simulation timestep
    shock_rate: float = len(shock_acceleration_g) / shock_duration_s
    samples_per_step: int = max(1, int(shock_rate * simulation_dt))

    dv_shock: np.ndarray = np.zeros(n_sim_steps)
    for i in range(n_sim_steps):
        start_idx: int = i * samples_per_step
        end_idx: int = min((i + 1) * samples_per_step, len(shock_acceleration_g))
        if start_idx < len(shock_acceleration_g):
            rms_g: float = np.sqrt(
                np.mean(shock_acceleration_g[start_idx:end_idx] ** 2)
            )
            dv_shock[i] = air_gap_sensitivity * rms_g

    return dv_shock


def run_tests() -> None:
    """Sanity checks for MIL-STD-810H generators."""
    # Test 1: Vibration PSD signal has correct length
    duration: float = 10.0
    fs: float = 100.0
    signal: np.ndarray = generate_vibration_psd(
        duration_s=duration,
        sampling_rate_hz=fs,
        category="Category_4",
        seed=42,
    )
    expected_len: int = int(duration * fs)
    assert len(signal) == expected_len, (
        f"Expected {expected_len} samples, got {len(signal)}"
    )

    # Test 2: Voltage ripple has same shape as input
    ripple: np.ndarray = vibration_to_voltage_ripple(signal)
    assert ripple.shape == signal.shape, "Ripple shape mismatch"

    # Test 3: Ballistic shock peak matches input
    shock: np.ndarray = generate_ballistic_shock_srs(
        peak_acceleration_g=40.0,
        duration_ms=11.0,
    )
    assert abs(np.max(shock) - 40.0) < 0.1, (
        f"Shock peak should be 40g, got {np.max(shock):.2f}g"
    )

    # Test 4: Voltage ripple magnitude is physically reasonable
    max_ripple_v: float = np.max(np.abs(ripple))
    assert max_ripple_v < 5.0, (
        f"Max ripple {max_ripple_v:.2f}V seems too large for 0.02 V/g coupling"
    )

    print("[PASS] simulator/mil_std_810h.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
