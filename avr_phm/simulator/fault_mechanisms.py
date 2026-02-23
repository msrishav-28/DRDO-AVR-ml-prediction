"""
Component degradation physics models for AVR fault mechanisms.

Implements exactly three degradation mechanisms as time-evolving parameter
drift in the DAE simulation:
    1. ThyristorThermalFatigue — Coffin-Manson model
    2. CapacitorDegradation — Arrhenius lifetime model
    3. TerminalLoosening — Miner's cumulative damage rule

Each mechanism modifies specific simulator parameters to create physically
realistic fault signatures that are distinct and detectable by the PHM system.

References:
    - Coffin, L.F. (1954). A study of the effects of cyclic thermal stresses.
    - Arrhenius, S. (1889). On the reaction velocity of the inversion of
      sucrose by acids.
    - Miner, M.A. (1945). Cumulative damage in fatigue.
"""

import random
from typing import Any

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from simulator.constants import (
    C_nominal_uf,
    E_activation_ev,
    Ka,
    Ke,
    L50_hours,
    SIGMA_SENSOR_NOMINAL,
    T_reference_k,
    Vf_degraded_v,
    Vf_nominal_v,
    k_boltzmann_ev,
    thermal_cycles_to_failure,
    vibration_fatigue_exponent,
)


class ThyristorThermalFatigue:
    """
    Models progressive increase in AVR thyristor forward voltage drop Vf
    due to thermal cycling fatigue (Coffin-Manson model).

    Physics:
        ΔVf(t) = Vf_nominal * (1 + α_Vf * N(t) / N_failure)
        where:
            N(t)      = cumulative thermal cycles at time t
            N_failure = thermal_cycles_to_failure from constants.py (50000)
            α_Vf      = 0.67 (Vf increases by 67% at failure)

    Effect on simulation:
        Modified Ke parameter in AVR: Ke_eff(t) = Ke * (1 - 0.3 * degradation_level)
        This causes progressively weaker exciter response to load steps.

    Observable signature:
        - Slower voltage recovery after load steps (recovery time increases)
        - Increased steady-state voltage error under heavy load
    """

    def __init__(self) -> None:
        """Initialize thyristor thermal fatigue model."""
        self.alpha_vf: float = 0.67
        self.n_failure: float = thermal_cycles_to_failure
        self.vf_nominal: float = Vf_nominal_v
        self.vf_degraded: float = Vf_degraded_v

    def get_degradation_level(
        self, cumulative_thermal_cycles: float
    ) -> float:
        """
        Compute degradation level from cumulative thermal cycle count.

        Purpose:
            Maps the number of thermal cycles endured to a normalized
            degradation level in [0, 1].

        Inputs:
            cumulative_thermal_cycles: Total thermal cycles accumulated.

        Outputs:
            degradation_level: Float in [0, 1]. 0.0 = new, 1.0 = failure.

        Mathematical basis:
            degradation = clamp(N(t) / N_failure, 0, 1)
        """
        level: float = min(
            cumulative_thermal_cycles / self.n_failure, 1.0
        )
        return max(level, 0.0)

    def modify_avr_parameter(
        self, base_ke: float, degradation_level: float
    ) -> float:
        """
        Compute modified exciter constant Ke for current degradation state.

        Purpose:
            As the thyristor degrades, its forward voltage drop increases,
            reducing the effective exciter gain.

        Inputs:
            base_ke: Nominal exciter constant (Ke from constants.py).
            degradation_level: Float in [0, 1] from get_degradation_level().

        Outputs:
            ke_effective: Modified Ke value.

        Mathematical basis:
            Ke_eff = Ke * (1 - 0.3 * degradation_level)
            At degradation_level=1: Ke reduces by 30%, causing significant
            excitation loss.
        """
        ke_effective: float = base_ke * (1.0 - 0.3 * degradation_level)
        return ke_effective

    def get_forward_voltage(self, degradation_level: float) -> float:
        """
        Compute current forward voltage drop.

        Inputs:
            degradation_level: Float in [0, 1].

        Outputs:
            Current Vf in Volts.

        Mathematical basis:
            Vf(t) = Vf_nominal * (1 + α_Vf * degradation_level)
        """
        vf: float = self.vf_nominal * (
            1.0 + self.alpha_vf * degradation_level
        )
        return vf

    def should_log_fault(
        self,
        voltage_error_pu: float,
        duration_samples: int,
        threshold_ms: float = 100.0,
        sampling_rate_hz: float = 10.0,
    ) -> bool:
        """
        Determine if a voltage error constitutes a loggable fault.

        Purpose:
            A fault is logged only if the voltage error has persisted
            beyond the fault_log_duration_ms threshold (100ms per spec).

        Inputs:
            voltage_error_pu: Magnitude of voltage deviation from nominal (pu).
            duration_samples: Number of consecutive samples with error.
            threshold_ms: Minimum persistence time in milliseconds.
            sampling_rate_hz: Data sampling rate in Hz.

        Outputs:
            True if the error qualifies as a loggable fault.
        """
        duration_ms: float = (duration_samples / sampling_rate_hz) * 1000.0
        return duration_ms > threshold_ms and abs(voltage_error_pu) > 0.05


class CapacitorDegradation:
    """
    Models reduction in EMI filter capacitance C using Arrhenius lifetime model.

    Physics:
        L(T) = L50 * exp(E_a / k_B * (1/T - 1/T_ref))
        As C decreases, high-frequency ripple amplitude INCREASES.

    Ripple increase model:
        Ripple_amplitude(t) = Ripple_nominal * (C_nominal / C_effective(t))
        C_effective(t) = C_nominal * (1 - 0.6 * degradation_level)

    Effect on simulation:
        Additional high-frequency noise term added to voltage output:
        V_ripple_hz = f_ripple * sin(2π * f_sw * t)
        where f_sw = 10kHz (switching frequency)
        f_ripple amplitude scales inversely with C_effective

    Observable signature:
        - Increased voltage ripple amplitude (detectable in voltage_rolling_std)
        - Ripple frequency fixed at switching frequency
    """

    def __init__(self) -> None:
        """Initialize capacitor degradation model."""
        self.c_nominal: float = C_nominal_uf
        self.e_activation: float = E_activation_ev
        self.k_boltzmann: float = k_boltzmann_ev
        self.t_reference: float = T_reference_k
        self.l50: float = L50_hours

    def compute_lifetime_hours(self, temperature_k: float) -> float:
        """
        Compute expected capacitor lifetime at given temperature.

        Purpose:
            Uses the Arrhenius model to predict electrolytic capacitor
            lifetime as a function of operating temperature.

        Inputs:
            temperature_k: Junction temperature in Kelvin.

        Outputs:
            Expected lifetime in hours.

        Mathematical basis:
            L(T) = L50 * exp(E_a / k_B * (1/T - 1/T_ref))
            where L50 = 2000 hours at T_ref = 358.15K (85°C)
        """
        exponent: float = self.e_activation / self.k_boltzmann * (
            1.0 / temperature_k - 1.0 / self.t_reference
        )
        # Clamp exponent to prevent overflow
        exponent = min(exponent, 50.0)
        lifetime: float = self.l50 * np.exp(exponent)
        return lifetime

    def get_degradation_level(
        self,
        operating_hours: float,
        temperature_k: float,
    ) -> float:
        """
        Compute capacitor degradation level from operating time and temperature.

        Inputs:
            operating_hours: Total operating hours accumulated.
            temperature_k: Average operating temperature in Kelvin.

        Outputs:
            degradation_level: Float in [0, 1]. 1.0 = capacitor failure.

        Mathematical basis:
            degradation = clamp(operating_hours / L(T), 0, 1)
        """
        lifetime: float = self.compute_lifetime_hours(temperature_k)
        level: float = min(operating_hours / max(lifetime, 1.0), 1.0)
        return max(level, 0.0)

    def get_effective_capacitance(
        self, degradation_level: float
    ) -> float:
        """
        Compute effective capacitance at current degradation state.

        Inputs:
            degradation_level: Float in [0, 1].

        Outputs:
            Effective capacitance in µF.

        Mathematical basis:
            C_eff(t) = C_nominal * (1 - 0.6 * degradation_level)
            At failure: C drops to 40% of nominal.
        """
        c_eff: float = self.c_nominal * (
            1.0 - 0.6 * degradation_level
        )
        return max(c_eff, 0.01)

    def get_ripple_amplitude(self, degradation_level: float) -> float:
        """
        Compute voltage ripple amplitude at current degradation state.

        Inputs:
            degradation_level: Float in [0, 1].

        Outputs:
            Ripple amplitude in Volts.

        Mathematical basis:
            Ripple(t) = Ripple_nominal * (C_nominal / C_effective(t))
            Nominal ripple = 0.75V (MIL-STD-1275E steady-state ripple / 2)
        """
        nominal_ripple_v: float = 0.75
        c_eff: float = self.get_effective_capacitance(degradation_level)
        ripple: float = nominal_ripple_v * (self.c_nominal / c_eff)
        return ripple

    def compute_ripple_signal(
        self,
        t_array: np.ndarray,
        degradation_level: float,
    ) -> np.ndarray:
        """
        Generate the ripple voltage signal for an array of time points.

        Inputs:
            t_array: Array of time values in seconds.
            degradation_level: Current degradation level.

        Outputs:
            Ripple voltage array in Volts.

        Mathematical basis:
            V_ripple(t) = amplitude * sin(2π * f_sw * t)
            f_sw = 10kHz (switching frequency)
        """
        amplitude: float = self.get_ripple_amplitude(degradation_level)
        f_sw: float = 10000.0
        ripple: np.ndarray = amplitude * np.sin(
            2.0 * np.pi * f_sw * t_array
        )
        return ripple


class TerminalLoosening:
    """
    Models increase in voltage sensing measurement noise variance due to
    mechanical loosening of sensing terminal from vibration.
    Uses Miner's cumulative damage rule.

    Physics:
        D_miner(t) = Σ (n_i / N_fi)
        where n_i = cycles at stress amplitude i
        When D_miner ≥ 1.0: terminal effectively loose → high noise variance

    Noise model:
        σ_sensor(t) = σ_nominal * (1 + 9 * D_miner(t))
        σ_nominal = 0.05V (clean sensor noise floor)

    Effect on simulation:
        AVR feedback uses noisy voltage measurement:
        V_measured = V_actual + N(0, σ_sensor)
        This causes oscillatory AVR response → voltage hunting

    Observable signature:
        - High-frequency voltage oscillations at 2-5 Hz
        - Increases with vibration severity — worst in artillery_firing
          and rough_terrain
    """

    def __init__(self) -> None:
        """Initialize terminal loosening model."""
        self.sigma_nominal: float = SIGMA_SENSOR_NOMINAL
        self.fatigue_exponent: float = vibration_fatigue_exponent

    def compute_miner_damage(
        self,
        vibration_rms_g: float,
        duration_hours: float,
        reference_amplitude_g: float = 1.0,
        reference_cycles_to_failure: float = 1e6,
    ) -> float:
        """
        Compute Miner's cumulative damage from vibration exposure.

        Purpose:
            Accumulates fatigue damage from vibration loading using the
            linear damage accumulation rule (Miner's rule).

        Inputs:
            vibration_rms_g: RMS vibration level in g.
            duration_hours: Duration of exposure in hours.
            reference_amplitude_g: Reference stress amplitude for S-N curve.
            reference_cycles_to_failure: Cycles to failure at reference amplitude.

        Outputs:
            Miner damage fraction. D ≥ 1.0 indicates failure.

        Mathematical basis:
            N_f(S) = N_ref * (S_ref / S)^b
            D = n / N_f
            where:
                b = vibration_fatigue_exponent = 3.5
                n = cycles = frequency * duration_hours * 3600
                Assumed dominant frequency = 20 Hz (typical first resonance)
        """
        dominant_freq_hz: float = 20.0
        n_cycles: float = dominant_freq_hz * duration_hours * 3600.0

        if vibration_rms_g < 1e-10:
            return 0.0

        # S-N curve: N_f = N_ref * (S_ref / S)^b
        n_failure: float = reference_cycles_to_failure * (
            reference_amplitude_g / vibration_rms_g
        ) ** self.fatigue_exponent

        damage: float = n_cycles / max(n_failure, 1.0)
        return min(damage, 2.0)  # Cap at 2x failure for numerical stability

    def get_sensor_noise_sigma(self, miner_damage: float) -> float:
        """
        Compute sensor noise standard deviation at current damage level.

        Inputs:
            miner_damage: Miner's cumulative damage fraction.

        Outputs:
            σ_sensor in Volts.

        Mathematical basis:
            σ_sensor(t) = σ_nominal * (1 + 9 * D_miner(t))
            At D=0: σ = 0.05V (clean)
            At D=1: σ = 0.50V (10× increase → significant noise)
        """
        clamped_damage: float = min(max(miner_damage, 0.0), 1.0)
        sigma: float = self.sigma_nominal * (1.0 + 9.0 * clamped_damage)
        return sigma

    def generate_measurement_noise(
        self,
        n_samples: int,
        miner_damage: float,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a noise array for voltage measurement corruption.

        Inputs:
            n_samples: Number of noise samples to generate.
            miner_damage: Current Miner's damage level.
            seed: Random seed.

        Outputs:
            Noise array in Volts of shape (n_samples,).

        Mathematical basis:
            noise ~ N(0, σ_sensor²)
        """
        rng: np.random.Generator = np.random.default_rng(seed)
        sigma: float = self.get_sensor_noise_sigma(miner_damage)
        noise: np.ndarray = rng.normal(0.0, sigma, n_samples)
        return noise


def run_tests() -> None:
    """Sanity checks for fault mechanism models."""
    # Test 1: Thyristor degradation level in [0, 1]
    thyristor: ThyristorThermalFatigue = ThyristorThermalFatigue()
    level: float = thyristor.get_degradation_level(25000.0)
    assert 0.0 <= level <= 1.0, f"Degradation level {level} out of range"
    assert abs(level - 0.5) < 0.01, (
        f"At 50% cycles, degradation should be ~0.5, got {level}"
    )

    # Test 2: Ke modification at failure
    ke_eff: float = thyristor.modify_avr_parameter(1.0, 1.0)
    assert abs(ke_eff - 0.7) < 0.01, (
        f"At full degradation, Ke should be 0.7, got {ke_eff}"
    )

    # Test 3: Capacitor ripple increases with degradation
    cap: CapacitorDegradation = CapacitorDegradation()
    ripple_healthy: float = cap.get_ripple_amplitude(0.0)
    ripple_degraded: float = cap.get_ripple_amplitude(0.8)
    assert ripple_degraded > ripple_healthy, (
        "Ripple should increase with degradation"
    )

    # Test 4: Terminal noise sigma at full damage
    terminal: TerminalLoosening = TerminalLoosening()
    sigma: float = terminal.get_sensor_noise_sigma(1.0)
    expected_sigma: float = SIGMA_SENSOR_NOMINAL * 10.0
    assert abs(sigma - expected_sigma) < 0.01, (
        f"At D=1, sigma should be {expected_sigma}, got {sigma}"
    )

    # Test 5: Capacitor lifetime physics
    lifetime_85c: float = cap.compute_lifetime_hours(358.15)
    assert abs(lifetime_85c - 2000.0) < 1.0, (
        f"Lifetime at reference temp should be L50=2000h, got {lifetime_85c}"
    )

    print("[PASS] simulator/fault_mechanisms.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
