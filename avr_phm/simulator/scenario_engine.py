"""
Scenario assembly engine.

Orchestrates the combination of DAE model, MIL-STD waveforms, fault
mechanisms, and scenario-specific parameters to generate complete
mission simulation traces.

Each scenario is assembled from:
    1. Base DAE simulation (dae_model.py)
    2. MIL-STD-1275E transient overlay (mil_std_1275e.py)
    3. MIL-STD-810H vibration overlay (mil_std_810h.py)
    4. Fault mechanism parameter drift (fault_mechanisms.py)
    5. Scenario-specific load and environmental conditions (scenarios.yaml)
"""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from simulator.constants import (
    Ke,
    V_base,
    T_ambient_nominal_c,
)
from simulator.dae_model import simulate_avr_mission
from simulator.fault_mechanisms import (
    CapacitorDegradation,
    TerminalLoosening,
    ThyristorThermalFatigue,
)
from simulator.mil_std_1275e import (
    compute_all_transient_perturbations,
    ies_waveform,
    load_dump_waveform,
    spike_waveform,
)
from simulator.mil_std_810h import (
    generate_ballistic_shock_srs,
    generate_vibration_psd,
    shock_to_voltage_perturbation,
    vibration_to_voltage_ripple,
)


def build_fault_schedule(
    scenario_name: str,
    scenario_params: dict[str, Any],
    duration_s: float,
    progressive_degradation: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate a fault injection schedule for a scenario run.

    Purpose:
        Creates a list of timestamped fault events based on the scenario's
        base_fault_probability_per_sample and whether progressive degradation
        is enabled.

    Inputs:
        scenario_name: Name of the scenario.
        scenario_params: Parameters dict from scenarios.yaml.
        duration_s: Simulation duration in seconds.
        progressive_degradation: If True, simulate a full degradation
            trajectory from healthy to failure.
        seed: Random seed.

    Outputs:
        List of fault event dicts: [{time_s, mechanism, severity}, ...]

    Mathematical basis:
        For progressive degradation: severity ramps linearly from 0 to 1.
        For random faults: Bernoulli sampling with base_fault_probability_per_sample.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    fault_schedule: list[dict[str, Any]] = []
    prob: float = scenario_params.get(
        "base_fault_probability_per_sample", 0.001
    )
    mechanisms: list[str] = ["thyristor", "capacitor", "terminal"]

    if progressive_degradation:
        # Full degradation trajectory — ramp one mechanism from 0 to 1
        chosen_mechanism: str = rng.choice(mechanisms)
        n_degradation_points: int = 10
        for i in range(n_degradation_points):
            time_s: float = (i + 1) * duration_s / (n_degradation_points + 1)
            severity: float = (i + 1) / n_degradation_points
            fault_schedule.append({
                "time_s": time_s,
                "mechanism": chosen_mechanism,
                "severity": severity,
            })
    else:
        # Stochastic fault injection
        sampling_rate: float = 10.0
        n_samples: int = int(duration_s * sampling_rate)
        for i in range(n_samples):
            if rng.random() < prob:
                time_s = i / sampling_rate
                mechanism: str = rng.choice(mechanisms)
                severity = float(rng.uniform(0.1, 0.9))
                fault_schedule.append({
                    "time_s": time_s,
                    "mechanism": mechanism,
                    "severity": severity,
                })

    return fault_schedule


def build_transient_events(
    scenario_name: str,
    scenario_params: dict[str, Any],
    duration_s: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate MIL-STD-1275E transient event schedule for a scenario.

    Purpose:
        Creates a list of operational transient events (IES, cranking,
        spikes, load dumps) appropriate for the scenario.

    Inputs:
        scenario_name: Name of the scenario.
        scenario_params: Parameters from scenarios.yaml.
        duration_s: Simulation duration in seconds.
        seed: Random seed.

    Outputs:
        List of transient event dicts: [{type, start_s, ...}, ...]
    """
    rng: np.random.Generator = np.random.default_rng(seed + 1000)
    events: list[dict[str, Any]] = []

    if scenario_name == "arctic_cold":
        # IES events — engine restarts in cold conditions
        ies_per_hour: int = scenario_params.get("ies_events_per_hour", 2)
        duration_hours: float = duration_s / 3600.0
        n_ies: int = max(1, int(ies_per_hour * duration_hours))
        ies_times: np.ndarray = rng.uniform(
            5.0, duration_s - 5.0, size=n_ies
        )
        for t_ies in sorted(ies_times):
            events.append({"type": "ies", "start_s": float(t_ies)})

    elif scenario_name == "weapons_active":
        # Load dump events
        ld_per_hour: int = scenario_params.get(
            "load_dump_events_per_hour", 4
        )
        duration_hours = duration_s / 3600.0
        n_ld: int = max(1, int(ld_per_hour * duration_hours))
        ld_times: np.ndarray = rng.uniform(
            10.0, duration_s - 10.0, size=n_ld
        )
        for t_ld in sorted(ld_times):
            events.append({"type": "load_dump", "start_s": float(t_ld)})

    elif scenario_name == "emp_simulation":
        # EMP spike event
        n_emp: int = scenario_params.get("emp_events", 1)
        emp_time: float = float(rng.uniform(
            duration_s * 0.3, duration_s * 0.7
        ))
        events.append({"type": "spike", "start_s": emp_time})

    elif scenario_name == "artillery_firing":
        # Firing events cause spikes
        firing_per_hour: int = scenario_params.get(
            "firing_events_per_hour", 6
        )
        duration_hours = duration_s / 3600.0
        n_fire: int = max(1, int(firing_per_hour * duration_hours))
        fire_times: np.ndarray = rng.uniform(
            5.0, duration_s - 5.0, size=n_fire
        )
        for t_fire in sorted(fire_times):
            events.append({"type": "spike", "start_s": float(t_fire)})

    return events


def simulate_scenario(
    scenario_name: str,
    scenario_params: dict[str, Any],
    run_id: int,
    progressive_degradation: bool = False,
    seed: int = 42,
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete scenario simulation with all overlays applied.

    Purpose:
        This is the top-level function that combines:
        1. Base DAE simulation
        2. MIL-STD-1275E transient overlays
        3. MIL-STD-810H vibration overlays
        4. Fault mechanism effects
        to produce a complete mission trace.

    Inputs:
        scenario_name: Name from scenarios.yaml.
        scenario_params: Full parameter dict for this scenario.
        run_id: Integer run identifier.
        progressive_degradation: If True, simulate run-to-failure trajectory.
        seed: Random seed for this run.
        save_dir: Directory to save output CSVs, or None.

    Outputs:
        (avr_timeseries_df, fault_log_df) — complete simulation outputs
        with all overlays applied. Columns use physical units:
        voltage_v, current_a, temperature_c.
    """
    duration_min: float = scenario_params.get("duration_minutes", 30)
    duration_s: float = duration_min * 60.0

    # Build augmented scenario params with name and run_id
    full_params: dict[str, Any] = {
        **scenario_params,
        "scenario_name": scenario_name,
        "run_id": run_id,
    }

    # Generate fault schedule
    fault_schedule: list[dict[str, Any]] = build_fault_schedule(
        scenario_name=scenario_name,
        scenario_params=scenario_params,
        duration_s=duration_s,
        progressive_degradation=progressive_degradation,
        seed=seed,
    )

    # Build save path
    save_path: str | None = None
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"avr_data_{scenario_name}_run{run_id}.csv",
        )

    # Run base DAE simulation
    avr_df, fault_df = simulate_avr_mission(
        duration_s=duration_s,
        scenario_params=full_params,
        fault_schedule=fault_schedule,
        dt=0.1,
        save_path=None,  # We save after overlays
    )

    # ─── Apply MIL-STD-1275E transient overlays ──────────────────────────────
    transient_events: list[dict[str, Any]] = build_transient_events(
        scenario_name=scenario_name,
        scenario_params=scenario_params,
        duration_s=duration_s,
        seed=seed,
    )

    if transient_events and len(avr_df) > 0:
        dv_transient: np.ndarray = np.zeros(len(avr_df))
        is_transient_flags: np.ndarray = np.zeros(len(avr_df), dtype=bool)

        for i, t_val in enumerate(avr_df["timestamp"].values):
            dv, is_trans = compute_all_transient_perturbations(
                float(t_val), transient_events
            )
            dv_transient[i] = dv
            is_transient_flags[i] = is_trans

        avr_df["voltage_v"] = avr_df["voltage_v"].values + dv_transient

        # Log operational transients in fault log
        for event in transient_events:
            fault_entry: dict[str, Any] = {
                "timestamp": event["start_s"],
                "fault_type": "operational_transient",
                "fault_mechanism": event["type"],
                "severity": 0.0,
                "duration_ms": 500.0 if event["type"] == "ies" else 50.0,
                "status": "normal",
                "is_operational_transient": True,
            }
            fault_df = pd.concat(
                [fault_df, pd.DataFrame([fault_entry])],
                ignore_index=True,
            )

    # ─── Apply MIL-STD-810H vibration overlay ────────────────────────────────
    vib_category: str | None = scenario_params.get(
        "vibration_psd_category", None
    )
    if vib_category is not None and len(avr_df) > 0:
        try:
            vib_signal: np.ndarray = generate_vibration_psd(
                duration_s=duration_s,
                sampling_rate_hz=10.0,
                category=vib_category,
                seed=seed + 2000,
            )
            # Trim or pad to match DataFrame length
            n_df: int = len(avr_df)
            if len(vib_signal) > n_df:
                vib_signal = vib_signal[:n_df]
            elif len(vib_signal) < n_df:
                vib_signal = np.pad(
                    vib_signal,
                    (0, n_df - len(vib_signal)),
                    mode="constant",
                )

            vib_ripple: np.ndarray = vibration_to_voltage_ripple(vib_signal)
            avr_df["voltage_v"] = avr_df["voltage_v"].values + vib_ripple
        except (ValueError, KeyError):
            pass  # Skip if category not found in config

    # ─── Apply ballistic shock if applicable ─────────────────────────────────
    if scenario_name == "artillery_firing":
        shock_g_peak: float = scenario_params.get(
            "ballistic_shock_g_peak", 40.0
        )
        # Apply shock at each firing event
        for event in transient_events:
            if event["type"] == "spike":
                shock_pulse: np.ndarray = generate_ballistic_shock_srs(
                    peak_acceleration_g=shock_g_peak,
                    duration_ms=11.0,
                )
                dv_shock: np.ndarray = shock_to_voltage_perturbation(
                    shock_acceleration_g=shock_pulse,
                    shock_duration_ms=11.0,
                    simulation_dt=0.1,
                )
                # Apply at event time
                start_idx: int = int(event["start_s"] * 10.0)
                end_idx: int = min(
                    start_idx + len(dv_shock), len(avr_df)
                )
                if start_idx < len(avr_df):
                    n_apply: int = end_idx - start_idx
                    avr_df.iloc[start_idx:end_idx, avr_df.columns.get_loc("voltage_v")] += (
                        dv_shock[:n_apply]
                    )

    # ─── Bug 38 fix: Apply scenario-specific physics from YAML params ──────
    # EMP post-recovery oscillation
    if scenario_name == "emp_simulation" and "emp_recovery_oscillation_std" in scenario_params:
        osc_std: float = scenario_params["emp_recovery_oscillation_std"]
        for event in transient_events:
            if event["type"] == "spike":
                emp_time: float = event["start_s"]
                osc_duration: float = 5.0  # 5s damped oscillation
                tau_damp: float = 1.0  # 1s damping time constant
                omega_osc: float = 2.0 * np.pi * 2.0  # 2 Hz
                for idx in range(len(avr_df)):
                    t_val: float = float(avr_df["timestamp"].iloc[idx])
                    if emp_time <= t_val < emp_time + osc_duration:
                        dt_emp: float = t_val - emp_time
                        dv_osc: float = osc_std * np.exp(-dt_emp / tau_damp) * np.sin(omega_osc * dt_emp)
                        avr_df.iloc[idx, avr_df.columns.get_loc("voltage_v")] += dv_osc

    # Weapons active load step (voltage sag)
    if scenario_name == "weapons_active" and "load_step_amps" in scenario_params:
        load_step_a: float = scenario_params["load_step_amps"]
        r_source: float = 0.05  # Ω typical source impedance
        dv_load_step: float = -load_step_a * r_source
        for event in transient_events:
            if event["type"] == "load_dump":
                ld_time: float = event["start_s"]
                ld_duration: float = 2.0  # load step lasts ~2s
                mask = (avr_df["timestamp"] >= ld_time) & (avr_df["timestamp"] < ld_time + ld_duration)
                avr_df.loc[mask, "voltage_v"] += dv_load_step

    # ─── Bug 25 fix: Clamp voltage to physical operating envelope ──────────
    # MIL-STD-1275E: -0.5V (reverse protection diode) to 100V (surge limit)
    avr_df["voltage_v"] = np.clip(avr_df["voltage_v"].values, -0.5, 100.0)

    # ─── Save final outputs ──────────────────────────────────────────────────
    if save_path is not None:
        avr_df.to_csv(save_path, index=False)
        fault_path: str = save_path.replace("avr_data_", "fault_log_")
        fault_df.to_csv(fault_path, index=False)

    # ─── Bug 36 fix: Run validator on outputs ────────────────────────────────
    import warnings
    from simulator.validator import validate_timeseries, validate_fault_log

    ts_result = validate_timeseries(avr_df, strict=False)
    if not ts_result.all_passed:
        warnings.warn(
            f"[VALIDATOR] Scenario '{scenario_name}' run {run_id} failed "
            f"{ts_result.failed}/{ts_result.passed + ts_result.failed} checks:\n"
            f"{ts_result.summary()}",
            RuntimeWarning, stacklevel=2,
        )

    fl_result = validate_fault_log(fault_df)
    if not fl_result.all_passed:
        warnings.warn(
            f"[VALIDATOR] Fault log check failed:\n{fl_result.summary()}",
            RuntimeWarning, stacklevel=2,
        )

    return avr_df, fault_df


def run_tests() -> None:
    """Sanity checks for scenario engine."""
    # Test 1: Fault schedule generation
    params: dict[str, Any] = {
        "base_fault_probability_per_sample": 0.01,
        "duration_minutes": 1,
    }
    schedule: list[dict[str, Any]] = build_fault_schedule(
        "baseline", params, 60.0, progressive_degradation=False, seed=42
    )
    assert isinstance(schedule, list), "Fault schedule must be a list"

    # Test 2: Progressive degradation schedule
    prog_schedule: list[dict[str, Any]] = build_fault_schedule(
        "baseline", params, 60.0, progressive_degradation=True, seed=42
    )
    assert len(prog_schedule) == 10, (
        f"Progressive schedule should have 10 points, got {len(prog_schedule)}"
    )
    assert prog_schedule[-1]["severity"] == 1.0, (
        "Last progressive point should have severity 1.0"
    )

    # Test 3: Transient event generation
    arctic_params: dict[str, Any] = {
        "ies_events_per_hour": 2,
        "duration_minutes": 30,
    }
    events: list[dict[str, Any]] = build_transient_events(
        "arctic_cold", arctic_params, 1800.0, seed=42
    )
    assert len(events) > 0, "Arctic scenario should generate IES events"
    assert events[0]["type"] == "ies", "Arctic events should be IES type"

    print("[PASS] simulator/scenario_engine.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
