"""
8th-order synchronous machine model with IEEE Type I AVR.

Implements the complete Differential-Algebraic Equation (DAE) system for a
military brushless synchronous generator driving a 28V DC bus through an
Automatic Voltage Regulator.

State vector:
    x = [delta, omega, Eq_prime, Ed_prime, Eq_dprime, Ed_dprime, Vf, Vr]
    where:
        delta     = rotor angle (rad)
        omega     = rotor angular velocity deviation (pu)
        Eq_prime  = q-axis transient EMF (pu)
        Ed_prime  = d-axis transient EMF (pu)
        Eq_dprime = q-axis subtransient EMF (pu)
        Ed_dprime = d-axis subtransient EMF (pu)
        Vf        = field voltage from exciter (pu)
        Vr        = voltage regulator state (pu)

Solver: scipy.integrate.solve_ivp with method='Radau' (stiff solver,
required for DAE systems with fast electrical + slow mechanical timescales).

Mathematical basis:
    - Park's transformation (d-q axis reference frame)
    - IEEE Type I excitation system model (IEEE Std 421.5)
    - Algebraic stator equations linking internal EMFs to terminal voltage

References:
    - Kundur, P. (1994). Power System Stability and Control. McGraw-Hill.
    - IEEE Std 421.5-2016: IEEE Recommended Practice for Excitation System Models
    - MIL-STD-1275E: Characteristics of 28 Volt DC Input Power
"""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from simulator.constants import (
    D,
    H,
    IDX_DELTA,
    IDX_ED_DPRIME,
    IDX_ED_PRIME,
    IDX_EQ_DPRIME,
    IDX_EQ_PRIME,
    IDX_OMEGA,
    IDX_VF,
    IDX_VR,
    I_base,
    Ka,
    Ke,
    Kf,
    LOAD_RAMP_DURATION_S,
    N_STATES,
    R_load_nominal,
    Ra,
    Ta,
    Td0_dprime,
    Td0_prime,
    Te,
    Tf,
    Tq0_dprime,
    Tq0_prime,
    V_base,
    Vref,
    X0_NOMINAL,
    X_load_nominal,
    Xd,
    Xd_dprime,
    Xd_prime,
    Xq,
    Xq_dprime,
    Xq_prime,
    omega_s,
    T_ambient_nominal_c,
    thermal_resistance_junction_ambient,
    SIGMA_SENSOR_NOMINAL,
    C_nominal_uf,
)


def _compute_load_impedance(
    t: float,
    scenario_params: dict[str, Any],
    load_events: list[dict[str, Any]],
) -> tuple[float, float]:
    """
    Compute time-varying load impedance R_load(t), X_load(t).

    Purpose:
        Models the electrical load on the generator as a function of time,
        including step changes for weapons engagement, arctic cold start,
        and other scenario-specific load profiles. All transitions use a
        50ms ramp (not instantaneous) to model inductive load dynamics.

    Inputs:
        t: Current simulation time in seconds.
        scenario_params: Dict of scenario-specific parameters from config.
        load_events: List of dicts with keys {time_s, r_load_target, x_load_target}.

    Outputs:
        (R_load, X_load) tuple in Ohms.

    Mathematical basis:
        Linear ramp between load states over LOAD_RAMP_DURATION_S = 50ms.
        R(t) = R_prev + (R_target - R_prev) * clamp((t - t_event) / ramp_duration, 0, 1)
    """
    r_load: float = R_load_nominal
    x_load: float = X_load_nominal

    for event in load_events:
        event_time: float = event["time_s"]
        r_target: float = event["r_load_target"]
        x_target: float = event.get("x_load_target", X_load_nominal)

        if t >= event_time:
            progress: float = min(
                (t - event_time) / LOAD_RAMP_DURATION_S, 1.0
            )
            r_load = r_load + (r_target - r_load) * progress
            x_load = x_load + (x_target - x_load) * progress

    return r_load, x_load


def _avr_dae_rhs(
    t: float,
    x: np.ndarray,
    tm: float,
    r_load_func: Any,
    vref_effective: float,
    ke_effective: float,
    sensor_noise_sigma: float,
) -> np.ndarray:
    """
    Right-hand side of the 8th-order synchronous machine + IEEE Type I AVR DAE.

    Purpose:
        Computes dx/dt for all 8 state variables at time t.

    Inputs:
        t: Current time (s).
        x: State vector [delta, omega, Eq', Ed', Eq'', Ed'', Vf, Vr].
        tm: Mechanical torque input (pu) — assumed constant per timestep.
        r_load_func: Callable(t) -> (R_load, X_load) returning load impedance.
        vref_effective: Effective voltage reference (may include degradation).
        ke_effective: Effective exciter constant (may include degradation).
        sensor_noise_sigma: Standard deviation of measurement noise on Vt (V).

    Outputs:
        dxdt: np.ndarray of shape (8,) — time derivatives of state vector.

    Mathematical basis:
        Differential equations:
            d(delta)/dt = omega_s * omega
            d(omega)/dt = (1/(2H)) * (Tm - Te - D*omega)
            d(Eq')/dt   = (1/Td0') * (-Eq' + (Xd - Xd')*Id + Vf)
            d(Ed')/dt   = (1/Tq0') * (-Ed' - (Xq - Xq')*Iq)
            d(Eq'')/dt  = (1/Td0'') * (-Eq'' + Eq' - (Xd' - Xd'')*Id)
            d(Ed'')/dt  = (1/Tq0'') * (-Ed'' + Ed' + (Xq' - Xq'')*Iq)
            d(Vf)/dt    = (1/Te) * (-Ke*Vf + Vr)
            d(Vr)/dt    = (1/Ta) * (-Vr + Ka*(Vref - Vt - Kf/Tf * Vf))

        Algebraic equations (stator):
            Vd = -Ra*Id - Xq''*Iq + Ed''
            Vq = -Ra*Iq + Xd''*Id + Eq''
            Vt = sqrt(Vd² + Vq²)

        d-q currents from load:
            Id = (Vd*R_load + Vq*X_load) / (R_load² + X_load²)
            Iq = (Vq*R_load - Vd*X_load) / (R_load² + X_load²)
    """
    # Unpack state variables
    delta: float = x[IDX_DELTA]
    omega_dev: float = x[IDX_OMEGA]
    eq_prime: float = x[IDX_EQ_PRIME]
    ed_prime: float = x[IDX_ED_PRIME]
    eq_dprime: float = x[IDX_EQ_DPRIME]
    ed_dprime: float = x[IDX_ED_DPRIME]
    vf: float = x[IDX_VF]
    vr: float = x[IDX_VR]

    # Get time-varying load impedance
    r_load, x_load = r_load_func(t)
    z_sq: float = r_load**2 + x_load**2

    # ─── Algebraic: Solve for d-q currents using iterative approach ─────────
    # Start with subtransient EMFs as initial voltage estimates
    vd_est: float = ed_dprime
    vq_est: float = eq_dprime

    # Iterative solution (2 iterations sufficient for convergence)
    for _ in range(3):
        id_val: float = (vd_est * r_load + vq_est * x_load) / z_sq
        iq_val: float = (vq_est * r_load - vd_est * x_load) / z_sq

        vd_est = -Ra * id_val - Xq_dprime * iq_val + ed_dprime
        vq_est = -Ra * iq_val + Xd_dprime * id_val + eq_dprime

    # Final current values
    id_val = (vd_est * r_load + vq_est * x_load) / z_sq
    iq_val = (vq_est * r_load - vd_est * x_load) / z_sq

    # Terminal voltage
    vd: float = vd_est
    vq: float = vq_est
    vt: float = np.sqrt(vd**2 + vq**2)

    # Add sensor noise for measurement (affects AVR feedback only)
    vt_measured: float = vt
    if sensor_noise_sigma > 0.0:
        vt_measured = vt + np.random.normal(0.0, sensor_noise_sigma / V_base)

    # Electrical torque
    te_val: float = ed_dprime * id_val + eq_dprime * iq_val

    # ─── Differential Equations ──────────────────────────────────────────────
    dxdt: np.ndarray = np.zeros(N_STATES)

    # Rotor dynamics
    dxdt[IDX_DELTA] = omega_s * omega_dev
    dxdt[IDX_OMEGA] = (1.0 / (2.0 * H)) * (tm - te_val - D * omega_dev)

    # Flux dynamics
    dxdt[IDX_EQ_PRIME] = (1.0 / Td0_prime) * (
        -eq_prime + (Xd - Xd_prime) * id_val + vf
    )
    dxdt[IDX_ED_PRIME] = (1.0 / Tq0_prime) * (
        -ed_prime - (Xq - Xq_prime) * iq_val
    )
    dxdt[IDX_EQ_DPRIME] = (1.0 / Td0_dprime) * (
        -eq_dprime + eq_prime - (Xd_prime - Xd_dprime) * id_val
    )
    dxdt[IDX_ED_DPRIME] = (1.0 / Tq0_dprime) * (
        -ed_dprime + ed_prime + (Xq_prime - Xq_dprime) * iq_val
    )

    # IEEE Type I AVR
    dxdt[IDX_VF] = (1.0 / Te) * (-ke_effective * vf + vr)
    dxdt[IDX_VR] = (1.0 / Ta) * (
        -vr + Ka * (vref_effective - vt_measured - (Kf / Tf) * vf)
    )

    return dxdt


def simulate_avr_mission(
    duration_s: float,
    scenario_params: dict[str, Any],
    fault_schedule: list[dict[str, Any]],
    dt: float = 0.1,
    initial_conditions: np.ndarray | None = None,
    degradation_state: dict[str, Any] | None = None,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a complete AVR mission using the 8th-order DAE model.

    Purpose:
        Runs the full synchronous generator + AVR simulation for a specified
        duration under a given scenario, applying faults according to the
        fault schedule. This is the primary data generation function.

    Inputs:
        duration_s: Total simulation duration in seconds.
        scenario_params: Dict of scenario parameters (from scenarios.yaml).
        fault_schedule: List of dicts, each with keys:
            {time_s: float, mechanism: str, severity: float}
            mechanism ∈ {'thyristor', 'capacitor', 'terminal'}
            severity ∈ [0.0, 1.0] (0=healthy, 1=failure)
        dt: Output timestep in seconds (0.1s = 10Hz).
        initial_conditions: Initial state vector, or None for nominal.
        degradation_state: Dict tracking progressive degradation levels, or None.
        save_path: If not None, save output CSVs to this path.

    Outputs:
        (avr_timeseries_df, fault_log_df) where:
        avr_timeseries_df columns:
            timestamp, voltage_v, current_a, temperature_c, delta, omega,
            Eq_prime, Ed_prime, Eq_dprime, Ed_dprime, Vf, Vr, scenario, run_id
        fault_log_df columns:
            timestamp, fault_type, fault_mechanism, severity, duration_ms,
            status, is_operational_transient

    Mathematical basis:
        Solves the DAE system defined in _avr_dae_rhs using scipy's Radau solver
        (implicit Runge-Kutta of order 5, suitable for stiff systems).
        Output is downsampled to dt resolution after solving at adaptive timesteps.
    """
    # ─── Initialize ──────────────────────────────────────────────────────────
    if initial_conditions is not None:
        x0: np.ndarray = np.array(initial_conditions, dtype=np.float64)
    else:
        x0 = np.array(X0_NOMINAL, dtype=np.float64)

    if degradation_state is None:
        degradation_state = {
            "thyristor_level": 0.0,
            "capacitor_level": 0.0,
            "terminal_miner_damage": 0.0,
            "thermal_cycles": 0.0,
        }

    scenario_name: str = scenario_params.get("scenario_name", "unknown")
    run_id: int = scenario_params.get("run_id", 0)
    ambient_temp_c: float = scenario_params.get(
        "ambient_temp_c", T_ambient_nominal_c
    )

    # Build load event schedule from scenario params
    load_events: list[dict[str, Any]] = _build_load_events(scenario_params)

    # ─── Solve the ODE in chunks ─────────────────────────────────────────────
    # We solve in chunks of 1 second to allow fault injection between chunks
    n_output_steps: int = int(duration_s / dt)
    t_output: np.ndarray = np.linspace(0.0, duration_s, n_output_steps + 1)

    # Pre-allocate output arrays
    state_history: np.ndarray = np.zeros((len(t_output), N_STATES))
    vt_history: np.ndarray = np.zeros(len(t_output))
    current_history: np.ndarray = np.zeros(len(t_output))

    # Fault log accumulator
    fault_log_entries: list[dict[str, Any]] = []

    # Mechanical torque (pu) — assumed constant at ~1.0 pu for nominal load
    tm: float = 1.0

    # Active degradation parameters
    ke_eff: float = Ke
    sensor_noise_sigma: float = 0.0
    ripple_amplitude: float = 0.0
    vref_eff: float = Vref

    # Chunk-based solving
    chunk_duration_s: float = 1.0
    n_chunks: int = int(np.ceil(duration_s / chunk_duration_s))
    current_x: np.ndarray = x0.copy()
    output_idx: int = 0

    for chunk_i in range(n_chunks):
        t_start: float = chunk_i * chunk_duration_s
        t_end: float = min((chunk_i + 1) * chunk_duration_s, duration_s)

        # ─── Apply fault schedule for this chunk ─────────────────────────────
        for fault in fault_schedule:
            if t_start <= fault["time_s"] < t_end:
                mechanism: str = fault["mechanism"]
                severity: float = fault["severity"]

                if mechanism == "thyristor":
                    degradation_state["thyristor_level"] = severity
                    ke_eff = Ke * (1.0 - 0.3 * severity)
                elif mechanism == "capacitor":
                    degradation_state["capacitor_level"] = severity
                    c_effective: float = C_nominal_uf * (1.0 - 0.6 * severity)
                    ripple_amplitude = 0.75 * (C_nominal_uf / max(c_effective, 1.0))
                elif mechanism == "terminal":
                    degradation_state["terminal_miner_damage"] = severity
                    sensor_noise_sigma = SIGMA_SENSOR_NOMINAL * (
                        1.0 + 9.0 * severity
                    )

                fault_log_entries.append({
                    "timestamp": t_start + (fault["time_s"] - t_start),
                    "fault_type": _severity_to_type(severity),
                    "fault_mechanism": mechanism,
                    "severity": severity,
                    "duration_ms": 0.0,
                    "status": "injected",
                    "is_operational_transient": False,
                })

        # ─── Build load function for this chunk ──────────────────────────────
        def _r_load_func(
            t: float,
            _params: dict[str, Any] = scenario_params,
            _events: list[dict[str, Any]] = load_events,
        ) -> tuple[float, float]:
            return _compute_load_impedance(t, _params, _events)

        # ─── Solve ODE on this chunk ─────────────────────────────────────────
        chunk_t_eval: np.ndarray = t_output[
            (t_output >= t_start) & (t_output <= t_end)
        ]
        if len(chunk_t_eval) == 0:
            continue

        sol = solve_ivp(
            fun=lambda t, x: _avr_dae_rhs(
                t, x, tm, _r_load_func, vref_eff, ke_eff, sensor_noise_sigma
            ),
            t_span=(t_start, t_end),
            y0=current_x,
            method="Radau",
            t_eval=chunk_t_eval,
            rtol=1e-4,
            atol=1e-6,
            max_step=0.05,
        )

        if not sol.success:
            # If solver fails, use last known state and fill
            n_fill: int = len(chunk_t_eval)
            for fi in range(n_fill):
                if output_idx + fi < len(t_output):
                    state_history[output_idx + fi] = current_x
            output_idx += n_fill
            continue

        # Clamp state to prevent numerical drift
        sol_y = sol.y.copy()
        sol_y = np.clip(sol_y, -1e6, 1e6)
        sol_y[np.isnan(sol_y)] = 0.0

        # Store results
        n_points: int = sol_y.shape[1]
        for pi in range(n_points):
            if output_idx < len(t_output):
                state_history[output_idx] = sol_y[:, pi]

                # Compute terminal voltage and current
                r_l, x_l = _r_load_func(sol.t[pi])
                z_sq_local: float = r_l**2 + x_l**2
                ed_dp: float = sol_y[IDX_ED_DPRIME, pi]
                eq_dp: float = sol_y[IDX_EQ_DPRIME, pi]

                # Iterative solve for Vd, Vq
                vd_local: float = ed_dp
                vq_local: float = eq_dp
                for _ in range(3):
                    id_l: float = (vd_local * r_l + vq_local * x_l) / z_sq_local
                    iq_l: float = (vq_local * r_l - vd_local * x_l) / z_sq_local
                    vd_local = -Ra * id_l - Xq_dprime * iq_l + ed_dp
                    vq_local = -Ra * iq_l + Xd_dprime * id_l + eq_dp

                vt_pu: float = np.sqrt(vd_local**2 + vq_local**2)
                vt_history[output_idx] = vt_pu * V_base

                # Add capacitor ripple if degraded
                if ripple_amplitude > 0.0:
                    f_sw: float = 10000.0  # 10kHz switching frequency
                    ripple_v: float = ripple_amplitude * np.sin(
                        2.0 * np.pi * f_sw * sol.t[pi]
                    )
                    vt_history[output_idx] += ripple_v

                # Current magnitude
                id_l = (vd_local * r_l + vq_local * x_l) / z_sq_local
                iq_l = (vq_local * r_l - vd_local * x_l) / z_sq_local
                i_pu: float = np.sqrt(id_l**2 + iq_l**2)
                current_history[output_idx] = i_pu * I_base

                output_idx += 1

        # Update state for next chunk
        current_x = sol_y[:, -1].copy()

    # ─── Compute temperature trace ───────────────────────────────────────────
    power_dissipated: np.ndarray = vt_history * current_history / 1000.0
    temperature_c: np.ndarray = (
        ambient_temp_c
        + power_dissipated * thermal_resistance_junction_ambient
    )

    # Apply scenario temperature ramp if specified
    if "temp_ramp_c_per_hour" in scenario_params:
        ramp_rate: float = scenario_params["temp_ramp_c_per_hour"]
        temperature_c += ramp_rate * t_output[:output_idx] / 3600.0

    # ─── Build output DataFrames ─────────────────────────────────────────────
    n_actual: int = min(output_idx, len(t_output))
    avr_timeseries_df: pd.DataFrame = pd.DataFrame({
        "timestamp": t_output[:n_actual],
        "voltage_v": vt_history[:n_actual],
        "current_a": current_history[:n_actual],
        "temperature_c": temperature_c[:n_actual],
        "delta": state_history[:n_actual, IDX_DELTA],
        "omega": state_history[:n_actual, IDX_OMEGA],
        "Eq_prime": state_history[:n_actual, IDX_EQ_PRIME],
        "Ed_prime": state_history[:n_actual, IDX_ED_PRIME],
        "Eq_dprime": state_history[:n_actual, IDX_EQ_DPRIME],
        "Ed_dprime": state_history[:n_actual, IDX_ED_DPRIME],
        "Vf": state_history[:n_actual, IDX_VF],
        "Vr": state_history[:n_actual, IDX_VR],
        "scenario": scenario_name,
        "run_id": run_id,
    })

    fault_log_df: pd.DataFrame = pd.DataFrame(
        fault_log_entries,
        columns=[
            "timestamp", "fault_type", "fault_mechanism", "severity",
            "duration_ms", "status", "is_operational_transient",
        ],
    )
    if fault_log_df.empty:
        fault_log_df = pd.DataFrame(columns=[
            "timestamp", "fault_type", "fault_mechanism", "severity",
            "duration_ms", "status", "is_operational_transient",
        ])

    # ─── Save to disk if requested ───────────────────────────────────────────
    if save_path is not None:
        import os

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        avr_timeseries_df.to_csv(save_path, index=False)
        fault_path: str = save_path.replace("avr_data_", "fault_log_")
        fault_log_df.to_csv(fault_path, index=False)

    return avr_timeseries_df, fault_log_df


def _build_load_events(
    scenario_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build load change event schedule from scenario parameters.

    Purpose:
        Translates scenario-level parameters (e.g., weapons active, arctic cold)
        into timestamped load impedance change events.

    Inputs:
        scenario_params: Dict from scenarios.yaml for a specific scenario.

    Outputs:
        List of load event dicts with {time_s, r_load_target, x_load_target}.
    """
    from simulator.constants import R_load_arctic, R_load_weapons

    events: list[dict[str, Any]] = []
    duration_min: float = scenario_params.get("duration_minutes", 30)
    duration_s: float = duration_min * 60.0

    scenario_name: str = scenario_params.get("scenario_name", "")

    if scenario_name == "weapons_active":
        # Weapons engage at random times
        rng: np.random.Generator = np.random.default_rng(42)
        load_dump_per_hour: int = scenario_params.get(
            "load_dump_events_per_hour", 4
        )
        n_events: int = max(
            1, int(load_dump_per_hour * duration_min / 60.0)
        )
        event_times: np.ndarray = rng.uniform(
            10.0, duration_s - 10.0, size=n_events
        )
        event_times.sort()
        for et in event_times:
            events.append({
                "time_s": float(et),
                "r_load_target": R_load_weapons,
                "x_load_target": X_load_nominal,
            })

    elif scenario_name == "arctic_cold":
        # Start with high impedance, ramp down as engine warms
        events.append({
            "time_s": 0.0,
            "r_load_target": R_load_arctic,
            "x_load_target": X_load_nominal,
        })
        # Gradual ramp back to nominal over 10 minutes
        n_ramp_steps: int = 10
        for ri in range(1, n_ramp_steps + 1):
            frac: float = ri / n_ramp_steps
            events.append({
                "time_s": frac * 600.0,
                "r_load_target": R_load_arctic
                + (R_load_nominal - R_load_arctic) * frac,
                "x_load_target": X_load_nominal,
            })

    return events


def _severity_to_type(severity: float) -> str:
    """
    Map numerical severity to categorical fault type label.

    Inputs:
        severity: Float in [0, 1].

    Outputs:
        String label: 'healthy', 'incipient', 'developing', or 'critical'.
    """
    if severity < 0.1:
        return "healthy"
    elif severity < 0.4:
        return "incipient"
    elif severity < 0.7:
        return "developing"
    else:
        return "critical"



def run_tests() -> None:
    """Sanity checks for the DAE model."""
    # Test 1: Nominal simulation produces reasonable voltage
    params: dict[str, Any] = {
        "scenario_name": "baseline",
        "run_id": 0,
        "ambient_temp_c": 25.0,
        "duration_minutes": 0.1,  # 6 seconds
    }
    df, fault_df = simulate_avr_mission(
        duration_s=6.0,
        scenario_params=params,
        fault_schedule=[],
        dt=0.1,
    )
    assert len(df) > 0, "Simulation produced no output"
    mean_v: float = df["voltage_v"].mean()
    assert 20.0 < mean_v < 36.0, (
        f"Mean voltage {mean_v:.2f}V is outside reasonable range [20, 36]V"
    )

    # Test 2: State vector has correct dimensions
    assert "delta" in df.columns, "Missing 'delta' column"
    assert "Vr" in df.columns, "Missing 'Vr' column"
    assert df.shape[1] == 14, f"Expected 14 columns, got {df.shape[1]}"

    print("[PASS] simulator/dae_model.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
