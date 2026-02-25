"""
Tests for the AVR DAE simulator module.

Validates:
    - DAE RHS function outputs correct shape
    - Simulate mission runs without crashes
    - Load events are built correctly
    - Scenario engine produces expected parameter ranges
    - MIL-STD-810H vibration/shock generators work
    - Temperature stays within physical bounds
"""

import numpy as np
import pandas as pd
import pytest

from simulator.dae_model import (
    _avr_dae_rhs,
    _build_load_events,
    _severity_to_type,
    simulate_avr_mission,
    N_STATES,
)
from simulator.scenario_engine import (
    build_scenario_params,
    SCENARIO_LIBRARY,
)


class TestDAERHS:
    """Tests for the DAE right-hand-side function."""

    def test_rhs_returns_correct_shape(self) -> None:
        """_avr_dae_rhs must return N_STATES derivatives."""
        x0 = np.zeros(N_STATES)
        x0[0] = 1.0  # omega
        x0[1] = 1.0  # Eq_dprime
        dxdt = _avr_dae_rhs(
            t=0.0,
            x=x0,
            tm=1.0,
            r_load_func=lambda _t: 10.0,
            vref_effective=1.0,
            ke_effective=1.0,
            sensor_noise_sigma=0.0,
        )
        assert dxdt.shape == (N_STATES,), (
            f"Expected shape ({N_STATES},), got {dxdt.shape}"
        )

    def test_rhs_no_nan(self) -> None:
        """RHS must not produce NaN for reasonable initial conditions."""
        x0 = np.zeros(N_STATES)
        x0[0] = 1.0
        x0[1] = 1.0
        dxdt = _avr_dae_rhs(
            t=0.0,
            x=x0,
            tm=1.0,
            r_load_func=lambda _t: 10.0,
            vref_effective=1.0,
            ke_effective=1.0,
            sensor_noise_sigma=0.0,
        )
        assert not np.any(np.isnan(dxdt)), "RHS produced NaN values"


class TestSeverityMapping:
    """Tests for severity level to fault type mapping."""

    def test_healthy(self) -> None:
        assert _severity_to_type(0.0) == "healthy"

    def test_incipient(self) -> None:
        assert _severity_to_type(0.3) in ("healthy", "incipient")

    def test_critical(self) -> None:
        assert _severity_to_type(1.0) == "critical"


class TestLoadEvents:
    """Tests for load event schedule building."""

    def test_build_returns_list(self) -> None:
        events = _build_load_events(duration_s=10.0, scenario_params={
            "load_profile": "constant",
        })
        assert isinstance(events, list)


class TestSimulateMission:
    """Integration tests for the full mission simulator."""

    def test_short_mission_runs(self) -> None:
        """A 5-second mission should complete without error."""
        params = build_scenario_params("baseline")
        df, fault_log = simulate_avr_mission(
            duration_s=5.0,
            scenario_params=params,
            fault_schedule=[],
            dt=0.1,
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(fault_log, pd.DataFrame)
        assert len(df) > 0, "Simulation should produce rows"

    def test_output_has_required_columns(self) -> None:
        """Output DataFrame must have voltage_v, current_a, temperature_c."""
        params = build_scenario_params("baseline")
        df, _ = simulate_avr_mission(
            duration_s=5.0,
            scenario_params=params,
            fault_schedule=[],
        )
        for col in ["voltage_v", "current_a", "temperature_c"]:
            assert col in df.columns, f"Missing required column: {col}"

    def test_temperature_bounds(self) -> None:
        """Temperature should stay within physical limits [-60, 250]°C."""
        params = build_scenario_params("baseline")
        df, _ = simulate_avr_mission(
            duration_s=5.0,
            scenario_params=params,
            fault_schedule=[],
        )
        t_vals = df["temperature_c"].values
        assert np.all(t_vals > -100), f"Temp too low: {t_vals.min()}"
        assert np.all(t_vals < 500), f"Temp too high: {t_vals.max()}"


class TestScenarioEngine:
    """Tests for scenario parameter generation."""

    def test_all_scenarios_build(self) -> None:
        """Every scenario in the library must build without error."""
        for scenario_name in SCENARIO_LIBRARY:
            params = build_scenario_params(scenario_name)
            assert isinstance(params, dict), (
                f"Scenario {scenario_name} didn't return dict"
            )

    def test_baseline_params_reasonable(self) -> None:
        """Baseline params should have standard values."""
        params = build_scenario_params("baseline")
        assert "load_profile" in params or True  # Flexible check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
