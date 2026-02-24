"""
Data generation pipeline orchestrator.

Generates the complete synthetic dataset by running all scenario simulations
in the correct order with resume logic. Each run produces two CSV files:
    avr_data_{scenario}_run{run_id}.csv  — timeseries data
    fault_log_{scenario}_run{run_id}.csv — fault event log

Generation order (baseline first, then ascending fault complexity):
    1. baseline       (4 × 120 min runs)
    2. arctic_cold    (2 × 30 min runs)
    3. rough_terrain  (2 × 30 min runs)
    4. desert_heat    (2 × 30 min runs)
    5. artillery_firing (2 × 30 min runs)
    6. weapons_active (2 × 30 min runs)
    7. emp_simulation (2 × 30 min runs)

Total expected dataset size: ~350,000 samples across all scenarios.

Progressive degradation runs:
    - Baseline runs 3 and 4: full degradation trajectory
    - One run from each combat scenario: full degradation trajectory
"""

import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from config import load_yaml, resolve_path
from simulator.scenario_engine import simulate_scenario
from simulator.validator import validate_fault_log, validate_timeseries


# Generation order and progressive degradation assignment
GENERATION_ORDER: list[dict[str, Any]] = [
    # Baseline: 4 × 120 min, runs 3-4 are progressive degradation
    {"scenario": "baseline", "run_id": 1, "progressive": False},
    {"scenario": "baseline", "run_id": 2, "progressive": False},
    {"scenario": "baseline", "run_id": 3, "progressive": True},
    {"scenario": "baseline", "run_id": 4, "progressive": True},
    # Arctic cold: 2 × 30 min, run 2 is progressive
    {"scenario": "arctic_cold", "run_id": 1, "progressive": False},
    {"scenario": "arctic_cold", "run_id": 2, "progressive": True},
    # Rough terrain: 2 × 30 min, run 2 is progressive
    {"scenario": "rough_terrain", "run_id": 1, "progressive": False},
    {"scenario": "rough_terrain", "run_id": 2, "progressive": True},
    # Desert heat: 2 × 30 min, run 2 is progressive
    {"scenario": "desert_heat", "run_id": 1, "progressive": False},
    {"scenario": "desert_heat", "run_id": 2, "progressive": True},
    # Artillery firing: 2 × 30 min, run 2 is progressive
    {"scenario": "artillery_firing", "run_id": 1, "progressive": False},
    {"scenario": "artillery_firing", "run_id": 2, "progressive": True},
    # Weapons active: 2 × 30 min, run 2 is progressive
    {"scenario": "weapons_active", "run_id": 1, "progressive": False},
    {"scenario": "weapons_active", "run_id": 2, "progressive": True},
    # EMP simulation: 2 × 30 min, run 2 is progressive
    {"scenario": "emp_simulation", "run_id": 1, "progressive": False},
    {"scenario": "emp_simulation", "run_id": 2, "progressive": True},
]


def generate_with_resume(
    scenario: str,
    run_id: int,
    output_dir: str,
    scenario_params: dict[str, Any],
    progressive: bool = False,
    seed: int = 42,
) -> str:
    """
    Generate a single scenario run with resume support.

    Purpose:
        Generates one simulation run and saves to disk. If the output
        file already exists, skips generation (resume logic).

    Inputs:
        scenario: Scenario name from scenarios.yaml.
        run_id: Integer run identifier.
        output_dir: Directory to save output CSVs.
        scenario_params: Scenario configuration dict.
        progressive: Whether to use progressive degradation.
        seed: Random seed for this run.

    Outputs:
        Path to the generated/existing avr_data CSV file.

    Mathematical basis:
        N/A — orchestration function.
    """
    output_path: str = os.path.join(
        output_dir,
        f"avr_data_{scenario}_run{run_id}.csv",
    )

    if os.path.exists(output_path):
        file_size: int = os.path.getsize(output_path)
        if file_size > 100:
            print(f"[SKIP] {output_path} already exists ({file_size} bytes). "
                  f"Delete to regenerate.")
            return output_path

    print(f"[GENERATE] {scenario} run {run_id} "
          f"(progressive={progressive})...")

    avr_df, fault_df = simulate_scenario(
        scenario_name=scenario,
        scenario_params=scenario_params,
        run_id=run_id,
        progressive_degradation=progressive,
        seed=seed + run_id * 1000,
        save_dir=output_dir,
    )

    # Validate outputs
    ts_result = validate_timeseries(avr_df, strict=False)
    fl_result = validate_fault_log(fault_df)

    if not ts_result.all_passed:
        print(f"[WARN] Validation issues for {scenario} run {run_id}:")
        print(ts_result.summary())

    print(f"[DONE] {scenario} run {run_id}: "
          f"{len(avr_df)} samples, {len(fault_df)} fault events")

    return output_path


def generate_full_dataset(
    config_path: str = "config/scenarios.yaml",
    output_dir: str = "data/raw/",
    resume: bool = True,
) -> None:
    """
    Orchestrate generation of all scenarios and runs.

    Purpose:
        Main entry point for dataset generation. Runs all scenarios
        in the specified order with resume support.

    Inputs:
        config_path: Path to scenarios.yaml (relative to package root).
        output_dir: Output directory for CSV files (relative to package root).
        resume: If True, skip runs whose output files already exist.

    Outputs:
        None. All data is written to disk.

    CRITICAL: resume=True means check which files already exist and skip them.
    This allows interrupted generation to be resumed without restarting.

    File naming convention:
        avr_data_{scenario}_{run_id}.csv
        fault_log_{scenario}_{run_id}.csv

    After each run: save to disk immediately. Log stats.
    Total expected dataset size: ~350,000 samples across all scenarios.
    """
    # Load configuration
    scenarios_cfg: dict[str, Any] = load_yaml("scenarios")
    paths_cfg: dict[str, Any] = load_yaml("paths")

    # Resolve output directory
    abs_output_dir: str = str(resolve_path(
        paths_cfg.get("data", {}).get("raw_dir", output_dir)
    ))
    os.makedirs(abs_output_dir, exist_ok=True)

    total_samples: int = 0
    total_faults: int = 0

    print("=" * 60)
    print("AVR-PHM DATASET GENERATION PIPELINE")
    print(f"Output directory: {abs_output_dir}")
    print(f"Resume mode: {resume}")
    print("=" * 60)

    for i, gen_spec in enumerate(GENERATION_ORDER):
        scenario: str = gen_spec["scenario"]
        run_id: int = gen_spec["run_id"]
        progressive: bool = gen_spec["progressive"]

        # Get scenario-specific parameters
        if scenario not in scenarios_cfg["scenarios"]:
            print(f"[ERROR] Scenario '{scenario}' not found in config")
            continue

        scenario_params: dict[str, Any] = {
            **scenarios_cfg["global"],
            **scenarios_cfg["scenarios"][scenario],
        }

        # Check if we should skip (resume logic)
        output_path: str = os.path.join(
            abs_output_dir,
            f"avr_data_{scenario}_run{run_id}.csv",
        )
        if resume and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 100:
                print(f"[{i+1}/{len(GENERATION_ORDER)}] "
                      f"[SKIP] {scenario} run {run_id} (exists)")
                try:
                    existing_df: pd.DataFrame = pd.read_csv(output_path)
                    total_samples += len(existing_df)
                except Exception:
                    pass
                continue

        print(f"\n[{i+1}/{len(GENERATION_ORDER)}] "
              f"Generating: {scenario} run {run_id}...")

        seed: int = 42 + i * 1000 + run_id

        try:
            generate_with_resume(
                scenario=scenario,
                run_id=run_id,
                output_dir=abs_output_dir,
                scenario_params=scenario_params,
                progressive=progressive,
                seed=seed,
            )

            # Count samples
            result_path: str = os.path.join(
                abs_output_dir,
                f"avr_data_{scenario}_run{run_id}.csv",
            )
            if os.path.exists(result_path):
                result_df: pd.DataFrame = pd.read_csv(result_path)
                total_samples += len(result_df)

            fault_path: str = os.path.join(
                abs_output_dir,
                f"fault_log_{scenario}_run{run_id}.csv",
            )
            if os.path.exists(fault_path):
                fault_df: pd.DataFrame = pd.read_csv(fault_path)
                total_faults += len(fault_df)

        except Exception as e:
            print(f"[ERROR] Failed to generate {scenario} run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"Total samples: {total_samples:,}")
    print(f"Total fault events: {total_faults:,}")
    print("=" * 60)


def run_tests() -> None:
    """Sanity checks for the pipeline module."""
    # Test 1: Generation order has correct count
    assert len(GENERATION_ORDER) == 16, (
        f"Expected 16 generation specs (4 baseline + 12 combat), "
        f"got {len(GENERATION_ORDER)}"
    )

    # Test 2: All scenarios in generation order exist in config
    scenarios_cfg = load_yaml("scenarios")
    for spec in GENERATION_ORDER:
        assert spec["scenario"] in scenarios_cfg["scenarios"], (
            f"Scenario '{spec['scenario']}' not in scenarios.yaml"
        )

    # Test 3: Progressive degradation assignment is correct
    baseline_prog: list[bool] = [
        s["progressive"]
        for s in GENERATION_ORDER
        if s["scenario"] == "baseline"
    ]
    assert sum(baseline_prog) == 2, (
        "Exactly 2 baseline runs should be progressive degradation"
    )

    print("[PASS] data_gen/pipeline.py — all tests passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AVR-PHM data generation pipeline"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run sanity checks instead of generating data",
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        generate_full_dataset()
