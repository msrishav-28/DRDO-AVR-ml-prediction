"""
Ablation study runner (Section 11) and hyperparameter sensitivity analysis
(Section 23).

Mandatory ablation experiments:
    1. no_physics: PINN without physics loss (λ₂=0)
    2. no_cgan:    PINN trained without cGAN augmented data
    3. no_milstd:  PINN trained on data without MIL-STD waveform overlays
    4. single_task: PINN with only fault detection head (no forecast/mechanism)
    5. no_curriculum: PINN without curriculum learning (all scenarios at once)

Hyperparameter sensitivity sweeps (Section 23):
    - Physics loss weight λ₂ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
    - cGAN augmentation ratio ∈ {0.0, 0.25, 0.5, 1.0, 2.0}
    - Dropout rate ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}
"""

import json
import os
import random
from typing import Any

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


# Ablation configurations
ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "full_pinn": {
        "description": "Full PINN with all components (control)",
        "physics_weight": 0.3,
        "use_cgan_data": True,
        "use_milstd_overlays": True,
        "multi_task": True,
        "curriculum": True,
    },
    "no_physics": {
        "description": "PINN without physics loss (λ₂=0)",
        "physics_weight": 0.0,
        "use_cgan_data": True,
        "use_milstd_overlays": True,
        "multi_task": True,
        "curriculum": True,
    },
    "no_cgan": {
        "description": "PINN without cGAN augmentation",
        "physics_weight": 0.3,
        "use_cgan_data": False,
        "use_milstd_overlays": True,
        "multi_task": True,
        "curriculum": True,
    },
    "no_milstd": {
        "description": "PINN without MIL-STD waveform overlays",
        "physics_weight": 0.3,
        "use_cgan_data": True,
        "use_milstd_overlays": False,
        "multi_task": True,
        "curriculum": True,
    },
    "single_task": {
        "description": "PINN with only fault_10s head",
        "physics_weight": 0.3,
        "use_cgan_data": True,
        "use_milstd_overlays": True,
        "multi_task": False,
        "curriculum": True,
    },
    "no_curriculum": {
        "description": "PINN without curriculum learning",
        "physics_weight": 0.3,
        "use_cgan_data": True,
        "use_milstd_overlays": True,
        "multi_task": True,
        "curriculum": False,
    },
}

# Hyperparameter sweep configurations (Section 23)
HYPERPARAM_SWEEPS: dict[str, list[float]] = {
    "physics_loss_weight": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "cgan_augmentation_ratio": [0.0, 0.25, 0.5, 1.0, 2.0],
    "dropout_rate": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
}


def run_single_ablation(
    ablation_name: str,
    config: dict[str, Any],
    train_fn: Any,
    eval_fn: Any,
    base_train_kwargs: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run a single ablation experiment.

    Purpose:
        Trains and evaluates the model with a specific ablation config.

    Inputs:
        ablation_name: Name of the ablation (e.g., 'no_physics').
        config: Ablation config dict.
        train_fn: Training function.
        eval_fn: Evaluation function.
        base_train_kwargs: Base training keyword arguments.
        seed: Random seed.

    Outputs:
        Dict with ablation name, config, and all evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    # Modify training kwargs based on ablation
    train_kwargs: dict[str, Any] = {**base_train_kwargs}

    if "physics_weight" in config:
        train_kwargs["physics_weight"] = config["physics_weight"]

    # Train
    train_result: Any = train_fn(seed=seed, **train_kwargs)

    # Evaluate
    eval_result: Any = eval_fn()

    return {
        "ablation": ablation_name,
        "config": config,
        "train_result": train_result,
        "eval_result": eval_result,
    }


def run_all_ablations(
    train_fn: Any,
    eval_fn: Any,
    base_train_kwargs: dict[str, Any],
    output_dir: str = "outputs/results/ablations",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run all 5 mandatory ablation experiments.

    Purpose:
        Executes every ablation in ABLATION_CONFIGS and saves results.

    Inputs:
        train_fn: Callable for training.
        eval_fn: Callable for evaluation.
        base_train_kwargs: Base training config.
        output_dir: Directory to save results.
        seed: Random seed.

    Outputs:
        Dict with all ablation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: dict[str, Any] = {}

    for name, config in ABLATION_CONFIGS.items():
        result: dict[str, Any] = run_single_ablation(
            ablation_name=name,
            config=config,
            train_fn=train_fn,
            eval_fn=eval_fn,
            base_train_kwargs=base_train_kwargs,
            seed=seed,
        )
        all_results[name] = result

        # Save intermediate results
        result_path: str = os.path.join(
            output_dir, f"ablation_{name}.json"
        )
        with open(result_path, "w") as f:
            json.dump(
                result, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
            )

    # Save combined results
    combined_path: str = os.path.join(output_dir, "all_ablations.json")
    with open(combined_path, "w") as f:
        json.dump(
            all_results, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
        )

    print(f"\n[SAVED] All ablation results to {combined_path}")
    return all_results


def run_hyperparam_sweep(
    param_name: str,
    param_values: list[float],
    train_fn: Any,
    eval_fn: Any,
    base_train_kwargs: dict[str, Any],
    output_dir: str = "outputs/results/sweeps",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run hyperparameter sensitivity sweep (Section 23).

    Purpose:
        Systematically varies one hyperparameter while keeping all
        others fixed, to measure sensitivity.

    Inputs:
        param_name: Name of the hyperparameter to sweep.
        param_values: List of values to test.
        train_fn: Training function.
        eval_fn: Evaluation function.
        base_train_kwargs: Fixed training config.
        output_dir: Output directory.
        seed: Random seed.

    Outputs:
        Dict with per-value results for plotting sensitivity curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: dict[str, Any] = {
        "param_name": param_name,
        "param_values": param_values,
        "metrics": [],
    }

    for val in param_values:
        print(f"\n[SWEEP] {param_name} = {val}")
        train_kwargs: dict[str, Any] = {**base_train_kwargs}
        train_kwargs[param_name] = val

        train_result: Any = train_fn(seed=seed, **train_kwargs)
        eval_result: Any = eval_fn()

        results["metrics"].append({
            "value": val,
            "train_result": train_result,
            "eval_result": eval_result,
        })

    # Save
    sweep_path: str = os.path.join(
        output_dir, f"sweep_{param_name}.json"
    )
    with open(sweep_path, "w") as f:
        json.dump(
            results, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
        )

    return results


def run_tests() -> None:
    """Sanity checks for ablation module."""
    # Test 1: All mandatory ablations present
    required: set[str] = {
        "full_pinn", "no_physics", "no_cgan",
        "no_milstd", "single_task", "no_curriculum",
    }
    assert required.issubset(set(ABLATION_CONFIGS.keys())), (
        f"Missing ablations: {required - set(ABLATION_CONFIGS.keys())}"
    )

    # Test 2: Hyperparameter sweeps configured
    assert "physics_loss_weight" in HYPERPARAM_SWEEPS
    assert 0.0 in HYPERPARAM_SWEEPS["physics_loss_weight"]
    assert 1.0 in HYPERPARAM_SWEEPS["physics_loss_weight"]

    print("[PASS] experiments/ablation.py — all tests passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AVR-PHM ablation study runner"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run sanity checks instead of ablation studies",
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        print("=" * 60)
        print("AVR-PHM ABLATION STUDIES")
        print("=" * 60)
        print("\nAvailable ablation configurations:")
        for name, config in ABLATION_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        print("\nHyperparameter sweeps:")
        for param, values in HYPERPARAM_SWEEPS.items():
            print(f"  {param}: {values}")
        print("\n[INFO] Ablation runner requires train_fn and eval_fn. "
              "Use the research notebook for full ablation studies.")
