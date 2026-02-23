"""
AVR-PHM: Physics-Informed Digital Twin for Military AVR Fault Prognostics.

MIL-STD-Aligned Synthetic Data + PINN Early Warning System.

This package implements a complete research pipeline for predictive health
management of military Automatic Voltage Regulators (AVRs) on 28V DC bus
systems, compliant with MIL-STD-1275E and MIL-STD-810H.

Modules:
    config      - YAML configuration loading and path resolution
    simulator   - Physics-based DAE simulator with MIL-STD waveforms
    data_gen    - Data generation pipeline and WGAN-GP augmentation
    features    - Feature engineering with physics-derived features
    models      - PINN and baseline models
    eval        - PHM-grade evaluation metrics and XAI
    experiments - Training harness, ablation studies, evaluation
"""

__version__: str = "0.1.0"
