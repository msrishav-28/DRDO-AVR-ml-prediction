"""
Setup script for the avr_phm package.

Physics-Informed Digital Twin for Military AVR Fault Prognostics.
AVR-PHM: MIL-STD-Aligned Synthetic Data + PINN Early Warning System.
"""

from setuptools import setup, find_packages


setup(
    name="avr_phm",
    version="0.1.0",
    description=(
        "Physics-Informed Digital Twin for Military AVR Fault Prognostics — "
        "MIL-STD-1275E/810H compliant synthetic data generation, WGAN-GP "
        "augmentation, and multi-task PINN early warning system."
    ),
    author="AVR-PHM Research Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.0",
        "deepxde==1.11.2",
        "scipy==1.14.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "shap==0.46.0",
        "wandb==0.18.5",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "pyyaml==6.0.2",
        "tqdm==4.66.5",
        "imbalanced-learn==0.12.3",
        "pytest==8.3.3",
        "pytorch-lightning==2.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
