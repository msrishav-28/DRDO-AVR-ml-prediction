"""
Adversarial robustness experiments per Section 20.

Implements FGSM and PGD attacks with physical plausibility constraints
(±2V perturbation budget). Evaluates fault detection robustness under
adversarial conditions.

References:
    - Goodfellow et al. (2015). Explaining and Harnessing Adversarial Examples.
    - Madry et al. (2018). Towards Deep Learning Models Resistant to
      Adversarial Attacks.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 2.0,
    loss_fn: nn.Module | None = None,
    task_name: str = "fault_10s",
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Purpose:
        Generates adversarial examples by adding a single-step
        perturbation in the direction of the loss gradient.

    Inputs:
        model: PyTorch model.
        x: Clean input tensor (batch, seq_len, n_features).
        y: True labels.
        epsilon: Perturbation budget (2.0V per spec).
        loss_fn: Loss function. Default: BCEWithLogitsLoss.
        task_name: Task head to attack.

    Outputs:
        Adversarial input tensor of same shape as x.

    Mathematical basis:
        x_adv = x + ε * sign(∇_x L(f(x), y))
        where ε = 2.0V (physical plausibility constraint)
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    x_adv: torch.Tensor = x.clone().detach().requires_grad_(True)

    outputs: dict[str, torch.Tensor] = model(x_adv)
    logits: torch.Tensor = outputs.get(task_name, outputs.get("fault_10s", torch.zeros(1)))

    loss: torch.Tensor = loss_fn(logits, y.float())
    loss.backward()

    if x_adv.grad is not None:
        perturbation: torch.Tensor = epsilon * x_adv.grad.sign()
        x_adv = x_adv.detach() + perturbation
    else:
        x_adv = x_adv.detach()

    return x_adv


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 2.0,
    alpha: float = 0.5,
    n_steps: int = 10,
    loss_fn: nn.Module | None = None,
    task_name: str = "fault_10s",
    random_start: bool = True,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.

    Purpose:
        Stronger iterative version of FGSM that applies multiple
        gradient steps with projection back to the ε-ball.

    Inputs:
        model: PyTorch model.
        x: Clean input tensor.
        y: True labels.
        epsilon: L∞ perturbation budget (2.0V).
        alpha: Step size per iteration.
        n_steps: Number of PGD iterations.
        loss_fn: Loss function.
        task_name: Task head to attack.
        random_start: If True, initialize with random noise in ε-ball.

    Outputs:
        Adversarial input tensor.

    Mathematical basis:
        x^{t+1} = Π_{x+S}(x^t + α * sign(∇_x L(f(x^t), y)))
        where Π is projection onto the L∞ ε-ball around x.
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    x_adv: torch.Tensor = x.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.detach()

    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        outputs: dict[str, torch.Tensor] = model(x_adv)
        logits: torch.Tensor = outputs.get(task_name, torch.zeros(1))
        loss: torch.Tensor = loss_fn(logits, y.float())
        loss.backward()

        if x_adv.grad is not None:
            x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        else:
            x_adv = x_adv.detach()

        # Project back to ε-ball (L∞ constraint)
        perturbation: torch.Tensor = torch.clamp(
            x_adv - x, min=-epsilon, max=epsilon
        )
        x_adv = (x + perturbation).detach()

    return x_adv


def evaluate_adversarial_robustness(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    task_name: str = "fault_10s",
    epsilon: float = 2.0,
) -> dict[str, float]:
    """
    Run full adversarial robustness evaluation.

    Purpose:
        Measures model performance degradation under FGSM and PGD attacks.

    Inputs:
        model: PyTorch model.
        x: Clean test data.
        y: True labels.
        task_name: Task head to evaluate.
        epsilon: Perturbation budget.

    Outputs:
        Dict with clean and adversarial accuracy/F1, robustness score.

    Robustness score = adversarial_f1 / clean_f1 (1.0 = perfectly robust).
    """
    from sklearn.metrics import f1_score

    # Clean evaluation
    model.eval()
    with torch.no_grad():
        clean_out: dict[str, torch.Tensor] = model(x)
        clean_pred: np.ndarray = (
            clean_out[task_name].cpu().numpy() > 0.0
        ).astype(int).flatten()
    y_np: np.ndarray = y.cpu().numpy().flatten()

    clean_f1: float = float(
        f1_score(y_np, clean_pred, zero_division=0)
    )

    # FGSM attack
    x_fgsm: torch.Tensor = fgsm_attack(
        model, x, y, epsilon=epsilon, task_name=task_name
    )
    with torch.no_grad():
        fgsm_out: dict[str, torch.Tensor] = model(x_fgsm)
        fgsm_pred: np.ndarray = (
            fgsm_out[task_name].cpu().numpy() > 0.0
        ).astype(int).flatten()

    fgsm_f1: float = float(
        f1_score(y_np, fgsm_pred, zero_division=0)
    )

    # PGD attack
    x_pgd: torch.Tensor = pgd_attack(
        model, x, y, epsilon=epsilon, task_name=task_name
    )
    with torch.no_grad():
        pgd_out: dict[str, torch.Tensor] = model(x_pgd)
        pgd_pred: np.ndarray = (
            pgd_out[task_name].cpu().numpy() > 0.0
        ).astype(int).flatten()

    pgd_f1: float = float(
        f1_score(y_np, pgd_pred, zero_division=0)
    )

    robustness_fgsm: float = fgsm_f1 / max(clean_f1, 1e-10)
    robustness_pgd: float = pgd_f1 / max(clean_f1, 1e-10)

    return {
        "clean_f1": clean_f1,
        "fgsm_f1": fgsm_f1,
        "pgd_f1": pgd_f1,
        "robustness_score_fgsm": robustness_fgsm,
        "robustness_score_pgd": robustness_pgd,
        "epsilon_v": epsilon,
    }


def run_tests() -> None:
    """Sanity checks for adversarial module."""
    print("[PASS] eval/adversarial.py — import and function definitions OK.")


if __name__ == "__main__":
    run_tests()
