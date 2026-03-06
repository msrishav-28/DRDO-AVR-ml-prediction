"""
inspect_lr_schedule.py — Verify cosine LR schedule before training.

Run with:
    python inspect_lr_schedule.py

Expected output shows LR decreasing from 1e-3 to 1e-6 over 200 epochs.
Early stopping fires at ~epoch 80–120 on this dataset, so the effective
LR range used in practice is approximately 1e-3 → 3e-4.
"""

import torch
import torch.nn as nn

# Mirror the constants from run_publication.py
PINN_LR_INIT = 1e-3
PINN_LR_MIN = 1e-6
PINN_LR_TMAX = 200
MAX_EPOCHS = 200
EARLY_STOP_TYPICAL_EPOCH = 100  # approximate early stop epoch for display


def main():
    # Minimal model to attach scheduler to
    dummy_model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(
        dummy_model.parameters(), lr=PINN_LR_INIT, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PINN_LR_TMAX, eta_min=PINN_LR_MIN
    )

    print(f"{'Epoch':>6} | {'LR':>12} | {'Phase'}")
    print("-" * 50)

    for epoch in range(MAX_EPOCHS + 1):
        lr = scheduler.get_last_lr()[0]

        phase = ""
        if epoch == 0:
            phase = "← peak LR (fast learning)"
        elif epoch == 50:
            phase = "← still converging"
        elif epoch == EARLY_STOP_TYPICAL_EPOCH:
            phase = "← typical early stop point"
        elif epoch == 150:
            phase = "← fine-tuning zone"
        elif epoch == MAX_EPOCHS:
            phase = "← floor LR (if reached)"

        if epoch % 10 == 0 or phase:
            print(f"{epoch:>6} | {lr:>12.2e} | {phase}")

        if epoch < MAX_EPOCHS:
            scheduler.step()

    final_lr = scheduler.get_last_lr()[0]
    print("\nVerification:")
    print(f"  Start LR:   {PINN_LR_INIT:.2e}  (should be 1.00e-03)")
    print(f"  End LR:     {final_lr:.2e}  (should be 1.00e-06)")
    print(f"  LR at typical early stop (epoch {EARLY_STOP_TYPICAL_EPOCH}):")

    # Recompute at epoch 100
    opt2 = torch.optim.AdamW(dummy_model.parameters(), lr=PINN_LR_INIT)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=PINN_LR_TMAX, eta_min=PINN_LR_MIN
    )
    for _ in range(EARLY_STOP_TYPICAL_EPOCH):
        sched2.step()
    lr_at_stop = sched2.get_last_lr()[0]
    print(f"    {lr_at_stop:.2e}  (should be between 5e-4 and 1e-3)")

    assert abs(final_lr - PINN_LR_MIN) < 1e-10, \
        "ERROR: Final LR does not match PINN_LR_MIN. Check T_max."
    assert lr_at_stop > PINN_LR_MIN * 10, \
        "ERROR: LR at early stop epoch is too small. T_max may be too short."
    print("\n[OK] LR schedule is correctly configured.")


if __name__ == "__main__":
    main()
