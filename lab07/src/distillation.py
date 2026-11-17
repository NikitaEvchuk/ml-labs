"""Knowledge distillation training routines for Lab 7."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from lab07.src.models import evaluate_model, set_seed, train_epoch_standard
except ImportError:
    from models import evaluate_model, set_seed, train_epoch_standard


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, float, float]:
    """Compute Hinton KD loss: alpha * CE + (1 - alpha) * T^2 * KL(p_teacher || p_student)."""
    ce = F.cross_entropy(student_logits, labels)
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    total = alpha * ce + (1.0 - alpha) * (temperature ** 2) * kl
    return total, float(ce.item()), float(kl.item())


def train_plain_student(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the compact student model from scratch without teacher supervision."""
    set_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(student.state_dict())

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = train_epoch_standard(student, train_loader, optimizer, criterion)
        val = evaluate_model(student, val_loader, criterion)
        scheduler.step()
        ep_time = time.perf_counter() - t0

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = copy.deepcopy(student.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_s": ep_time,
            }
        )
        print(
            f"[Plain Student] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.2f}% | "
            f"val_acc={val['acc']:.2f}% ({ep_time:.2f}s)"
        )

    total_time = time.perf_counter() - t_start
    student.load_state_dict(best_state)
    best_val = evaluate_model(student, val_loader, criterion)
    test_res = evaluate_model(student, test_loader, criterion)

    return {
        "history": history,
        "best_val_acc": best_val["acc"],
        "best_val_loss": best_val["loss"],
        "test_acc": test_res["acc"],
        "test_loss": test_res["loss"],
        "train_time_s": total_time,
    }


def train_distilled_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the compact student with knowledge distillation from the frozen teacher."""
    set_seed(seed)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ce_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(student.state_dict())

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss, ce_val, kl_val = distillation_loss(
                student_logits, teacher_logits, y, temperature=temperature, alpha=alpha
            )
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_ce += ce_val * bs
            total_kl += kl_val * bs
            correct += int((student_logits.argmax(dim=1) == y).sum().item())
            total += bs

        val = evaluate_model(student, val_loader, ce_eval)
        scheduler.step()
        ep_time = time.perf_counter() - t0

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = copy.deepcopy(student.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total,
                "train_ce": total_ce / total,
                "train_kl": total_kl / total,
                "train_acc": 100.0 * correct / total,
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_s": ep_time,
            }
        )
        print(
            f"[Distilled Student] Epoch {epoch:02d}/{epochs} | "
            f"loss={total_loss/total:.4f} (ce={total_ce/total:.4f}, kl={total_kl/total:.4f}) "
            f"train_acc={100.0*correct/total:.2f}% | val_acc={val['acc']:.2f}% ({ep_time:.2f}s)"
        )

    total_time = time.perf_counter() - t_start
    student.load_state_dict(best_state)
    best_val = evaluate_model(student, val_loader, ce_eval)
    test_res = evaluate_model(student, test_loader, ce_eval)

    return {
        "history": history,
        "best_val_acc": best_val["acc"],
        "best_val_loss": best_val["loss"],
        "test_acc": test_res["acc"],
        "test_loss": test_res["loss"],
        "train_time_s": total_time,
        "temperature": temperature,
        "alpha": alpha,
    }
