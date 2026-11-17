"""End-to-end experiment suite and plotting for Lab 7 (Compression: Pruning & Distillation)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from lab07.src.benchmark import (
        benchmark_dense_vs_csr_linear,
        measure_inference_energy,
        measure_inference_latency,
    )
    from lab07.src.distillation import train_distilled_student, train_plain_student
    from lab07.src.models import (
        OversizedCNN,
        StudentCNN,
        compute_macs_and_params,
        count_parameters,
        evaluate_model,
        load_fashion_mnist,
        set_seed,
        train_epoch_standard,
    )
    from lab07.src.pruning import (
        build_magnitude_masks,
        finetune_with_fixed_mask,
        measure_storage_formats,
        structured_prune_filters,
        summarize_model_metrics,
    )
except ImportError:
    from benchmark import (
        benchmark_dense_vs_csr_linear,
        measure_inference_energy,
        measure_inference_latency,
    )
    from distillation import train_distilled_student, train_plain_student
    from models import (
        OversizedCNN,
        StudentCNN,
        compute_macs_and_params,
        count_parameters,
        evaluate_model,
        load_fashion_mnist,
        set_seed,
        train_epoch_standard,
    )
    from pruning import (
        build_magnitude_masks,
        finetune_with_fixed_mask,
        measure_storage_formats,
        structured_prune_filters,
        summarize_model_metrics,
    )


DEFAULT_CONFIG_PATH = Path("lab07/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_loaders(cfg: dict[str, Any]):
    d = cfg["data"]
    return load_fashion_mnist(
        root=d["root"],
        train_size=d["train_size"],
        val_size=d["val_size"],
        batch_size=d["batch_size"],
        eval_batch_size=d["eval_batch_size"],
        seed=cfg["seed"],
        download=d["download"],
    )


def build_baseline_model(cfg: dict[str, Any]) -> OversizedCNN:
    b = cfg["baseline"]
    return OversizedCNN(
        channels=tuple(b["channels"]),
        fc_hidden=b["fc_hidden"],
        num_classes=b["num_classes"],
    )


def build_student_model(cfg: dict[str, Any]) -> StudentCNN:
    d = cfg["distillation"]
    return StudentCNN(
        conv_channels=tuple(d["conv_channels"]),
        fc_hidden=d["fc_hidden"],
        num_classes=d["num_classes"],
    )


def run_task1_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 1: Train oversized ~1M-parameter CNN baseline on Fashion-MNIST."""
    print("\n=== TASK 1: Training Oversized Baseline CNN (~1M Params) ===")
    set_seed(cfg["seed"])
    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = build_baseline_model(cfg)
    total_params = count_parameters(model)
    print(f"OversizedCNN total trainable parameters: {total_params:,}")

    b = cfg["baseline"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=b["lr"], weight_decay=b["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=b["epochs"], eta_min=1e-5
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    t_start = time.perf_counter()
    for epoch in range(1, b["epochs"] + 1):
        t0 = time.perf_counter()
        tr = train_epoch_standard(model, train_loader, optimizer, criterion)
        val = evaluate_model(model, val_loader, criterion)
        scheduler.step()
        ep_time = time.perf_counter() - t0

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = copy.deepcopy(model.state_dict())

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
            f"[Baseline] Epoch {epoch:02d}/{b['epochs']} | "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.2f}% | "
            f"val_acc={val['acc']:.2f}% ({ep_time:.2f}s)"
        )

    total_train_time_s = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "baseline_oversized_best.pt"
    torch.save(model.state_dict(), best_ckpt_path)

    results_dir = Path(cfg["paths"]["results_dir"])
    storage = measure_storage_formats(model, ckpt_dir / "storage", "baseline_0")
    bm = cfg["benchmark"]
    latency = measure_inference_latency(
        model,
        warmup_single=bm["warmup_single"],
        iters_single=bm["iters_single"],
        warmup_batch=bm["warmup_batch"],
        iters_batch=bm["iters_batch"],
        batch_size=bm["latency_batch_size"],
    )

    val_res = evaluate_model(model, val_loader, criterion)
    test_res = evaluate_model(model, test_loader, criterion)
    macs, thop_params = compute_macs_and_params(model)

    baseline_data = {
        "architecture": {
            "name": "OversizedCNN",
            "channels": b["channels"],
            "fc_hidden": b["fc_hidden"],
            "total_params": total_params,
            "thop_params": thop_params,
            "dense_macs": macs,
            "dense_flops": 2.0 * macs,
        },
        "training": {
            "epochs": b["epochs"],
            "optimizer": "Adam",
            "lr": b["lr"],
            "weight_decay": b["weight_decay"],
            "scheduler": "CosineAnnealingLR",
            "total_train_time_s": total_train_time_s,
            "history": history,
        },
        "metrics": {
            "best_val_acc": val_res["acc"],
            "best_val_loss": val_res["loss"],
            "test_acc": test_res["acc"],
            "test_loss": test_res["loss"],
            "storage": storage,
            "latency": latency,
        },
        "checkpoint_path": str(best_ckpt_path),
    }
    save_json(baseline_data, results_dir / "baseline_metrics.json")
    print(
        f"Baseline done: params={total_params:,}, test_acc={test_res['acc']:.2f}%, "
        f"size={storage['dense_pt_mb']:.2f} MB, single_lat={latency['single_mean_ms']:.3f} ms"
    )
    return baseline_data


def run_task2_pruning(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 2: Unstructured global/layer-wise magnitude pruning + structured L2 filter pruning."""
    print("\n=== TASK 2: Magnitude Pruning & Structured Filter Pruning ===")
    train_loader, val_loader, test_loader = build_loaders(cfg)
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    best_ckpt_path = ckpt_dir / "baseline_oversized_best.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {best_ckpt_path}. Run --step baseline first."
        )

    baseline_state = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
    base_model = build_baseline_model(cfg)
    base_model.load_state_dict(baseline_state)
    baseline_params = count_parameters(base_model)
    base_summary = summarize_model_metrics(
        base_model, val_loader, test_loader, baseline_params=baseline_params
    )

    p_cfg = cfg["pruning"]
    unstructured_results: dict[str, list[dict[str, Any]]] = {"global": [], "layerwise": []}

    for mode in ("global", "layerwise"):
        print(f"\n--- Unstructured Magnitude Pruning ({mode}) ---")
        for sp_target in p_cfg["unstructured_sparsities"]:
            if sp_target == 0.0:
                entry = {
                    "mode": mode,
                    "target_sparsity": 0.0,
                    "target_sparsity_pct": 0.0,
                    "zeroshot": copy.deepcopy(base_summary),
                    "finetuned": copy.deepcopy(base_summary),
                    "finetune_history": [],
                    "checkpoint_path": str(best_ckpt_path),
                }
                unstructured_results[mode].append(entry)
                print(
                    f"[{mode} sp=0%] test_acc={base_summary['test_acc']:.2f}% (baseline)"
                )
                continue

            set_seed(cfg["seed"])
            model = build_baseline_model(cfg)
            model.load_state_dict(copy.deepcopy(baseline_state))
            masks = build_magnitude_masks(model, sparsity=sp_target, mode=mode)

            # Evaluate immediately after masking (before fine-tuning)
            from pruning import apply_masks_  # safe local import
            apply_masks_(model, masks)
            zs_summary = summarize_model_metrics(
                model, val_loader, test_loader, baseline_params=baseline_params
            )

            # Fine-tune 3 epochs with mask fixed
            ft_hist = finetune_with_fixed_mask(
                model,
                masks,
                train_loader,
                val_loader,
                epochs=p_cfg["finetune_epochs"],
                lr=p_cfg["finetune_lr"],
                weight_decay=p_cfg["weight_decay"],
            )
            ft_summary = summarize_model_metrics(
                model, val_loader, test_loader, baseline_params=baseline_params
            )

            sp_tag = int(round(sp_target * 100))
            ckpt_path = ckpt_dir / f"pruned_{mode}_{sp_tag}.pt"
            torch.save(model.state_dict(), ckpt_path)

            unstructured_results[mode].append(
                {
                    "mode": mode,
                    "target_sparsity": sp_target,
                    "target_sparsity_pct": 100.0 * sp_target,
                    "zeroshot": zs_summary,
                    "finetuned": ft_summary,
                    "finetune_history": ft_hist,
                    "checkpoint_path": str(ckpt_path),
                }
            )
            print(
                f"[{mode} sp={sp_tag:2d}%] actual_sp={ft_summary['weight_sparsity_pct']:.2f}% | "
                f"zeroshot_test={zs_summary['test_acc']:.2f}% -> "
                f"finetuned_test={ft_summary['test_acc']:.2f}%"
            )

    print("\n--- Structured L2 Filter Pruning ---")
    structured_results: list[dict[str, Any]] = []
    # Include 0% reference point for structured curve clarity
    structured_results.append(
        {
            "filter_prune_ratio": 0.0,
            "filter_prune_pct": 0.0,
            "channels": list(cfg["baseline"]["channels"]),
            "zeroshot": copy.deepcopy(base_summary),
            "finetuned": copy.deepcopy(base_summary),
            "finetune_history": [],
            "checkpoint_path": str(best_ckpt_path),
        }
    )

    for ratio in p_cfg["structured_filter_ratios"]:
        set_seed(cfg["seed"])
        teacher_copy = build_baseline_model(cfg)
        teacher_copy.load_state_dict(copy.deepcopy(baseline_state))
        pruned_model = structured_prune_filters(teacher_copy, filter_prune_ratio=ratio)

        zs_summary = summarize_model_metrics(
            pruned_model, val_loader, test_loader, baseline_params=baseline_params
        )
        ft_hist = finetune_with_fixed_mask(
            pruned_model,
            masks=None,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=p_cfg["finetune_epochs"],
            lr=p_cfg["finetune_lr"],
            weight_decay=p_cfg["weight_decay"],
        )
        ft_summary = summarize_model_metrics(
            pruned_model, val_loader, test_loader, baseline_params=baseline_params
        )

        ratio_tag = int(round(ratio * 100))
        ckpt_path = ckpt_dir / f"pruned_structured_{ratio_tag}.pt"
        torch.save(
            {
                "channels": list(pruned_model.channels),
                "fc_hidden": pruned_model.fc_hidden,
                "num_classes": pruned_model.num_classes,
                "state_dict": pruned_model.state_dict(),
            },
            ckpt_path,
        )

        structured_results.append(
            {
                "filter_prune_ratio": ratio,
                "filter_prune_pct": 100.0 * ratio,
                "channels": list(pruned_model.channels),
                "zeroshot": zs_summary,
                "finetuned": ft_summary,
                "finetune_history": ft_hist,
                "checkpoint_path": str(ckpt_path),
            }
        )
        print(
            f"[Structured filters -{ratio_tag}%] channels={pruned_model.channels} | "
            f"param_sparsity={ft_summary['structural_param_sparsity_pct']:.2f}% | "
            f"zeroshot_test={zs_summary['test_acc']:.2f}% -> "
            f"finetuned_test={ft_summary['test_acc']:.2f}%"
        )

    pruning_data = {
        "unstructured": unstructured_results,
        "structured": structured_results,
    }
    save_json(pruning_data, results_dir / "pruning_results.json")
    plot_task2_pruning(pruning_data, results_dir / "pruning_sparsity_curves.png")
    return pruning_data


def plot_task2_pruning(pruning_data: dict[str, Any], out_path: Path) -> None:
    """Plot accuracy vs sparsity for unstructured (global & layer-wise) and structured pruning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    g_rows = pruning_data["unstructured"]["global"]
    l_rows = pruning_data["unstructured"]["layerwise"]
    s_rows = pruning_data["structured"]

    g_sp = [r["finetuned"]["weight_sparsity_pct"] for r in g_rows]
    g_ft = [r["finetuned"]["test_acc"] for r in g_rows]
    g_zs = [r["zeroshot"]["test_acc"] for r in g_rows]

    l_sp = [r["finetuned"]["weight_sparsity_pct"] for r in l_rows]
    l_ft = [r["finetuned"]["test_acc"] for r in l_rows]
    l_zs = [r["zeroshot"]["test_acc"] for r in l_rows]

    s_sp = [r["finetuned"]["structural_param_sparsity_pct"] for r in s_rows]
    s_ft = [r["finetuned"]["test_acc"] for r in s_rows]
    s_labels = [f"{int(r['filter_prune_pct'])}% filt." for r in s_rows]

    ax1.plot(g_sp, g_ft, "o-", linewidth=2.2, color="#1f77b4", label="Global Magnitude (Fine-tuned 3 ep)")
    ax1.plot(g_sp, g_zs, "o--", linewidth=1.3, alpha=0.65, color="#1f77b4", label="Global Magnitude (Zero-shot)")
    ax1.plot(l_sp, l_ft, "s-", linewidth=2.2, color="#ff7f0e", label="Layer-wise Magnitude (Fine-tuned 3 ep)")
    ax1.plot(l_sp, l_zs, "s--", linewidth=1.3, alpha=0.65, color="#ff7f0e", label="Layer-wise Magnitude (Zero-shot)")
    ax1.plot(s_sp, s_ft, "^-", linewidth=2.4, markersize=8, color="#2ca02c", label="Structured L2 Filters (Fine-tuned 3 ep)")

    for x, y, lbl in zip(s_sp[1:], s_ft[1:], s_labels[1:]):
        ax1.annotate(
            lbl,
            xy=(x, y),
            xytext=(4, -13),
            textcoords="offset points",
            fontsize=8.5,
            color="#1b6b1b",
            fontweight="bold",
        )

    ax1.set_xlabel("Parameter / Weight Sparsity (%)", fontsize=10.5)
    ax1.set_ylabel("Fashion-MNIST Test Accuracy (%)", fontsize=10.5)
    ax1.set_title("Accuracy vs. Sparsity: Global, Layer-wise & Structured Pruning", fontsize=11.5)
    ax1.set_xlim(-3, 99)
    y_min = max(0.0, min(min(g_ft), min(l_ft), min(s_ft), min(g_zs), min(l_zs)) - 4)
    ax1.set_ylim(min(60.0, y_min), 95.0)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower left", fontsize=8.5)

    # Panel 2: Layer-by-layer sparsity distribution for Global 90% vs Layerwise 90%
    g_90 = next(r for r in g_rows if int(round(r["target_sparsity_pct"])) == 90)
    layer_names = list(g_90["finetuned"]["layer_sparsity_pct"].keys())
    short_names = [
        n.replace("features.", "conv.").replace("classifier.", "fc.") for n in layer_names
    ]
    g90_vals = [g_90["finetuned"]["layer_sparsity_pct"][n] for n in layer_names]
    l90_vals = [90.0 for _ in layer_names]

    x_idx = np.arange(len(short_names))
    w = 0.36
    bars1 = ax2.bar(x_idx - w / 2, g90_vals, width=w, color="#1f77b4", label="Global (90% overall)")
    ax2.bar(x_idx + w / 2, l90_vals, width=w, color="#ff7f0e", alpha=0.85, label="Layer-wise (90% each)")
    for b, val in zip(bars1, g90_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(short_names, rotation=20, ha="right", fontsize=9.5)
    ax2.set_ylabel("Layer Weight Sparsity (%)", fontsize=10.5)
    ax2.set_ylim(0, 108)
    ax2.set_title("Per-Layer Sparsity Allocation at 90% Target", fontsize=11.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_task4_distillation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 4: Train ~100k-param StudentCNN with and without Teacher distillation."""
    print("\n=== TASK 4: Knowledge Distillation (Teacher ~1M -> Student ~100k) ===")
    train_loader, val_loader, test_loader = build_loaders(cfg)
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    best_ckpt_path = ckpt_dir / "baseline_oversized_best.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {best_ckpt_path}. Run --step baseline first."
        )

    teacher = build_baseline_model(cfg)
    teacher.load_state_dict(torch.load(best_ckpt_path, map_location="cpu", weights_only=True))
    teacher_params = count_parameters(teacher)
    teacher_macs, _ = compute_macs_and_params(teacher)
    teacher_test = evaluate_model(teacher, test_loader)

    d_cfg = cfg["distillation"]
    plain_student = build_student_model(cfg)
    student_params = count_parameters(plain_student)
    student_macs, _ = compute_macs_and_params(plain_student)
    print(
        f"StudentCNN parameters: {student_params:,} "
        f"({teacher_params / student_params:.2f}x smaller than Teacher)"
    )

    plain_res = train_plain_student(
        plain_student,
        train_loader,
        val_loader,
        test_loader,
        epochs=d_cfg["epochs"],
        lr=d_cfg["lr"],
        weight_decay=d_cfg["weight_decay"],
        seed=cfg["seed"],
    )
    plain_ckpt = ckpt_dir / "student_plain_best.pt"
    torch.save(plain_student.state_dict(), plain_ckpt)

    distilled_student = build_student_model(cfg)
    kd_res = train_distilled_student(
        distilled_student,
        teacher,
        train_loader,
        val_loader,
        test_loader,
        epochs=d_cfg["epochs"],
        lr=d_cfg["lr"],
        weight_decay=d_cfg["weight_decay"],
        temperature=d_cfg["temperature"],
        alpha=d_cfg["alpha"],
        seed=cfg["seed"],
    )
    kd_ckpt = ckpt_dir / "student_distilled_best.pt"
    torch.save(distilled_student.state_dict(), kd_ckpt)

    bm = cfg["benchmark"]
    teacher_lat = measure_inference_latency(
        teacher,
        warmup_single=bm["warmup_single"],
        iters_single=bm["iters_single"],
        warmup_batch=bm["warmup_batch"],
        iters_batch=bm["iters_batch"],
        batch_size=bm["latency_batch_size"],
    )
    plain_lat = measure_inference_latency(
        plain_student,
        warmup_single=bm["warmup_single"],
        iters_single=bm["iters_single"],
        warmup_batch=bm["warmup_batch"],
        iters_batch=bm["iters_batch"],
        batch_size=bm["latency_batch_size"],
    )
    kd_lat = measure_inference_latency(
        distilled_student,
        warmup_single=bm["warmup_single"],
        iters_single=bm["iters_single"],
        warmup_batch=bm["warmup_batch"],
        iters_batch=bm["iters_batch"],
        batch_size=bm["latency_batch_size"],
    )

    storage_dir = ckpt_dir / "storage"
    teacher_storage = measure_storage_formats(teacher, storage_dir, "kd_teacher")
    plain_storage = measure_storage_formats(plain_student, storage_dir, "student_plain")
    kd_storage = measure_storage_formats(distilled_student, storage_dir, "student_distilled")

    distill_data = {
        "teacher": {
            "params": teacher_params,
            "macs": teacher_macs,
            "test_acc": teacher_test["acc"],
            "test_loss": teacher_test["loss"],
            "storage": teacher_storage,
            "latency": teacher_lat,
        },
        "plain_student": {
            "params": student_params,
            "macs": student_macs,
            "checkpoint_path": str(plain_ckpt),
            "storage": plain_storage,
            "latency": plain_lat,
            **plain_res,
        },
        "distilled_student": {
            "params": student_params,
            "macs": student_macs,
            "checkpoint_path": str(kd_ckpt),
            "storage": kd_storage,
            "latency": kd_lat,
            **kd_res,
        },
        "kd_gain_test_acc_pct": kd_res["test_acc"] - plain_res["test_acc"],
    }
    save_json(distill_data, results_dir / "distillation_results.json")
    plot_task4_distillation(distill_data, results_dir / "distillation_comparison.png")
    print(
        f"Distillation done: Teacher={teacher_test['acc']:.2f}%, "
        f"Plain Student={plain_res['test_acc']:.2f}%, "
        f"Distilled Student={kd_res['test_acc']:.2f}% "
        f"(gain={distill_data['kd_gain_test_acc_pct']:+.2f}%)"
    )
    return distill_data


def plot_task4_distillation(distill_data: dict[str, Any], out_path: Path) -> None:
    """Plot Plain vs Distilled Student training curves and Teacher/Student comparison bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    p_hist = distill_data["plain_student"]["history"]
    k_hist = distill_data["distilled_student"]["history"]
    epochs = [h["epoch"] for h in p_hist]

    ax1.plot(
        epochs,
        [h["val_acc"] for h in p_hist],
        "o-",
        linewidth=2,
        color="#ff7f0e",
        label="Plain Student (Val Acc)",
    )
    ax1.plot(
        epochs,
        [h["val_acc"] for h in k_hist],
        "s-",
        linewidth=2,
        color="#2ca02c",
        label="Distilled Student T=4, a=0.5 (Val Acc)",
    )
    ax1.axhline(
        distill_data["teacher"]["test_acc"],
        linestyle="--",
        color="#1f77b4",
        linewidth=1.8,
        label=f"Teacher Baseline ({distill_data['teacher']['test_acc']:.2f}% test)",
    )
    ax1.set_xlabel("Epoch", fontsize=10.5)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10.5)
    ax1.set_title("Student Validation Accuracy: Plain vs. Distilled", fontsize=11.5)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel 2: Bar comparison across Test Acc, Size (MB), Single Latency (ms)
    names = ["Teacher\n(~1.01M)", "Plain Student\n(~100k)", "Distilled Student\n(~100k)"]
    accs = [
        distill_data["teacher"]["test_acc"],
        distill_data["plain_student"]["test_acc"],
        distill_data["distilled_student"]["test_acc"],
    ]
    sizes_kb = [
        distill_data["teacher"]["storage"]["dense_pt_bytes"] / 1024.0,
        distill_data["plain_student"]["storage"]["dense_pt_bytes"] / 1024.0,
        distill_data["distilled_student"]["storage"]["dense_pt_bytes"] / 1024.0,
    ]
    lats = [
        distill_data["teacher"]["latency"]["single_mean_ms"],
        distill_data["plain_student"]["latency"]["single_mean_ms"],
        distill_data["distilled_student"]["latency"]["single_mean_ms"],
    ]

    x = np.arange(len(names))
    ax2_twin = ax2.twinx()

    b1 = ax2.bar(x - 0.18, accs, width=0.34, color="#4C78A8", label="Test Accuracy (%)")
    b2 = ax2_twin.bar(x + 0.18, lats, width=0.34, color="#E45756", alpha=0.88, label="Single-Sample Latency (ms)")

    for bar, acc, sz in zip(b1, accs, sizes_kb):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{acc:.2f}%\n({sz:.0f} KB)",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#1e3c5c",
            fontweight="bold",
        )
    for bar, lat in zip(b2, lats):
        ax2_twin.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{lat:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#9e2a29",
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=10.5, color="#1e3c5c")
    ax2.set_ylim(80, 96)
    ax2_twin.set_ylabel("Single-Sample CPU Latency (ms)", fontsize=10.5, color="#9e2a29")
    ax2_twin.set_ylim(0, max(lats) * 1.35)
    ax2.set_title("Teacher vs. Plain vs. Distilled Student", fontsize=11.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_task3_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 3: Measure latency, dense vs sparse disk sizes, CSR vs dense microbenchmark, & energy."""
    print("\n=== TASK 3: Latency, Dense vs. Sparse Storage, and Energy Analysis ===")
    _, _, test_loader = build_loaders(cfg)
    test_x = test_loader.dataset.tensors[0]  # type: ignore[attr-defined]
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    storage_dir = ckpt_dir / "storage"
    bm = cfg["benchmark"]

    # 1. Storage & Latency across Global Unstructured Sparsities
    unstructured_bench: list[dict[str, Any]] = []
    for sp_target in cfg["pruning"]["unstructured_sparsities"]:
        sp_tag = int(round(sp_target * 100))
        model = build_baseline_model(cfg)
        ckpt_path = (
            ckpt_dir / "baseline_oversized_best.pt"
            if sp_tag == 0
            else ckpt_dir / f"pruned_global_{sp_tag}.pt"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

        storage = measure_storage_formats(model, storage_dir, f"global_{sp_tag}")
        latency = measure_inference_latency(
            model,
            warmup_single=bm["warmup_single"],
            iters_single=bm["iters_single"],
            warmup_batch=bm["warmup_batch"],
            iters_batch=bm["iters_batch"],
            batch_size=bm["latency_batch_size"],
        )
        unstructured_bench.append(
            {
                "label": f"Unstructured Global {sp_tag}%",
                "target_sparsity_pct": float(sp_tag),
                "storage": storage,
                "latency": latency,
            }
        )
        print(
            f"[Bench Unstruct {sp_tag:2d}%] single={latency['single_mean_ms']:.3f}ms, "
            f"batch={latency['batch_mean_ms']:.2f}ms | "
            f"dense.pt={storage['dense_pt_mb']:.2f}MB, "
            f"csr.npz={storage['sparse_csr_npz_mb']:.2f}MB, "
            f"csr_comp.npz={storage['sparse_csr_npz_compressed_mb']:.2f}MB"
        )

    # 2. Storage & Latency across Structured Filter Pruned models
    structured_bench: list[dict[str, Any]] = []
    for ratio in cfg["pruning"]["structured_filter_ratios"]:
        ratio_tag = int(round(ratio * 100))
        ckpt_path = ckpt_dir / f"pruned_structured_{ratio_tag}.pt"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = OversizedCNN(
            channels=tuple(payload["channels"]),
            fc_hidden=payload["fc_hidden"],
            num_classes=payload["num_classes"],
        )
        model.load_state_dict(payload["state_dict"])

        storage = measure_storage_formats(model, storage_dir, f"structured_{ratio_tag}")
        latency = measure_inference_latency(
            model,
            warmup_single=bm["warmup_single"],
            iters_single=bm["iters_single"],
            warmup_batch=bm["warmup_batch"],
            iters_batch=bm["iters_batch"],
            batch_size=bm["latency_batch_size"],
        )
        structured_bench.append(
            {
                "label": f"Structured -{ratio_tag}% Filters",
                "filter_prune_pct": float(ratio_tag),
                "channels": payload["channels"],
                "params": count_parameters(model),
                "storage": storage,
                "latency": latency,
            }
        )
        print(
            f"[Bench Struct -{ratio_tag:2d}%] single={latency['single_mean_ms']:.3f}ms, "
            f"batch={latency['batch_mean_ms']:.2f}ms | "
            f"dense.pt={storage['dense_pt_mb']:.2f}MB"
        )

    # 3. Sparse CSR vs Dense BLAS Linear Microbenchmark
    print("\n--- Dense BLAS vs. PyTorch Sparse CSR Linear Microbenchmark (1440 -> 640) ---")
    csr_micro = benchmark_dense_vs_csr_linear(
        in_features=cfg["baseline"]["channels"][-1] * 9,
        out_features=cfg["baseline"]["fc_hidden"],
        sparsities=(0.5, 0.75, 0.9, 0.95),
        batch_size=bm["latency_batch_size"],
    )
    for row in csr_micro:
        print(
            f"  {row['kernel']:16s} sp={row['sparsity_pct']:4.0f}% | "
            f"single={row['single_ms']:.3f}ms ({row['speedup_vs_dense_single']:.2f}x) | "
            f"batch128={row['batch_ms']:.3f}ms ({row['speedup_vs_dense_batch']:.2f}x)"
        )

    # 4. Energy measurement on key representatives via CodeCarbon
    print("\n--- CodeCarbon Inference Energy Measurement ---")
    energy_models: list[tuple[str, nn.Module]] = []

    m_base = build_baseline_model(cfg)
    m_base.load_state_dict(
        torch.load(ckpt_dir / "baseline_oversized_best.pt", map_location="cpu", weights_only=True)
    )
    energy_models.append(("baseline", m_base))

    m_g90 = build_baseline_model(cfg)
    m_g90.load_state_dict(
        torch.load(ckpt_dir / "pruned_global_90.pt", map_location="cpu", weights_only=True)
    )
    energy_models.append(("unstructured_global_90", m_g90))

    p50 = torch.load(ckpt_dir / "pruned_structured_50.pt", map_location="cpu", weights_only=True)
    m_s50 = OversizedCNN(
        channels=tuple(p50["channels"]),
        fc_hidden=p50["fc_hidden"],
        num_classes=p50["num_classes"],
    )
    m_s50.load_state_dict(p50["state_dict"])
    energy_models.append(("structured_50", m_s50))

    m_plain = build_student_model(cfg)
    m_plain.load_state_dict(
        torch.load(ckpt_dir / "student_plain_best.pt", map_location="cpu", weights_only=True)
    )
    energy_models.append(("student_plain", m_plain))

    m_kd = build_student_model(cfg)
    m_kd.load_state_dict(
        torch.load(ckpt_dir / "student_distilled_best.pt", map_location="cpu", weights_only=True)
    )
    energy_models.append(("student_distilled", m_kd))

    energy_results: dict[str, dict[str, float]] = {}
    for key, mod in energy_models:
        e_res = measure_inference_energy(
            mod,
            test_x,
            repeats=bm["energy_repeats"],
            batch_size=bm["latency_batch_size"],
            country_iso_code=bm["country_iso_code"],
        )
        energy_results[key] = e_res
        print(
            f"[Energy {key:22s}] {e_res['throughput_samples_s']:.0f} img/s | "
            f"{e_res['energy_joules_per_10k_inferences']:.2f} J / 10k imgs | "
            f"P_avg={e_res['avg_power_watts']:.2f} W"
        )

    bench_data = {
        "unstructured_bench": unstructured_bench,
        "structured_bench": structured_bench,
        "csr_linear_microbenchmark": csr_micro,
        "energy": energy_results,
    }
    save_json(bench_data, results_dir / "latency_storage_results.json")
    plot_task3_analysis(bench_data, results_dir / "latency_and_size_analysis.png")
    return bench_data


def plot_task3_analysis(bench_data: dict[str, Any], out_path: Path) -> None:
    """Plot on-disk storage formats, full-model CPU latency, and sparse CSR microbenchmark."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel 1: On-disk size Dense vs Sparse CSR vs Compressed CSR
    u_rows = bench_data["unstructured_bench"]
    sps = [r["target_sparsity_pct"] for r in u_rows]
    dense_mb = [r["storage"]["dense_pt_mb"] for r in u_rows]
    csr_pt_mb = [r["storage"]["sparse_csr_pt_mb"] for r in u_rows]
    csr_npz_mb = [r["storage"]["sparse_csr_npz_mb"] for r in u_rows]
    csr_comp_mb = [r["storage"]["sparse_csr_npz_compressed_mb"] for r in u_rows]

    axes[0].plot(sps, dense_mb, "o-", linewidth=2, color="#4C78A8", label="Dense FP32 (.pt)")
    axes[0].plot(sps, csr_pt_mb, "s-", linewidth=2, color="#E45756", label="PyTorch CSR (.pt)")
    axes[0].plot(sps, csr_npz_mb, "^--", linewidth=1.8, color="#F58518", label="SciPy CSR (.npz)")
    axes[0].plot(sps, csr_comp_mb, "D-", linewidth=2, color="#54A24B", label="Compressed CSR (.npz)")
    axes[0].set_xlabel("Unstructured Weight Sparsity (%)", fontsize=10)
    axes[0].set_ylabel("On-Disk Size (MB)", fontsize=10)
    axes[0].set_title("On-Disk Size: Dense vs. Sparse Formats", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8)

    # Panel 2: Full-model CPU single-sample latency (Baseline vs Unstructured vs Structured)
    s_rows = bench_data["structured_bench"]
    labels = [
        "Baseline\n(0%)",
        "Unstruct.\n(50%)",
        "Unstruct.\n(90%)",
        "Unstruct.\n(95%)",
        "Struct.\n(-10% filt)",
        "Struct.\n(-25% filt)",
        "Struct.\n(-50% filt)",
    ]
    u_map = {int(r["target_sparsity_pct"]): r["latency"]["single_mean_ms"] for r in u_rows}
    s_map = {int(r["filter_prune_pct"]): r["latency"]["single_mean_ms"] for r in s_rows}
    vals = [u_map[0], u_map[50], u_map[90], u_map[95], s_map[10], s_map[25], s_map[50]]
    colors = ["#4C78A8", "#F58518", "#F58518", "#F58518", "#54A24B", "#54A24B", "#54A24B"]

    bars = axes[1].bar(np.arange(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Single-Sample Mean Latency (ms)", fontsize=10)
    axes[1].set_title("Inference Latency: Unstructured vs. Structured", fontsize=11)
    axes[1].set_ylim(0, max(vals) * 1.2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)

    # Panel 3: Dense BLAS vs PyTorch Sparse CSR Linear(1440 -> 640) at batch=128
    csr_rows = bench_data["csr_linear_microbenchmark"]
    c_labels = [
        "Dense\nBLAS" if r["kernel"] == "dense_blas" else f"CSR\n{int(r['sparsity_pct'])}%"
        for r in csr_rows
    ]
    c_vals = [r["batch_ms"] for r in csr_rows]
    c_colors = ["#4C78A8"] + ["#E45756"] * (len(csr_rows) - 1)
    bars3 = axes[2].bar(np.arange(len(c_labels)), c_vals, color=c_colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars3, c_vals):
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.03,
            f"{v:.2f}ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[2].set_xticks(np.arange(len(c_labels)))
    axes[2].set_xticklabels(c_labels, fontsize=8.5)
    axes[2].set_ylabel("Batched FC1 Latency (ms, bs=128)", fontsize=10)
    axes[2].set_title("Why CSR Fails on CPU: Dense vs. Sparse GEMM", fontsize=11)
    axes[2].set_ylim(0, max(c_vals) * 1.2)
    axes[2].grid(axis="y", linestyle="--", alpha=0.45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_task5_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 5: Consolidate comparison table organized by Ch. 9 pillars (Data, Model, Compute, Energy)."""
    print("\n=== TASK 5: Generating Ch. 9 Efficiency Pillars Summary ===")
    results_dir = Path(cfg["paths"]["results_dir"])
    baseline = load_json(results_dir / "baseline_metrics.json")
    pruning = load_json(results_dir / "pruning_results.json")
    bench = load_json(results_dir / "latency_storage_results.json")
    distill = load_json(results_dir / "distillation_results.json")

    g90 = next(
        r for r in pruning["unstructured"]["global"] if int(round(r["target_sparsity_pct"])) == 90
    )
    g90_bench = next(
        r for r in bench["unstructured_bench"] if int(round(r["target_sparsity_pct"])) == 90
    )
    s50 = next(
        r for r in pruning["structured"] if int(round(r["filter_prune_pct"])) == 50
    )
    s50_bench = next(
        r for r in bench["structured_bench"] if int(round(r["filter_prune_pct"])) == 50
    )

    rows = [
        {
            "variant": "Baseline Oversized CNN",
            "data_pillar": "55k train / 5k val / 10k test (1x28x28 FP32)",
            "model_params": baseline["architecture"]["total_params"],
            "model_nonzero_params": baseline["architecture"]["total_params"],
            "model_dense_mb": baseline["metrics"]["storage"]["dense_pt_mb"],
            "model_compressed_mb": baseline["metrics"]["storage"]["sparse_csr_npz_compressed_mb"],
            "test_acc_pct": baseline["metrics"]["test_acc"],
            "compute_theoretical_macs": baseline["architecture"]["dense_macs"],
            "compute_executed_macs": baseline["architecture"]["dense_macs"],
            "single_latency_ms": baseline["metrics"]["latency"]["single_mean_ms"],
            "batch_throughput_samples_s": baseline["metrics"]["latency"]["batch_throughput_samples_s"],
            "energy_j_per_10k": bench["energy"]["baseline"]["energy_joules_per_10k_inferences"],
        },
        {
            "variant": "Unstructured Global Pruned (90%)",
            "data_pillar": "Same dataset + 3 ep mask-fixed fine-tune",
            "model_params": g90["finetuned"]["total_params"],
            "model_nonzero_params": g90["finetuned"]["nonzero_weights"],
            "model_dense_mb": g90_bench["storage"]["dense_pt_mb"],
            "model_compressed_mb": g90_bench["storage"]["sparse_csr_npz_compressed_mb"],
            "test_acc_pct": g90["finetuned"]["test_acc"],
            "compute_theoretical_macs": g90["finetuned"]["theoretical_nonzero_macs"],
            "compute_executed_macs": g90["finetuned"]["dense_macs"],
            "single_latency_ms": g90_bench["latency"]["single_mean_ms"],
            "batch_throughput_samples_s": g90_bench["latency"]["batch_throughput_samples_s"],
            "energy_j_per_10k": bench["energy"]["unstructured_global_90"]["energy_joules_per_10k_inferences"],
        },
        {
            "variant": "Structured L2 Pruned (-50% Filters)",
            "data_pillar": "Same dataset + 3 ep fine-tune",
            "model_params": s50["finetuned"]["total_params"],
            "model_nonzero_params": s50["finetuned"]["total_params"],
            "model_dense_mb": s50_bench["storage"]["dense_pt_mb"],
            "model_compressed_mb": s50_bench["storage"]["sparse_csr_npz_compressed_mb"],
            "test_acc_pct": s50["finetuned"]["test_acc"],
            "compute_theoretical_macs": s50["finetuned"]["dense_macs"],
            "compute_executed_macs": s50["finetuned"]["dense_macs"],
            "single_latency_ms": s50_bench["latency"]["single_mean_ms"],
            "batch_throughput_samples_s": s50_bench["latency"]["batch_throughput_samples_s"],
            "energy_j_per_10k": bench["energy"]["structured_50"]["energy_joules_per_10k_inferences"],
        },
        {
            "variant": "Plain Student CNN (~100k)",
            "data_pillar": "Same dataset, hard labels only (CE)",
            "model_params": distill["plain_student"]["params"],
            "model_nonzero_params": distill["plain_student"]["params"],
            "model_dense_mb": distill["plain_student"]["storage"]["dense_pt_mb"],
            "model_compressed_mb": distill["plain_student"]["storage"]["sparse_csr_npz_compressed_mb"],
            "test_acc_pct": distill["plain_student"]["test_acc"],
            "compute_theoretical_macs": distill["plain_student"]["macs"],
            "compute_executed_macs": distill["plain_student"]["macs"],
            "single_latency_ms": distill["plain_student"]["latency"]["single_mean_ms"],
            "batch_throughput_samples_s": distill["plain_student"]["latency"]["batch_throughput_samples_s"],
            "energy_j_per_10k": bench["energy"]["student_plain"]["energy_joules_per_10k_inferences"],
        },
        {
            "variant": "Distilled Student CNN (~100k, T=4)",
            "data_pillar": "Hard labels + soft Teacher logits (alpha=0.5)",
            "model_params": distill["distilled_student"]["params"],
            "model_nonzero_params": distill["distilled_student"]["params"],
            "model_dense_mb": distill["distilled_student"]["storage"]["dense_pt_mb"],
            "model_compressed_mb": distill["distilled_student"]["storage"]["sparse_csr_npz_compressed_mb"],
            "test_acc_pct": distill["distilled_student"]["test_acc"],
            "compute_theoretical_macs": distill["distilled_student"]["macs"],
            "compute_executed_macs": distill["distilled_student"]["macs"],
            "single_latency_ms": distill["distilled_student"]["latency"]["single_mean_ms"],
            "batch_throughput_samples_s": distill["distilled_student"]["latency"]["batch_throughput_samples_s"],
            "energy_j_per_10k": bench["energy"]["student_distilled"]["energy_joules_per_10k_inferences"],
        },
    ]

    summary = {"rows": rows}
    save_json(summary, results_dir / "summary_pillars.json")
    print("Saved summary_pillars.json.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lab 7 compression experiments.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["baseline", "pruning", "distillation", "benchmark", "summary", "all"],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.step in ("baseline", "all"):
        run_task1_baseline(cfg)
    if args.step in ("pruning", "all"):
        run_task2_pruning(cfg)
    if args.step in ("distillation", "all"):
        run_task4_distillation(cfg)
    if args.step in ("benchmark", "all"):
        run_task3_benchmark(cfg)
    if args.step in ("summary", "all"):
        run_task5_summary(cfg)


if __name__ == "__main__":
    main()
