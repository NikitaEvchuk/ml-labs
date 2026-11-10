"""Experiment runner and plotting suite for Lab 6 (Tasks 2-7)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

try:
    from lab06.src.train import load_config, run_training
except ImportError:
    from train import load_config, run_training

BASE_CONFIG_PATH = Path("lab06/config.yaml")
RESULTS_DIR = Path("lab06/results")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_cfg(base_cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Deep-copy base config and apply dotted overrides."""
    cfg = copy.deepcopy(base_cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = val
    return cfg


def run_task2_profiler(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 2: Profile one epoch with torch.profiler."""
    print("\n=== TASK 2: Profiling 1 Epoch with torch.profiler ===")
    cfg = make_cfg(
        base_cfg,
        **{
            "training.epochs": 1,
            "training.evaluate_test": False,
            "profiling.enabled": True,
            "profiling.export_trace": False,
            "mlflow.run_name": "task2_profiler_1epoch",
        },
    )
    res = run_training(cfg)
    prof_data = {
        "mlflow_run_id": res["mlflow_run_id"],
        "epoch_time_s": res["history"][0]["epoch_time_s"],
        "throughput_samples_s": res["history"][0]["throughput_samples_s"],
        "profiler": res["profiler"],
    }
    save_json(prof_data, RESULTS_DIR / "profiler_breakdown.json")
    plot_task2_profiler(prof_data)
    return prof_data


def plot_task2_profiler(prof_data: dict[str, Any]) -> None:
    prof = prof_data["profiler"]
    shares = prof["phase_share_pct"]
    times = prof["phase_time_ms"]
    labels = ["Data Loading", "Forward Pass", "Backward Pass", "Optimizer Step"]
    keys = ["data_loading", "forward_pass", "backward_pass", "optimizer_step"]
    vals_pct = [shares[k] for k in keys]
    vals_ms = [times[k] for k in keys]
    colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = ax1.barh(labels[::-1], vals_pct[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
    for bar, pct, ms in zip(bars, vals_pct[::-1], vals_ms[::-1]):
        ax1.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({ms/1000:.2f}s)",
            va="center",
            fontsize=9,
        )
    ax1.set_xlim(0, max(vals_pct) + 32)
    ax1.set_xlabel("Share of Training Step Time (%)")
    ax1.set_title("1-Epoch CPU Profiler Breakdown by Phase")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    top_ops = prof["top_ops"][:8]
    op_names = [op["name"][:28] for op in top_ops][::-1]
    op_ms = [op["self_cpu_time_ms"] / 1000.0 for op in top_ops][::-1]
    ax2.barh(op_names, op_ms, color="#72B7B2", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(op_ms):
        ax2.text(v + 0.05, i, f"{v:.2f}s", va="center", fontsize=8)
    ax2.set_xlim(0, max(op_ms) * 1.25 if op_ms else 1)
    ax2.set_xlabel("Self CPU Time (s)")
    ax2.set_title("Top PyTorch Operators (self_cpu_time_total)")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = RESULTS_DIR / "profiler_breakdown.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_task3_num_workers(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Task 3: Vary num_workers [0, 1, 2, 4, 8] and identify bottleneck."""
    print("\n=== TASK 3: DataLoader num_workers Study [0, 1, 2, 4, 8] ===")
    rows = []
    for nw in [0, 1, 2, 4, 8]:
        cfg = make_cfg(
            base_cfg,
            **{
                "data.num_workers": nw,
                "training.epochs": 1,
                "training.evaluate_test": False,
                "mlflow.run_name": f"task3_num_workers_{nw}",
            },
        )
        res = run_training(cfg)
        h = res["history"][0]
        rows.append(
            {
                "num_workers": nw,
                "epoch_time_s": h["epoch_time_s"],
                "data_time_s": h["data_time_s"],
                "compute_time_s": h["compute_time_s"],
                "data_share_pct": round(h["data_time_s"] / h["epoch_time_s"] * 100.0, 2),
                "throughput_samples_s": h["throughput_samples_s"],
                "peak_rss_mb": res["peak_rss_mb"],
                "mlflow_run_id": res["mlflow_run_id"],
            }
        )
    save_json(rows, RESULTS_DIR / "num_workers_study.json")
    plot_task3_num_workers(rows)
    return rows


def plot_task3_num_workers(rows: list[dict[str, Any]]) -> None:
    workers = [str(r["num_workers"]) for r in rows]
    throughputs = [r["throughput_samples_s"] for r in rows]
    data_t = [r["data_time_s"] for r in rows]
    comp_t = [r["compute_time_s"] for r in rows]
    spawn_t = [max(0.0, r["epoch_time_s"] - r["data_time_s"] - r["compute_time_s"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(workers))
    ax1.plot(x, throughputs, marker="o", linewidth=2, color="#4C78A8")
    for xi, val in zip(x, throughputs):
        ax1.annotate(f"{val:.0f}", (xi, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workers)
    ax1.set_xlabel("num_workers")
    ax1.set_ylabel("Throughput (samples/s)")
    ax1.set_title("Training Throughput vs. DataLoader num_workers")
    ax1.set_ylim(0, max(throughputs) * 1.15)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(x, data_t, label="In-Loop Data Fetch (next)", color="#F58518", edgecolor="black", linewidth=0.5)
    ax2.bar(x, comp_t, bottom=data_t, label="Forward + Backward + Step", color="#54A24B", edgecolor="black", linewidth=0.5)
    bottom2 = [d + c for d, c in zip(data_t, comp_t)]
    ax2.bar(x, spawn_t, bottom=bottom2, label="Windows Worker Spawn/Teardown", color="#E45756", edgecolor="black", linewidth=0.5)
    for xi, r in zip(x, rows):
        ax2.text(xi, r["epoch_time_s"] + 1.0, f"{r['epoch_time_s']:.1f}s", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(workers)
    ax2.set_xlabel("num_workers")
    ax2.set_ylabel("1-Epoch Wall Time (s)")
    ax2.set_title("Epoch Wall-Time Breakdown (Data, Compute, Spawn)")
    ax2.set_ylim(0, max(r["epoch_time_s"] for r in rows) * 1.15)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "num_workers_plot.png", dpi=200)
    plt.close()


def run_task4_batch_size(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Task 4: Batch-size study [32, 64, 128, 256, 512] over 5 epochs."""
    print("\n=== TASK 4: Batch-Size Study [32, 64, 128, 256, 512] ===")
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        cfg = make_cfg(
            base_cfg,
            **{
                "data.batch_size": bs,
                "training.epochs": 5,
                "training.evaluate_test": True,
                "mlflow.run_name": f"task4_bs_{bs}",
            },
        )
        res = run_training(cfg)
        rows.append(
            {
                "batch_size": bs,
                "avg_throughput_samples_s": res["avg_throughput_samples_s"],
                "peak_rss_mb": res["peak_rss_mb"],
                "delta_rss_mb": res["delta_rss_mb"],
                "step_tensor_mem_mb": res["step_tensor_mem_mb"],
                "final_val_acc": res["final_val_acc"],
                "test_acc": res["test_acc"],
                "total_time_s": round(sum(h["epoch_time_s"] for h in res["history"]), 2),
                "mlflow_run_id": res["mlflow_run_id"],
            }
        )
    save_json(rows, RESULTS_DIR / "batch_size_study.json")
    plot_task4_batch_size(rows)
    return rows


def plot_task4_batch_size(rows: list[dict[str, Any]]) -> None:
    bs_labels = [str(r["batch_size"]) for r in rows]
    x = np.arange(len(bs_labels))
    throughput = [r["avg_throughput_samples_s"] for r in rows]
    tensor_mem = [r["step_tensor_mem_mb"] for r in rows]
    rss_mem = [r["peak_rss_mb"] for r in rows]
    test_acc = [r["test_acc"] * 100.0 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(x, throughput, marker="o", linewidth=2, color="#4C78A8")
    for xi, val in zip(x, throughput):
        axes[0].annotate(f"{val:.0f}", (xi, val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bs_labels)
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Throughput (samples/s)")
    axes[0].set_title("Throughput vs. Batch Size")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    ax_m = axes[1]
    ax_m.bar(x, tensor_mem, color="#F58518", edgecolor="black", linewidth=0.5, label="Step Tensor Alloc (MB)")
    for xi, val in zip(x, tensor_mem):
        ax_m.text(xi, val + 0.3, f"{val:.1f}", ha="center", fontsize=8)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(bs_labels)
    ax_m.set_xlabel("Batch Size")
    ax_m.set_ylabel("Step Tensor Allocation (MB)")
    ax_m.set_title("Step Tensor Memory & Peak RSS")
    ax_m.grid(axis="y", linestyle="--", alpha=0.5)
    ax_r = ax_m.twinx()
    ax_r.plot(x, rss_mem, marker="s", color="#E45756", linewidth=1.8, label="Peak RSS (MB)")
    ax_r.set_ylabel("Peak Process RSS (MB)", color="#E45756")

    axes[2].plot(x, test_acc, marker="^", linewidth=2, color="#54A24B")
    for xi, val in zip(x, test_acc):
        axes[2].annotate(f"{val:.2f}%", (xi, val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(bs_labels)
    axes[2].set_xlabel("Batch Size")
    axes[2].set_ylabel("Final Test Accuracy (%)")
    axes[2].set_title("Test Accuracy (5 Epochs) vs. Batch Size")
    axes[2].set_ylim(min(test_acc) - 2.0, max(test_acc) + 1.5)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "batch_size_study.png", dpi=200)
    plt.close()


def run_task5_lr_and_optimizers(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Task 5: Compare LR schedules and SGD+momentum vs Adam under 10-epoch budgets."""
    print("\n=== TASK 5: LR Schedules & Optimizers (10-Epoch Budget) ===")
    optimizers = [
        ("adam", 0.001, 0.0, "Adam (lr=1e-3)"),
        ("sgd", 0.02, 0.9, "SGD+Momentum (lr=0.02, m=0.9)"),
    ]
    schedulers = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    results = []

    for opt_name, lr, momentum, opt_label in optimizers:
        for sched_name in schedulers:
            run_name = f"task5_{opt_name}_{sched_name.lower()}"
            cfg = make_cfg(
                base_cfg,
                **{
                    "training.epochs": 10,
                    "training.evaluate_test": True,
                    "optimizer.name": opt_name,
                    "optimizer.lr": lr,
                    "optimizer.momentum": momentum,
                    "scheduler.name": sched_name,
                    "scheduler.T_max": 10,
                    "scheduler.step_size": 3,
                    "scheduler.gamma": 0.5,
                    "scheduler.factor": 0.5,
                    "scheduler.patience": 2,
                    "mlflow.run_name": run_name,
                },
            )
            res = run_training(cfg)
            results.append(
                {
                    "optimizer": opt_name,
                    "optimizer_label": opt_label,
                    "initial_lr": lr,
                    "scheduler": sched_name,
                    "final_train_acc": res["final_train_acc"],
                    "final_val_acc": res["final_val_acc"],
                    "best_val_acc": res["best_val_acc"],
                    "test_acc": res["test_acc"],
                    "final_val_loss": res["final_val_loss"],
                    "history": res["history"],
                    "mlflow_run_id": res["mlflow_run_id"],
                }
            )
    save_json(results, RESULTS_DIR / "lr_opt_comparison.json")
    plot_task5_lr_opt(results)
    return results


def plot_task5_lr_opt(results: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sched_styles = {
        "StepLR": ("#4C78A8", "-"),
        "CosineAnnealingLR": ("#54A24B", "--"),
        "ReduceLROnPlateau": ("#F58518", "-."),
    }

    for row_idx, opt_name in enumerate(["adam", "sgd"]):
        subset = [r for r in results if r["optimizer"] == opt_name]
        title_prefix = "Adam (lr=1e-3)" if opt_name == "adam" else "SGD + Momentum (lr=0.02)"
        ax_acc = axes[row_idx, 0]
        ax_lr = axes[row_idx, 1]

        for r in subset:
            epochs = [h["epoch"] for h in r["history"]]
            val_acc = [h["val_acc"] * 100.0 for h in r["history"]]
            lrs = [h["lr"] for h in r["history"]]
            color, ls = sched_styles[r["scheduler"]]
            ax_acc.plot(epochs, val_acc, label=f"{r['scheduler']} (test {r['test_acc']*100:.2f}%)", color=color, linestyle=ls, marker="o", markersize=4)
            ax_lr.plot(epochs, lrs, label=r["scheduler"], color=color, linestyle=ls, marker="s", markersize=4)

        ax_acc.set_title(f"{title_prefix}: Validation Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Validation Accuracy (%)")
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        ax_acc.legend(fontsize=8)

        ax_lr.set_title(f"{title_prefix}: Learning Rate Schedule")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.grid(True, linestyle="--", alpha=0.5)
        ax_lr.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lr_opt_comparison.png", dpi=200)
    plt.close()


def run_task6_grad_accum(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 6: Emulate batch 256 via batch 64 + grad_accum_steps 4."""
    print("\n=== TASK 6: Gradient Accumulation (bs=256 vs bs=64 x accum=4) ===")
    cfg_256 = make_cfg(
        base_cfg,
        **{
            "data.batch_size": 256,
            "training.epochs": 5,
            "training.grad_accum_steps": 1,
            "training.evaluate_test": True,
            "mlflow.run_name": "task6_bs256_accum1",
        },
    )
    cfg_64_accum = make_cfg(
        base_cfg,
        **{
            "data.batch_size": 64,
            "training.epochs": 5,
            "training.grad_accum_steps": 4,
            "training.evaluate_test": True,
            "mlflow.run_name": "task6_bs64_accum4",
        },
    )

    res_256 = run_training(cfg_256)
    res_64 = run_training(cfg_64_accum)

    out = {
        "batch_256_accum_1": {
            "effective_batch_size": 256,
            "test_acc": res_256["test_acc"],
            "final_val_acc": res_256["final_val_acc"],
            "final_train_loss": res_256["final_train_loss"],
            "avg_throughput_samples_s": res_256["avg_throughput_samples_s"],
            "peak_rss_mb": res_256["peak_rss_mb"],
            "step_tensor_mem_mb": res_256["step_tensor_mem_mb"],
            "history": res_256["history"],
            "mlflow_run_id": res_256["mlflow_run_id"],
        },
        "batch_64_accum_4": {
            "effective_batch_size": 256,
            "test_acc": res_64["test_acc"],
            "final_val_acc": res_64["final_val_acc"],
            "final_train_loss": res_64["final_train_loss"],
            "avg_throughput_samples_s": res_64["avg_throughput_samples_s"],
            "peak_rss_mb": res_64["peak_rss_mb"],
            "step_tensor_mem_mb": res_64["step_tensor_mem_mb"],
            "history": res_64["history"],
            "mlflow_run_id": res_64["mlflow_run_id"],
        },
    }
    save_json(out, RESULTS_DIR / "grad_accum_study.json")
    plot_task6_grad_accum(out)
    return out


def plot_task6_grad_accum(out: dict[str, Any]) -> None:
    r1 = out["batch_256_accum_1"]
    r2 = out["batch_64_accum_4"]
    ep1 = [h["epoch"] for h in r1["history"]]
    ep2 = [h["epoch"] for h in r2["history"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ep1, [h["train_loss"] for h in r1["history"]], "o-", label="bs=256 (train)", color="#4C78A8")
    ax1.plot(ep2, [h["train_loss"] for h in r2["history"]], "s--", label="bs=64, accum=4 (train)", color="#F58518")
    ax1.plot(ep1, [h["val_loss"] for h in r1["history"]], "o:", label="bs=256 (val)", color="#4C78A8", alpha=0.7)
    ax1.plot(ep2, [h["val_loss"] for h in r2["history"]], "s:", label="bs=64, accum=4 (val)", color="#F58518", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curves: True Batch 256 vs. Accumulated (64 x 4)")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(ep1, [h["val_acc"] * 100 for h in r1["history"]], "o-", label=f"bs=256 (test {r1['test_acc']*100:.2f}%)", color="#4C78A8")
    ax2.plot(ep2, [h["val_acc"] * 100 for h in r2["history"]], "s--", label=f"bs=64, accum=4 (test {r2['test_acc']*100:.2f}%)", color="#F58518")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Validation Accuracy: Effective Batch Size = 256")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "grad_accum_comparison.png", dpi=200)
    plt.close()


def run_task7_resume(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 7: Interrupt training at epoch 3, resume from checkpoint, verify smooth curves."""
    print("\n=== TASK 7: Interrupt at Epoch 3 & Resume to Epoch 10 ===")
    common = {
        "data.batch_size": 128,
        "training.epochs": 10,
        "scheduler.name": "CosineAnnealingLR",
        "scheduler.T_max": 10,
    }
    cfg_full = make_cfg(base_cfg, **common, **{"mlflow.run_name": "task7_uninterrupted_10ep"})
    res_full = run_training(cfg_full)

    cfg_part1 = make_cfg(
        base_cfg,
        **common,
        **{
            "training.interrupt_after_epoch": 3,
            "training.evaluate_test": False,
            "mlflow.run_name": "task7_interrupted_ep3",
        },
    )
    res_part1 = run_training(cfg_part1)

    ckpt_path = res_part1["last_checkpoint_path"]
    cfg_part2 = make_cfg(
        base_cfg,
        **common,
        **{
            "training.resume_from": ckpt_path,
            "training.evaluate_test": True,
            "mlflow.run_name": "task7_resumed_ep4_to_10",
        },
    )
    res_part2 = run_training(cfg_part2)

    out = {
        "uninterrupted": {
            "test_acc": res_full["test_acc"],
            "history": res_full["history"],
            "mlflow_run_id": res_full["mlflow_run_id"],
        },
        "interrupted_part1": {
            "epochs_completed": res_part1["epochs_completed"],
            "checkpoint_path": ckpt_path,
            "history": res_part1["history"],
            "mlflow_run_id": res_part1["mlflow_run_id"],
        },
        "resumed_part2": {
            "test_acc": res_part2["test_acc"],
            "history": res_part2["history"],
            "mlflow_run_id": res_part2["mlflow_run_id"],
        },
    }
    save_json(out, RESULTS_DIR / "resume_study.json")
    plot_task7_resume(out)
    return out


def plot_task7_resume(out: dict[str, Any]) -> None:
    h_full = out["uninterrupted"]["history"]
    h_resumed = out["resumed_part2"]["history"]

    ep_full = [h["epoch"] for h in h_full]
    ep_p1 = [h["epoch"] for h in h_resumed if h["epoch"] <= 3]
    ep_p2 = [h["epoch"] for h in h_resumed if h["epoch"] >= 3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ep_full, [h["train_loss"] for h in h_full], "o-", color="#4C78A8", alpha=0.6, linewidth=2.5, label="Uninterrupted (1-10)")
    ax1.plot(ep_p1, [h["train_loss"] for h in h_resumed if h["epoch"] <= 3], "s--", color="#F58518", label="Pre-interrupt (1-3)")
    ax1.plot(ep_p2, [h["train_loss"] for h in h_resumed if h["epoch"] >= 3], "^--", color="#54A24B", label="Resumed (3-10)")
    ax1.axvline(3, color="#E45756", linestyle=":", linewidth=1.5, label="Interrupt / Resume (Epoch 3)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss Continuity Across Checkpoint Resume")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(ep_full, [h["val_acc"] * 100 for h in h_full], "o-", color="#4C78A8", alpha=0.6, linewidth=2.5, label="Uninterrupted (1-10)")
    ax2.plot(ep_p1, [h["val_acc"] * 100 for h in h_resumed if h["epoch"] <= 3], "s--", color="#F58518", label="Pre-interrupt (1-3)")
    ax2.plot(ep_p2, [h["val_acc"] * 100 for h in h_resumed if h["epoch"] >= 3], "^--", color="#54A24B", label="Resumed (3-10)")
    ax2.axvline(3, color="#E45756", linestyle=":", linewidth=1.5, label="Interrupt / Resume (Epoch 3)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Validation Accuracy Continuity Across Checkpoint Resume")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "resume_training_curves.png", dpi=200)
    plt.close()


def export_mlflow_summary(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Query local MLflow tracking directory and export compact run summary."""
    tracking_path = Path(base_cfg["mlflow"]["tracking_uri"]).resolve()
    mlflow.set_tracking_uri(tracking_path.as_uri())
    exp = mlflow.get_experiment_by_name(base_cfg["mlflow"]["experiment_name"])
    if exp is None:
        return []
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    summary = []
    for _, row in df.iterrows():
        summary.append(
            {
                "run_id": row["run_id"],
                "run_name": row.get("tags.mlflow.runName"),
                "status": row["status"],
                "batch_size": row.get("params.data.batch_size"),
                "num_workers": row.get("params.data.num_workers"),
                "optimizer": row.get("params.optimizer.name"),
                "scheduler": row.get("params.scheduler.name"),
                "grad_accum_steps": row.get("params.training.grad_accum_steps"),
                "test_acc": row.get("metrics.test_acc"),
                "val_acc": row.get("metrics.val_acc"),
                "avg_throughput_samples_s": row.get("metrics.avg_throughput_samples_s"),
                "peak_rss_mb": row.get("metrics.peak_rss_mb"),
            }
        )
    save_json(summary, RESULTS_DIR / "mlflow_runs_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lab 6 experiments.")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "task2", "task3", "task4", "task5", "task6", "task7", "summary"],
        help="Which experiment step to execute.",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(BASE_CONFIG_PATH)

    if args.step in ("all", "task2"):
        run_task2_profiler(base_cfg)
    if args.step in ("all", "task3"):
        run_task3_num_workers(base_cfg)
    if args.step in ("all", "task4"):
        run_task4_batch_size(base_cfg)
    if args.step in ("all", "task5"):
        run_task5_lr_and_optimizers(base_cfg)
    if args.step in ("all", "task6"):
        run_task6_grad_accum(base_cfg)
    if args.step in ("all", "task7"):
        run_task7_resume(base_cfg)
    export_mlflow_summary(base_cfg)


if __name__ == "__main__":
    main()
