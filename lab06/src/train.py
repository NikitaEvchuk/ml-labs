"""YAML-driven training pipeline for Lab 6 (FashionMNIST CNN from Lab 3)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from lab06.src.model import build_model
except ImportError:
    from model import build_model


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a single YAML configuration object from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}, got {type(cfg).__name__}")
    return cfg


def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten nested configuration dicts for MLflow parameter logging."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture RNG states for deterministic checkpoint resuming."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_states(
    state: dict[str, Any], loader_generator: torch.Generator | None = None
) -> None:
    """Restore saved RNG states from a checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])


class MemoryTracker:
    """Background RSS memory sampler using psutil."""

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self._proc = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            rss_mb = self._proc.memory_info().rss / (1024 * 1024)
            if rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = rss_mb
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemoryTracker":
        self.start_rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        self.peak_rss_mb = self.start_rss_mb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        final_rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        if final_rss_mb > self.peak_rss_mb:
            self.peak_rss_mb = final_rss_mb

    @property
    def delta_rss_mb(self) -> float:
        return max(0.0, self.peak_rss_mb - self.start_rss_mb)


def build_dataloaders(
    cfg: dict[str, Any], loader_generator: torch.Generator
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build FashionMNIST train, validation, and test DataLoaders from config."""
    data_cfg = cfg["data"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    root = Path(data_cfg["root"])
    download = bool(data_cfg.get("download", True))
    full_train = datasets.FashionMNIST(
        root=str(root), train=True, download=download, transform=transform
    )
    test_ds = datasets.FashionMNIST(
        root=str(root), train=False, download=download, transform=transform
    )

    split_gen = torch.Generator().manual_seed(int(cfg["seed"]))
    train_size = int(data_cfg.get("train_size", 55000))
    val_size = int(data_cfg.get("val_size", 5000))
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=split_gen)

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_optimizer(model: nn.Module, opt_cfg: dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from config."""
    name = str(opt_cfg["name"]).lower()
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name in ("sgd", "sgd_momentum"):
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']!r}")


def build_scheduler(
    optimizer: optim.Optimizer, sched_cfg: dict[str, Any]
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create learning-rate scheduler from config."""
    name = str(sched_cfg.get("name", "none")).lower()
    if name in ("none", "constant", "null"):
        return None
    if name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 3)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
    if name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("T_max", 10)),
            eta_min=float(sched_cfg.get("eta_min", 1e-5)),
        )
    if name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 2)),
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg.get('name')!r}")


def step_scheduler(scheduler: Any, val_loss: float) -> None:
    """Step scheduler uniformly regardless of plateau vs epoch-based API."""
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate loss and accuracy on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def measure_step_peak_memory_mb(
    model: nn.Module, batch_size: int, device: torch.device
) -> float:
    """Measure peak CPU tensor memory allocated during one forward+backward step."""
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model_copy.parameters(), lr=1e-3)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        opt.zero_grad()
        out = model_copy(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

    events = prof.key_averages()
    total_alloc_bytes = sum(max(0, evt.cpu_memory_usage) for evt in events)
    peak_single_op_bytes = max((max(0, evt.cpu_memory_usage) for evt in events), default=0)
    # Return total positive tensor allocations across the step (in MB)
    _ = peak_single_op_bytes
    return total_alloc_bytes / (1024 * 1024)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    enable_profiler: bool = False,
) -> dict[str, Any]:
    """Run one training epoch, optionally wrapped with torch.profiler."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    data_time_s = 0.0
    compute_time_s = 0.0

    optimizer.zero_grad()

    def _run_loop(prof_ctx: Any = None) -> None:
        nonlocal total_loss, correct, total, data_time_s, compute_time_s
        data_iter = iter(loader)
        for batch_idx in range(num_batches):
            t_d0 = time.perf_counter()
            if prof_ctx is not None:
                with record_function("data_loading"):
                    x, y = next(data_iter)
            else:
                x, y = next(data_iter)
            t_d1 = time.perf_counter()
            data_time_s += t_d1 - t_d0

            t_c0 = time.perf_counter()
            x, y = x.to(device), y.to(device)

            if prof_ctx is not None:
                with record_function("forward_pass"):
                    out = model(x)
                    loss = criterion(out, y)
            else:
                out = model(x)
                loss = criterion(out, y)

            scaled_loss = loss / grad_accum_steps

            if prof_ctx is not None:
                with record_function("backward_pass"):
                    scaled_loss.backward()
            else:
                scaled_loss.backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                (batch_idx + 1) == num_batches
            )
            if should_step:
                if prof_ctx is not None:
                    with record_function("optimizer_step"):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
            t_c1 = time.perf_counter()
            compute_time_s += t_c1 - t_c0

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    t0 = time.perf_counter()
    profiler_summary: dict[str, Any] | None = None
    if enable_profiler:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            _run_loop(prof_ctx=prof)
        profiler_summary = extract_profiler_summary(prof)
    else:
        _run_loop(prof_ctx=None)
    t1 = time.perf_counter()

    epoch_time_s = t1 - t0
    return {
        "train_loss": total_loss / total,
        "train_acc": correct / total,
        "epoch_time_s": epoch_time_s,
        "data_time_s": data_time_s,
        "compute_time_s": compute_time_s,
        "throughput_samples_s": total / epoch_time_s if epoch_time_s > 0 else 0.0,
        "num_samples": total,
        "profiler_summary": profiler_summary,
        "profiler_obj": prof if enable_profiler else None,
    }


def extract_profiler_summary(prof: profile) -> dict[str, Any]:
    """Extract phase shares and top operators from a torch.profiler run."""
    key_avgs = prof.key_averages()
    phases = ["data_loading", "forward_pass", "backward_pass", "optimizer_step"]
    phase_ms: dict[str, float] = {p: 0.0 for p in phases}

    for evt in key_avgs:
        if evt.key in phase_ms:
            # cpu_time_total is in microseconds in PyTorch 2.3
            phase_ms[evt.key] = evt.cpu_time_total / 1000.0

    total_step_ms = sum(phase_ms.values())
    phase_shares = {
        p: (phase_ms[p] / total_step_ms * 100.0) if total_step_ms > 0 else 0.0
        for p in phases
    }

    top_ops = []
    sorted_evts = sorted(key_avgs, key=lambda e: e.self_cpu_time_total, reverse=True)
    for evt in sorted_evts:
        if evt.key in phases:
            continue
        top_ops.append(
            {
                "name": evt.key,
                "self_cpu_time_ms": round(evt.self_cpu_time_total / 1000.0, 3),
                "cpu_time_total_ms": round(evt.cpu_time_total / 1000.0, 3),
                "count": int(evt.count),
            }
        )
        if len(top_ops) >= 12:
            break

    return {
        "phase_time_ms": {k: round(v, 3) for k, v in phase_ms.items()},
        "phase_share_pct": {k: round(v, 2) for k, v in phase_shares.items()},
        "total_profiled_step_ms": round(total_step_ms, 3),
        "top_ops": top_ops,
        "table": key_avgs.table(sort_by="self_cpu_time_total", row_limit=20),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    best_val_acc: float,
    history: list[dict[str, Any]],
    loader_generator: torch.Generator,
    cfg: dict[str, Any],
) -> None:
    """Save full training state checkpoint for fault-tolerant resuming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "history": history,
        "rng_states": get_rng_states(loader_generator),
        "config": cfg,
    }
    torch.save(payload, path)


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute training driven strictly by a single YAML configuration dict."""
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, loader_gen)

    model = build_model(
        cfg["model"]["name"], num_classes=int(cfg["model"].get("num_classes", 10))
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {"name": "none"}))

    train_cfg = cfg["training"]
    total_epochs = int(train_cfg["epochs"])
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    interrupt_after = train_cfg.get("interrupt_after_epoch")
    resume_from = train_cfg.get("resume_from")

    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("dir", "lab06/results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_tag = cfg.get("mlflow", {}).get("run_name", "run")
    last_ckpt_path = ckpt_dir / f"{run_tag}_last.pt"
    best_ckpt_path = ckpt_dir / f"{run_tag}_best.pt"

    start_epoch = 1
    best_val_acc = -1.0
    history: list[dict[str, Any]] = []

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        history = list(ckpt.get("history", []))
        if "rng_states" in ckpt:
            restore_rng_states(ckpt["rng_states"], loader_gen)
        start_epoch = int(ckpt["epoch"]) + 1

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    prof_cfg = cfg.get("profiling", {})
    profiling_enabled = bool(prof_cfg.get("enabled", False))

    step_tensor_mem_mb = measure_step_peak_memory_mb(
        model, int(cfg["data"]["batch_size"]), device
    )

    active_run = None
    mlflow_run_id: str | None = None
    if mlflow_enabled:
        tracking_path = Path(mlflow_cfg.get("tracking_uri", "lab06/mlruns")).resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_path.as_uri())
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "lab06_fashionmnist_cnn"))
        active_run = mlflow.start_run(run_name=run_tag)
        mlflow_run_id = active_run.info.run_id
        flat_params = flatten_dict(cfg)
        # MLflow param values are stored as strings
        mlflow.log_params({k: str(v) for k, v in flat_params.items()})
        mlflow.log_param("device", str(device))
        mlflow.log_param("num_params", sum(p.numel() for p in model.parameters()))

    profiler_result: dict[str, Any] | None = None
    interrupted = False

    try:
        with MemoryTracker() as mem_tracker:
            for epoch in range(start_epoch, total_epochs + 1):
                do_profile = profiling_enabled and (epoch == start_epoch)
                ep_res = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    grad_accum_steps=grad_accum_steps,
                    enable_profiler=do_profile,
                )

                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                current_lr = optimizer.param_groups[0]["lr"]

                row = {
                    "epoch": epoch,
                    "train_loss": round(ep_res["train_loss"], 6),
                    "train_acc": round(ep_res["train_acc"], 6),
                    "val_loss": round(val_loss, 6),
                    "val_acc": round(val_acc, 6),
                    "lr": float(current_lr),
                    "epoch_time_s": round(ep_res["epoch_time_s"], 4),
                    "data_time_s": round(ep_res["data_time_s"], 4),
                    "compute_time_s": round(ep_res["compute_time_s"], 4),
                    "throughput_samples_s": round(ep_res["throughput_samples_s"], 2),
                }
                history.append(row)

                if ep_res["profiler_summary"] is not None:
                    profiler_result = ep_res["profiler_summary"]
                    if prof_cfg.get("export_trace", False) and ep_res["profiler_obj"] is not None:
                        trace_path = Path(
                            prof_cfg.get("trace_path", "lab06/results/profiler_trace.json")
                        )
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        ep_res["profiler_obj"].export_chrome_trace(str(trace_path))

                if mlflow_enabled:
                    mlflow.log_metrics(
                        {
                            "train_loss": row["train_loss"],
                            "train_acc": row["train_acc"],
                            "val_loss": row["val_loss"],
                            "val_acc": row["val_acc"],
                            "lr": row["lr"],
                            "epoch_time_s": row["epoch_time_s"],
                            "throughput_samples_s": row["throughput_samples_s"],
                        },
                        step=epoch,
                    )

                step_scheduler(scheduler, val_loss)

                if ckpt_cfg.get("save_best", True) and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(
                        best_ckpt_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        best_val_acc,
                        history,
                        loader_gen,
                        cfg,
                    )

                if ckpt_cfg.get("save_last", True):
                    save_checkpoint(
                        last_ckpt_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        best_val_acc,
                        history,
                        loader_gen,
                        cfg,
                    )

                print(
                    f"[{run_tag}] Epoch {epoch}/{total_epochs} | "
                    f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']*100:.2f}% | "
                    f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']*100:.2f}% | "
                    f"lr={current_lr:.6f} | {row['throughput_samples_s']:.1f} samples/s"
                )

                if interrupt_after is not None and epoch == int(interrupt_after):
                    interrupted = True
                    print(f"[{run_tag}] Simulated interruption after epoch {epoch}. Saved {last_ckpt_path}")
                    break

        test_loss, test_acc = (None, None)
        if train_cfg.get("evaluate_test", True) and not interrupted:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        avg_throughput = (
            float(np.mean([h["throughput_samples_s"] for h in history])) if history else 0.0
        )

        summary_metrics = {
            "run_name": run_tag,
            "mlflow_run_id": mlflow_run_id,
            "interrupted": interrupted,
            "epochs_completed": len(history),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "final_val_acc": history[-1]["val_acc"] if history else None,
            "best_val_acc": round(best_val_acc, 6) if best_val_acc >= 0 else None,
            "test_loss": round(test_loss, 6) if test_loss is not None else None,
            "test_acc": round(test_acc, 6) if test_acc is not None else None,
            "avg_throughput_samples_s": round(avg_throughput, 2),
            "peak_rss_mb": round(mem_tracker.peak_rss_mb, 2),
            "delta_rss_mb": round(mem_tracker.delta_rss_mb, 2),
            "step_tensor_mem_mb": round(step_tensor_mem_mb, 3),
            "last_checkpoint_path": str(last_ckpt_path),
            "best_checkpoint_path": str(best_ckpt_path),
            "history": history,
            "profiler": profiler_result,
        }

        if mlflow_enabled:
            final_log = {
                "avg_throughput_samples_s": summary_metrics["avg_throughput_samples_s"],
                "peak_rss_mb": summary_metrics["peak_rss_mb"],
                "delta_rss_mb": summary_metrics["delta_rss_mb"],
                "step_tensor_mem_mb": summary_metrics["step_tensor_mem_mb"],
            }
            if test_acc is not None:
                final_log["test_acc"] = float(test_acc)
                final_log["test_loss"] = float(test_loss)
            mlflow.log_metrics(final_log)

            with tempfile.TemporaryDirectory() as tmpdir:
                cfg_dump = Path(tmpdir) / "resolved_config.yaml"
                with open(cfg_dump, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                mlflow.log_artifact(str(cfg_dump), artifact_path="config")

                hist_dump = Path(tmpdir) / "history.json"
                with open(hist_dump, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
                mlflow.log_artifact(str(hist_dump), artifact_path="metrics")

            if best_ckpt_path.exists():
                mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
            elif last_ckpt_path.exists():
                mlflow.log_artifact(str(last_ckpt_path), artifact_path="checkpoints")

            if profiler_result is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    prof_dump = Path(tmpdir) / "profiler_summary.json"
                    with open(prof_dump, "w", encoding="utf-8") as f:
                        json.dump(profiler_result, f, indent=2)
                    mlflow.log_artifact(str(prof_dump), artifact_path="profiler")

    finally:
        if active_run is not None:
            mlflow.end_run()

    return summary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML-driven FashionMNIST CNN training.")
    parser.add_argument(
        "--config",
        type=str,
        default="lab06/config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    res = run_training(cfg)
    print(f"\nCompleted run '{res['run_name']}' | test_acc={res['test_acc']} | mlflow_run_id={res['mlflow_run_id']}")


if __name__ == "__main__":
    main()
