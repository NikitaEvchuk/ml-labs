"""Latency, sparse-kernel microbenchmark, and energy measurement utilities for Lab 7."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from codecarbon import OfflineEmissionsTracker


def measure_inference_latency(
    model: nn.Module,
    warmup_single: int = 50,
    iters_single: int = 400,
    warmup_batch: int = 15,
    iters_batch: int = 60,
    batch_size: int = 128,
) -> dict[str, float]:
    """Measure CPU inference latency at batch_size=1 and batch_size=128."""
    model.eval()
    x_single = torch.randn(1, 1, 28, 28)
    x_batch = torch.randn(batch_size, 1, 28, 28)

    single_times_ms: list[float] = []
    batch_times_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup_single):
            _ = model(x_single)
        for _ in range(iters_single):
            t0 = time.perf_counter()
            _ = model(x_single)
            single_times_ms.append((time.perf_counter() - t0) * 1000.0)

        for _ in range(warmup_batch):
            _ = model(x_batch)
        for _ in range(iters_batch):
            t0 = time.perf_counter()
            _ = model(x_batch)
            batch_times_ms.append((time.perf_counter() - t0) * 1000.0)

    s_arr = np.asarray(single_times_ms, dtype=np.float64)
    b_arr = np.asarray(batch_times_ms, dtype=np.float64)
    b_mean_s = float(b_arr.mean()) / 1000.0

    return {
        "single_mean_ms": float(s_arr.mean()),
        "single_p50_ms": float(np.percentile(s_arr, 50)),
        "single_p95_ms": float(np.percentile(s_arr, 95)),
        "single_std_ms": float(s_arr.std()),
        "batch_size": batch_size,
        "batch_mean_ms": float(b_arr.mean()),
        "batch_p50_ms": float(np.percentile(b_arr, 50)),
        "batch_p95_ms": float(np.percentile(b_arr, 95)),
        "batch_throughput_samples_s": float(batch_size / b_mean_s),
    }


def benchmark_dense_vs_csr_linear(
    in_features: int = 1440,
    out_features: int = 640,
    sparsities: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95),
    batch_size: int = 128,
    iters: int = 150,
) -> list[dict[str, Any]]:
    """Microbenchmark dense F.linear vs torch.sparse_csr matmul on FC1 dimensions."""
    torch.manual_seed(42)
    w_base = torch.randn(out_features, in_features)
    x_batch = torch.randn(batch_size, in_features)
    x_single = torch.randn(1, in_features)

    results: list[dict[str, Any]] = []

    with torch.inference_mode():
        # Dense baseline
        for _ in range(20):
            _ = F.linear(x_single, w_base)
            _ = F.linear(x_batch, w_base)

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.linear(x_single, w_base)
        dense_single_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.linear(x_batch, w_base)
        dense_batch_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        results.append(
            {
                "kernel": "dense_blas",
                "sparsity_pct": 0.0,
                "single_ms": dense_single_ms,
                "batch_ms": dense_batch_ms,
                "speedup_vs_dense_single": 1.0,
                "speedup_vs_dense_batch": 1.0,
            }
        )

        for sp_level in sparsities:
            w = w_base.clone()
            k = int(round(sp_level * w.numel()))
            order = torch.argsort(w.abs().view(-1))
            mask = torch.ones_like(w).view(-1)
            mask[order[:k]] = 0.0
            w = w * mask.view_as(w)
            w_csr = w.to_sparse_csr()

            # PyTorch sparse_csr @ dense_2d requires (M, K) @ (K, N)
            x_single_t = x_single.t().contiguous()
            x_batch_t = x_batch.t().contiguous()

            for _ in range(20):
                _ = torch.matmul(w_csr, x_single_t)
                _ = torch.matmul(w_csr, x_batch_t)

            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(w_csr, x_single_t)
            csr_single_ms = ((time.perf_counter() - t0) / iters) * 1000.0

            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(w_csr, x_batch_t)
            csr_batch_ms = ((time.perf_counter() - t0) / iters) * 1000.0

            results.append(
                {
                    "kernel": "torch_sparse_csr",
                    "sparsity_pct": sp_level * 100.0,
                    "single_ms": csr_single_ms,
                    "batch_ms": csr_batch_ms,
                    "speedup_vs_dense_single": dense_single_ms / csr_single_ms,
                    "speedup_vs_dense_batch": dense_batch_ms / csr_batch_ms,
                }
            )

    return results


def measure_inference_energy(
    model: nn.Module,
    test_tensor: torch.Tensor,
    repeats: int = 30,
    batch_size: int = 256,
    country_iso_code: str = "POL",
) -> dict[str, float]:
    """Measure inference energy using CodeCarbon OfflineEmissionsTracker over `repeats` passes."""
    model.eval()
    n_samples = test_tensor.size(0)
    total_inferences = repeats * n_samples
    proc = psutil.Process(os.getpid())

    # Warmup pass
    with torch.inference_mode():
        for start in range(0, n_samples, batch_size):
            _ = model(test_tensor[start : start + batch_size])

    tracker = OfflineEmissionsTracker(
        country_iso_code=country_iso_code,
        log_level="error",
        save_to_file=False,
    )
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)
    tracker.start()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            for start in range(0, n_samples, batch_size):
                _ = model(test_tensor[start : start + batch_size])
    elapsed_s = time.perf_counter() - t0
    emissions_kg = float(tracker.stop() or 0.0)
    rss_after_mb = proc.memory_info().rss / (1024 * 1024)

    data = tracker.final_emissions_data
    energy_kwh = float(getattr(data, "energy_consumed", 0.0))
    energy_j = energy_kwh * 3_600_000.0
    energy_j_per_10k = energy_j * (10_000.0 / total_inferences)
    avg_power_w = energy_j / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "total_inferences": total_inferences,
        "elapsed_s": elapsed_s,
        "throughput_samples_s": total_inferences / elapsed_s,
        "energy_kwh": energy_kwh,
        "energy_joules_total": energy_j,
        "energy_joules_per_10k_inferences": energy_j_per_10k,
        "avg_power_watts": avg_power_w,
        "emissions_g_co2": emissions_kg * 1000.0,
        "rss_mb": max(rss_before_mb, rss_after_mb),
    }
