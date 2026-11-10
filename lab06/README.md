# Lab 6: Training Pipelines and Project Proposal

## Overview
Reproducible YAML-driven training pipeline for the Lab 3 CNN model on FashionMNIST with deterministic seeding, checkpoint saving/resuming, MLflow experiment tracking, PyTorch CPU profiling, DataLoader worker scaling, batch-size trade-off analysis, optimizer/scheduler benchmarking, gradient accumulation, and an individual project proposal.

## Structure
- `config.yaml`: Single YAML configuration file driving `train.py`.
- `src/`:
  - `model.py`: Lab 3 `CNN` model definition and model factory.
  - `train.py`: Config-driven training script with seeding, checkpointing, MLflow logging, `torch.profiler` hooks, gradient accumulation, and resume support.
  - `experiments.py`: Automated runner for Tasks 2–7 and visualization generator.
- `results/`:
  - `profiler_breakdown.json`, `profiler_breakdown.png`: 1-epoch `torch.profiler` phase and operator breakdown.
  - `num_workers_study.json`, `num_workers_plot.png`: DataLoader `num_workers` `[0, 1, 2, 4, 8]` scaling and bottleneck analysis.
  - `batch_size_study.json`, `batch_size_study.png`: Batch-size `[32, 64, 128, 256, 512]` throughput, memory, and 5-epoch accuracy trade-offs.
  - `lr_opt_comparison.json`, `lr_opt_comparison.png`: 10-epoch comparison of `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` across `Adam` and `SGD+momentum`.
  - `grad_accum_study.json`, `grad_accum_comparison.png`: Batch 256 vs. Batch 64 with accumulation 4 comparison.
  - `resume_study.json`, `resume_training_curves.png`: Uninterrupted vs. epoch-3 interrupted & resumed training continuity verification.
  - `mlflow_runs_summary.json`: Exported summary of logged MLflow runs.
- `report.md`: Engineering report with graded evidence, tables, plots, and system analysis.
- `proposal.md`: One-page individual project proposal.

## Quickstart
Run the YAML-driven training pipeline:
```bash
python lab06/src/train.py --config lab06/config.yaml
```

Run individual or all lab experiments:
```bash
python lab06/src/experiments.py --step all
```
