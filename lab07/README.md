# Lab 7: Compression — Pruning and Distillation

## Overview
Model compression experiments on **Fashion-MNIST** comparing an oversized baseline CNN (~1.01M parameters) against unstructured magnitude pruning (global and per-layer), structured L2-norm filter pruning, and knowledge distillation into a compact ~100k-parameter student CNN across accuracy, parameter count, dense vs. sparse disk size, CPU inference latency, MACs, and energy.

## Structure
- `config.yaml`: Single YAML configuration file driving all Lab 7 tasks.
- `src/`:
  - `models.py`: `OversizedCNN` (~1.01M params), `StudentCNN` (~99.6k params), deterministic Fashion-MNIST TensorDataset loader, and MAC/parameter helpers.
  - `pruning.py`: Global and layer-wise magnitude masking, fixed-mask fine-tuning, structured L2 filter removal & network reconstruction, and dense vs. CSR sparse storage measurement.
  - `distillation.py`: Plain student training vs. Knowledge Distillation (`T=4.0`, `alpha=0.5`).
  - `benchmark.py`: Single-sample (`bs=1`) and batched (`bs=128`) CPU latency profiling, dense BLAS vs. `torch.sparse_csr` linear microbenchmark, and `codecarbon` energy measurement.
  - `experiments.py`: Automated runner for Tasks 1–5 and visualization generator.
- `results/`:
  - `baseline_metrics.json`: Task 1 oversized baseline CNN metrics.
  - `pruning_results.json`, `pruning_sparsity_curves.png`: Task 2 unstructured (global/layer-wise) and structured pruning results and sparsity curves.
  - `latency_storage_results.json`, `latency_and_size_analysis.png`: Task 3 latency, dense vs. sparse disk sizes, CSR kernel microbenchmark, and energy measurements.
  - `distillation_results.json`, `distillation_comparison.png`: Task 4 plain vs. distilled student comparison.
  - `summary_pillars.json`: Task 5 unified comparison across Reddi Ch. 9 pillars (Data, Model, Compute, Energy).
- `report.md`: Engineering report with graded evidence, analysis, and conclusion.

## Quickstart
Run the full Lab 7 suite:
```bash
python lab07/src/experiments.py --step all
```

Run an individual stage (`baseline`, `pruning`, `distillation`, `benchmark`, `summary`):
```bash
python lab07/src/experiments.py --step pruning
```
