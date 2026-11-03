# Lab 5: Frameworks and Automatic Differentiation

## Overview
Implementation of a scalar automatic-differentiation engine from first principles, verification against PyTorch autograd and numerical finite differences, training a 2 → 8 → 1 neural network on `make_moons`, and benchmarking PyTorch execution modes (eager vs. traced vs. scripted) and memory contiguity effects.

## Structure
- `src/`:
  - `autograd.py`: Implementation of `Value` node, computation graph construction, topological backpropagation, `Tensor` vectorized container, `TinyMLP` architecture, and PyTorch benchmarking routines.
- `results/`:
  - `gradient_verification.json`: Comparison of analytical, PyTorch autograd, and central-difference numerical gradients.
  - `training_benchmark.json`: 5-run training speed benchmark and loss trajectories comparing custom autograd vs. PyTorch.
  - `pytorch_benchmark.json`: Graph structure, execution mode latencies (eager, traced, scripted), and tensor contiguity benchmark.
  - `benchmark_plots.png`: Multi-panel visualization comparing training loss curves, training runtime, inference latency across execution modes, and non-contiguous memory access penalty.
- `report.md`: Detailed engineering report answering all lab tasks and presenting graded evidence.

## Quickstart
To run the full suite and generate all artifacts:
```bash
python lab05/src/autograd.py
```
