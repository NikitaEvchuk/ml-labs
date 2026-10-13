# Lab 2: A Neural Network from Scratch

## Overview
Implementation of a two-hidden-layer fully connected neural network (Multilayer Perceptron) completely from scratch using NumPy, without high-level deep learning frameworks. The model is trained on the handwritten digits classification task (`load_digits`), validated via analytical vs. numerical gradient checking, and profiled for execution latency.

## Architecture & Specifications
- **Dataset**: Optical Recognition of Handwritten Digits (`load_digits`), scaled by 16, one-hot encoded targets, stratified 70/30 train/test split (`random_state=42`).
- **Network Layout**: 
  - Input: 64 features (8x8 pixel values)
  - Hidden Layer 1: Width $W \in \{16, 64\}$ + ReLU activation
  - Hidden Layer 2: Width $W \in \{16, 64\}$ + ReLU activation
  - Output Layer: 10 units + Softmax activation
- **Weight Initialization**: Gaussian distribution $\mathcal{N}(0, 0.01)$, biases initialized to zeros.
- **Loss Function**: Multiclass Cross-Entropy loss (initial untrained loss sanity check $\approx \ln(10) \approx 2.3026$).

## Key Tasks & Implementation Details
1. **Forward Propagation**: Fully vectorized matrix transformations with numerically stable Softmax normalization.
2. **Backpropagation**: Analytical gradients derived via the chain rule for weights and biases across all three layers.
3. **Gradient Checking**: Central difference numerical gradient verification ($\epsilon = 10^{-5}$) on randomly sampled weights, guaranteeing relative error $< 10^{-6}$.
4. **Mini-Batch SGD**: 50 epochs, batch size 32, with per-epoch shuffle across grid configurations:
   - Widths: $W \in [16, 64]$
   - Learning Rates: $\eta \in [0.001, 0.01, 0.1]$
5. **Profiling & Benchmarking**: Separate latency measurements (median of 100 runs) comparing forward vs. backward pass durations.

