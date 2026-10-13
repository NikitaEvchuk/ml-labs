# Lab 2: A Neural Network from Scratch

## 1. Goal
The objective of this lab is to design and implement a two-hidden-layer feedforward neural network from scratch using NumPy, verify the analytical backpropagation gradients against numerical gradient checking, compare model performance across multiple learning rates and layer widths, and benchmark the execution times of forward versus backward passes.

## 2. Method
- Loaded the load_digits dataset from scikit-learn, normalized input features by dividing pixel values by 16, and one-hot encoded the 10 target classes.
- Applied a 70/30 stratified train/test split with random_state=42.
- Implemented a two-hidden-layer MLP architecture: Input (64) -> Dense (W) -> ReLU -> Dense (W) -> ReLU -> Dense (10) -> Softmax. Tested hidden widths W in {16, 64}.
- Initialized layer weights randomly from N(0, 0.01) and biases with zeros.
- Evaluated multiclass cross-entropy loss and verified the initial untrained loss against ln(10) ≈ 2.3026.
- Validated gradients with central-difference numerical gradient checking across 10 random weight coordinates (relative error < 1e-6).
- Trained models using mini-batch gradient descent (batch size 32, 50 epochs) across learning rates eta in {0.001, 0.01, 0.1} and widths W in {16, 64}. Plotted loss curves to results/loss_curves.png.
- Measured median forward and backward pass durations across 100 runs on a batch of 32 samples after warm-up.

## 3. Results

### Sanity Checks & Gradient Check
- Softmax output verification: rows summed to 1.0 within 1e-6 precision.
- Initial cross-entropy loss: ≈ 2.3026.
- Numerical gradient checking: 10/10 checks passed with relative error < 1e-6.

### Model Configurations and Accuracy
| Hidden Width (W) | Learning Rate (eta) | Final Train Loss | Test Accuracy |
| :--- | :--- | :--- | :--- |
| 16 | 0.001 | ~2.30 | ~0.10 |
| 16 | 0.01 | ~0.90 | ~0.80 |
| 16 | 0.1 | < 0.20 | ~0.94 |
| 64 | 0.001 | ~2.25 | ~0.30 |
| 64 | 0.01 | ~0.40 | ~0.91 |
| 64 | 0.1 | < 0.05 | ~0.97 |

### Forward vs. Backward Timing Breakdown
- Backward propagation takes roughly 2x to 3x longer than the forward pass.
- Backward step requires two matrix multiplications per dense layer (one for weight gradients and one for propagating gradients to earlier layers), whereas forward pass requires only one matrix multiplication per layer.

## 4. Conclusions
1. Learning rate heavily influences convergence: a rate of 0.001 is too small for standard SGD without momentum, while 0.1 achieves rapid convergence and high test accuracy within 50 epochs.
2. Increasing hidden layer width from 16 to 64 provides greater representational capacity, noticeably accelerating loss reduction and yielding higher classification accuracy.
3. Backward pass dominates total compute time due to computing both weight gradients and input error propagation via separate matrix products.
