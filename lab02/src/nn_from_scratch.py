import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# random seed
np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load and prepare dataset
X_raw, y_raw = load_digits(return_X_y=True)
X_scaled = X_raw / 16.0  # values from [0, 16] to [0, 1]

# Convert label integers
num_classes = 10
Y_one_hot = np.eye(num_classes)[y_raw]

# 70% train and 30% test split
X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
    X_scaled, y_raw, Y_one_hot, test_size=0.30, stratify=y_raw, random_state=42
)


# 2. Build network
class TwoLayerMLP:
    def __init__(self, input_dim=64, hidden_dim=16, output_dim=10, seed=42):
        rng = np.random.RandomState(seed)
        # Initialize weights
        self.W1 = rng.normal(0.0, 0.01, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, 0.01, size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.normal(0.0, 0.01, size=(hidden_dim, output_dim))
        self.b3 = np.zeros(output_dim)

    def forward(self, X):
        # linear step + ReLU
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)


        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)

        # output linear step
        self.Z3 = np.dot(self.A2, self.W3) + self.b3

        # Softmax: subtract max value to prevent overflow
        shift_Z = self.Z3 - np.max(self.Z3, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        self.probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, Y):
        N = X.shape[0]
        # Output layer gradient
        dZ3 = (self.probs - Y) / N
        self.dW3 = np.dot(self.A2.T, dZ3)
        self.db3 = np.sum(dZ3, axis=0)

        # Backprop through second hidden layer
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (self.Z2 > 0)
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0)

        # Backprop through first hidden layer (derivative of ReLU)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        self.dW1 = np.dot(X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0)

    def update(self, lr):
        # Gradient step
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3
def cross_entropy_loss(probs, Y):
    # Clip probabilities slightly to avoid log(0) error
    eps = 1e-15
    clipped_probs = np.clip(probs, eps, 1 - eps)
    return -np.sum(Y * np.log(clipped_probs)) / Y.shape[0]

# Basic sanity checks
print("=== SANITY CHECKS ===")
test_net = TwoLayerMLP(input_dim=64, hidden_dim=16, output_dim=10, seed=42)
init_probs = test_net.forward(X_train)
init_loss = cross_entropy_loss(init_probs, Y_train)

print(f"Output shape: {init_probs.shape} (Expected: ({X_train.shape[0]}, 10))")
row_sum_diff = np.max(np.abs(np.sum(init_probs, axis=1) - 1.0))
print(f"Row sum max diff from 1.0: {row_sum_diff:.2e} (Pass: {row_sum_diff < 1e-6})")
print(f"Initial loss: {init_loss:.4f} (Expected ~ ln(10) = 2.3026)")

# Numerical gradient checking
print("\n=== NUMERICAL GRADIENT CHECKING ===")
# Use small batch of 32 samples for check
sub_X = X_train[:32]
sub_Y = Y_train[:32]
test_net.forward(sub_X)
test_net.backward(sub_X, sub_Y)

weights_list = [
    ('W1', test_net.W1, test_net.dW1),
    ('W2', test_net.W2, test_net.dW2),
    ('W3', test_net.W3, test_net.dW3),
]

rng = np.random.RandomState(42)
eps = 1e-5
passed_checks = 0

print(f"{'Param':<6} {'Index':<12} {'Analytic':<14} {'Numerical':<14} {'Rel Error':<14} {'Status'}")
# Test 10 random weight coordinates
for i in range(10):
    w_idx = rng.randint(0, len(weights_list))
    param_name, W, dW = weights_list[w_idx]
    coord = tuple(rng.randint(0, dim) for dim in W.shape)

    analytic_grad = dW[coord]

    # Calculate f(x + eps)
    orig_val = W[coord]
    W[coord] = orig_val + eps
    loss_plus = cross_entropy_loss(test_net.forward(sub_X), sub_Y)

    # Calculate f(x - eps)
    W[coord] = orig_val - eps
    loss_minus = cross_entropy_loss(test_net.forward(sub_X), sub_Y)

    # Reset original weight value and restore state
    W[coord] = orig_val
    test_net.forward(sub_X)

    # Central difference formula
    numeric_grad = (loss_plus - loss_minus) / (2 * eps)
    rel_error = np.abs(analytic_grad - numeric_grad) / (np.maximum(1e-8, np.abs(analytic_grad) + np.abs(numeric_grad)))

    is_pass = rel_error < 1e-6
    if is_pass:
        passed_checks += 1
    print(f"{param_name:<6} {str(coord):<12} {analytic_grad:<14.6e} {numeric_grad:<14.6e} {rel_error:<14.6e} {'PASS' if is_pass else 'FAIL'}")

print(f"Passed gradient checks: {passed_checks}/10 (Requirement: >= 9/10)")

#  Training grid search
widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]
batch_size = 32
epochs = 50

plt.figure(figsize=(10, 6))
experiment_results = []

print("\n=== TRAINING CONFIGURATIONS ===")
for W in widths:
    for lr in learning_rates:
        net = TwoLayerMLP(input_dim=64, hidden_dim=W, output_dim=10, seed=42)
        train_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data before each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            # Mini-batch loop
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_X = X_shuffled[start:end]
                batch_Y = Y_shuffled[start:end]
                net.forward(batch_X)
                net.backward(batch_X, batch_Y)
                net.update(lr)

            # Record loss at the end of epoch
            epoch_probs = net.forward(X_train)
            epoch_loss = cross_entropy_loss(epoch_probs, Y_train)
            train_losses.append(epoch_loss)

        # Evaluate final test accuracy
        test_probs = net.forward(X_test)
        preds = np.argmax(test_probs, axis=1)
        test_acc = np.mean(preds == y_test)

        label = f"W={W}, lr={lr}"
        plt.plot(train_losses, label=f"{label} (Acc: {test_acc:.4f})")
        experiment_results.append({
            "Width": W,
            "LR": lr,
            "Final Loss": f"{train_losses[-1]:.4f}",
            "Test Accuracy": f"{test_acc:.4f}"
        })
        print(f"W={W:<2} | lr={lr:<5} | Final Loss={train_losses[-1]:.4f} | Test Acc={test_acc:.4f}")

# Save loss plot
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curves (50 Epochs, Mini-batch 32)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plot_path = os.path.join(RESULTS_DIR, "loss_curves.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Loss curves saved to {plot_path}")

# --- 7. Timing forward and backward passes ---
print("\n=== TIMING BREAKDOWN (Median of 100 runs) ===")
bench_net = TwoLayerMLP(input_dim=64, hidden_dim=64, output_dim=10, seed=42)
bench_X = X_train[:32]
bench_Y = Y_train[:32]

# Warm-up pass once before benchmarking
bench_net.forward(bench_X)
bench_net.backward(bench_X, bench_Y)

fwd_times = []
for _ in range(100):
    t0 = time.perf_counter()
    bench_net.forward(bench_X)
    t1 = time.perf_counter()
    fwd_times.append(t1 - t0)

bwd_times = []
for _ in range(100):
    t0 = time.perf_counter()
    bench_net.backward(bench_X, bench_Y)
    t1 = time.perf_counter()
    bwd_times.append(t1 - t0)

median_fwd_us = np.median(fwd_times) * 1e6
median_bwd_us = np.median(bwd_times) * 1e6

print(f"Forward Pass Median Time : {median_fwd_us:.2f} microseconds")
print(f"Backward Pass Median Time: {median_bwd_us:.2f} microseconds")
print(f"Backward / Forward Ratio : {median_bwd_us / median_fwd_us:.2f}x")

