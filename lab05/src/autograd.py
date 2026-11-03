
import os
import math
import time
import json
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# 1. Scalar autograd Value node
class Value:
    """Scalar value node supporting automatic differentiation."""
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be int or float"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1.0)

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        # Clip input to prevent log(0)
        safe_data = max(self.data, 1e-15)
        out = Value(math.log(safe_data), (self,), "log")

        def _backward():
            self.grad += (1.0 / safe_data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        # Numerically stable sigmoid
        if self.data >= 0:
            s = 1.0 / (1.0 + math.exp(-self.data))
        else:
            ez = math.exp(self.data)
            s = ez / (1.0 + ez)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        # Compatibility method for scalar expressions
        return self

    def backward(self):
        # Topologically sort computation graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Seed gradient and reverse pass accumulation
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# 2. Tensor wrapper for vectorized operations over Value nodes
class Tensor:
    """Vector wrapper around Value nodes for elementwise operations."""
    def __init__(self, data):
        if isinstance(data, Tensor):
            self.elements = list(data.elements)
            self.shape = data.shape
        elif isinstance(data, (list, tuple, np.ndarray)):
            flat = []
            for item in np.array(data).flat:
                flat.append(item if isinstance(item, Value) else Value(float(item)))
            self.elements = flat
            self.shape = np.array(data).shape
        else:
            self.elements = [data if isinstance(data, Value) else Value(float(data))]
            self.shape = (1,)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        return self.elements[idx]

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert len(self.elements) == len(other.elements), "Tensors must have matching size"
        return Tensor([a + b for a, b in zip(self.elements, other.elements)])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert len(self.elements) == len(other.elements), "Tensors must have matching size"
        return Tensor([a * b for a, b in zip(self.elements, other.elements)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        return Tensor([a ** power for a in self.elements])

    def relu(self):
        return Tensor([a.relu() for a in self.elements])

    def sum(self):
        res = Value(0.0)
        for elem in self.elements:
            res = res + elem
        return res

    @property
    def grad(self):
        return np.array([elem.grad for elem in self.elements]).reshape(self.shape)

    @property
    def data(self):
        return np.array([elem.data for elem in self.elements]).reshape(self.shape)


# 3. Neural network modules
class Neuron:
    """Neuron with linear weights and bias."""
    def __init__(self, nin, act="relu"):
        # Xavier uniform initialization
        scale = 1.0 / math.sqrt(nin)
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        act_val = self.b
        for wi, xi in zip(self.w, x):
            act_val = act_val + wi * xi
        if self.act == "relu":
            return act_val.relu()
        elif self.act == "sigmoid":
            return act_val.sigmoid()
        return act_val

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """Fully connected layer of neurons."""
    def __init__(self, nin, nout, act="relu"):
        self.neurons = [Neuron(nin, act=act) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class TinyMLP:
    """2 -> 8 -> 1 architecture with Sigmoid output."""
    def __init__(self, nin=2, nhid=8, nout=1):
        self.l1 = Layer(nin, nhid, act="relu")
        self.l2 = Layer(nhid, nout, act="sigmoid")

    def __call__(self, x):
        h = self.l1(x)
        out = self.l2(h)
        return out

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# 4. Binary Cross Entropy loss
def bce_loss(preds, targets):
    total_loss = Value(0.0)
    for p, y in zip(preds, targets):
        term1 = Value(y) * p.log()
        term2 = Value(1.0 - y) * (Value(1.0) - p).log()
        total_loss = total_loss + (-(term1 + term2))
    return total_loss * (1.0 / len(targets))


# 5. Task 4: Gradient verification against PyTorch & Numerical estimates
def verify_gradients():
    print("=== Task 4: Verifying Gradients against PyTorch & Numerical Estimates ===")
    
    # Input vectors
    a_vals = [1.5, -2.0, 0.75]
    b_vals = [-0.5, 3.0, 1.25]
    c_vals = [2.0, -1.0, -0.25]

    # Custom autograd forward and backward
    a_t = Tensor(a_vals)
    b_t = Tensor(b_vals)
    c_t = Tensor(c_vals)
    out_custom = ((a_t * b_t + c_t) ** 2).relu().sum()
    out_custom.backward()

    # PyTorch forward and backward
    a_pt = torch.tensor(a_vals, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_vals, dtype=torch.float64, requires_grad=True)
    c_pt = torch.tensor(c_vals, dtype=torch.float64, requires_grad=True)
    out_pt = ((a_pt * b_pt + c_pt) ** 2).relu().sum()
    out_pt.backward()

    # Numerical gradient check using central difference
    eps = 1e-7

    def eval_num(a_in, b_in, c_in):
        val = np.maximum(0.0, (np.array(a_in) * np.array(b_in) + np.array(c_in)) ** 2)
        return float(np.sum(val))

    num_grad_a = []
    for i in range(len(a_vals)):
        a_plus, a_minus = list(a_vals), list(a_vals)
        a_plus[i] += eps
        a_minus[i] -= eps
        num_grad_a.append((eval_num(a_plus, b_vals, c_vals) - eval_num(a_minus, b_vals, c_vals)) / (2 * eps))

    num_grad_b = []
    for i in range(len(b_vals)):
        b_plus, b_minus = list(b_vals), list(b_vals)
        b_plus[i] += eps
        b_minus[i] -= eps
        num_grad_b.append((eval_num(a_vals, b_plus, c_vals) - eval_num(a_vals, b_minus, c_vals)) / (2 * eps))

    num_grad_c = []
    for i in range(len(c_vals)):
        c_plus, c_minus = list(c_vals), list(c_vals)
        c_plus[i] += eps
        c_minus[i] -= eps
        num_grad_c.append((eval_num(a_vals, b_vals, c_plus) - eval_num(a_vals, b_vals, c_minus)) / (2 * eps))

    # Evaluate gradient differences
    diff_pt_a = np.max(np.abs(a_t.grad - a_pt.grad.numpy()))
    diff_pt_b = np.max(np.abs(b_t.grad - b_pt.grad.numpy()))
    diff_pt_c = np.max(np.abs(c_t.grad - c_pt.grad.numpy()))
    max_diff_pytorch = max(diff_pt_a, diff_pt_b, diff_pt_c)

    diff_num_a = np.max(np.abs(a_t.grad - np.array(num_grad_a)))
    diff_num_b = np.max(np.abs(b_t.grad - np.array(num_grad_b)))
    diff_num_c = np.max(np.abs(c_t.grad - np.array(num_grad_c)))
    max_diff_numerical = max(diff_num_a, diff_num_b, diff_num_c)

    print(f"Custom forward output:  {out_custom.data:.8f}")
    print(f"PyTorch forward output: {out_pt.item():.8f}")
    print(f"Max abs diff with PyTorch autograd:   {max_diff_pytorch:.2e} (< 1e-6: {max_diff_pytorch < 1e-6})")
    print(f"Max abs diff with Numerical estimate: {max_diff_numerical:.2e} (< 1e-6: {max_diff_numerical < 1e-6})")

    # Scalar Value node check
    a_sc = Value(2.5)
    b_sc = Value(-1.5)
    c_sc = Value(1.0)
    out_sc = ((a_sc * b_sc + c_sc) ** 2).relu().sum()
    out_sc.backward()

    a_sc_pt = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    b_sc_pt = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
    c_sc_pt = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    out_sc_pt = ((a_sc_pt * b_sc_pt + c_sc_pt) ** 2).relu().sum()
    out_sc_pt.backward()

    sc_diff_a = abs(a_sc.grad - a_sc_pt.grad.item())
    sc_diff_b = abs(b_sc.grad - b_sc_pt.grad.item())
    sc_diff_c = abs(c_sc.grad - c_sc_pt.grad.item())
    max_diff_scalar = max(sc_diff_a, sc_diff_b, sc_diff_c)

    print(f"Scalar verification max abs diff:     {max_diff_scalar:.2e} (< 1e-6: {max_diff_scalar < 1e-6})")

    results = {
        "forward_custom": out_custom.data,
        "forward_pytorch": out_pt.item(),
        "max_diff_pytorch": float(max_diff_pytorch),
        "max_diff_numerical": float(max_diff_numerical),
        "max_diff_scalar": float(max_diff_scalar),
        "vector_grads": {
            "a_custom": a_t.grad.tolist(),
            "a_pytorch": a_pt.grad.tolist(),
            "b_custom": b_t.grad.tolist(),
            "b_pytorch": b_pt.grad.tolist(),
            "c_custom": c_t.grad.tolist(),
            "c_pytorch": c_pt.grad.tolist()
        }
    }
    with open(os.path.join(RESULTS_DIR, "gradient_verification.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# 6. Task 5: Train tiny model on make_moons and compare speed with PyTorch
class TorchTinyMLP(nn.Module):
    def __init__(self, nin=2, nhid=8, nout=1):
        super().__init__()
        self.fc1 = nn.Linear(nin, nhid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nhid, nout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


def train_custom_model(X, y, epochs=50, lr=0.5):
    model = TinyMLP(nin=2, nhid=8, nout=1)
    X_val = [[Value(x0), Value(x1)] for x0, x1 in X]
    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        preds = [model(x) for x in X_val]
        loss = bce_loss(preds, y)
        loss_history.append(loss.data)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update weights (SGD)
        for p in model.parameters():
            p.data -= lr * p.grad

    final_preds = [1 if model(x).data >= 0.5 else 0 for x in X_val]
    acc = sum(p == t for p, t in zip(final_preds, y)) / len(y)
    return model, loss_history, acc


def train_pytorch_model(X, y, epochs=50, lr=0.5):
    model = TorchTinyMLP(nin=2, nhid=8, nout=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    with torch.no_grad():
        final_preds = (model(X_tensor) >= 0.5).float()
        acc = (final_preds == y_tensor).float().mean().item()
    return model, loss_history, acc


def benchmark_make_moons():
    print("\n=== Task 5: Training Tiny Model on make_moons (Median of 5 runs) ===")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

    num_runs = 5
    epochs = 50
    lr = 0.5

    # PyTorch warm-up run to eliminate cold-start module import overhead
    _ = train_pytorch_model(X, y, epochs=5, lr=lr)

    custom_times = []
    custom_losses = []
    custom_accs = []
    last_custom_curve = []
    for run in range(num_runs):
        start = time.perf_counter()
        _, curve, acc = train_custom_model(X, y, epochs=epochs, lr=lr)
        elapsed = time.perf_counter() - start
        custom_times.append(elapsed)
        custom_losses.append(curve[-1])
        custom_accs.append(acc)
        last_custom_curve = curve

    torch_times = []
    torch_losses = []
    torch_accs = []
    last_torch_curve = []
    for run in range(num_runs):
        start = time.perf_counter()
        _, curve, acc = train_pytorch_model(X, y, epochs=epochs, lr=lr)
        elapsed = time.perf_counter() - start
        torch_times.append(elapsed)
        torch_losses.append(curve[-1])
        torch_accs.append(acc)
        last_torch_curve = curve

    custom_median = float(np.median(custom_times))
    torch_median = float(np.median(torch_times))
    speedup = custom_median / torch_median

    print(f"Custom autograd training times (s): {[round(t, 4) for t in custom_times]}")
    print(f"Custom median time:  {custom_median:.4f} s | Final Loss: {custom_losses[-1]:.4f} | Acc: {custom_accs[-1]*100:.1f}%")
    print(f"PyTorch training times (s):        {[round(t, 4) for t in torch_times]}")
    print(f"PyTorch median time: {torch_median:.4f} s | Final Loss: {torch_losses[-1]:.4f} | Acc: {torch_accs[-1]*100:.1f}%")
    print(f"PyTorch Speedup:     {speedup:.1f}x faster")

    results = {
        "epochs": epochs,
        "n_samples": len(y),
        "custom_times_s": custom_times,
        "custom_median_s": custom_median,
        "custom_final_loss": custom_losses[-1],
        "custom_final_acc": custom_accs[-1],
        "torch_times_s": torch_times,
        "torch_median_s": torch_median,
        "torch_final_loss": torch_losses[-1],
        "torch_final_acc": torch_accs[-1],
        "speedup_ratio": speedup,
        "custom_curve": last_custom_curve,
        "torch_curve": last_torch_curve
    }
    with open(os.path.join(RESULTS_DIR, "training_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# 7. Task 6: Graph inspection, Eager vs Compiled/Traced, and Contiguity benchmark
def inspect_and_benchmark_pytorch():
    print("\n=== Task 6: PyTorch Graph, Execution Modes, and Contiguity ===")

    model = TorchTinyMLP(nin=2, nhid=8, nout=1)
    model.eval()
    sample_input = torch.randn(100, 2)

    # 7.1 Inspect computation graph
    traced_model = torch.jit.trace(model, sample_input)
    graph_str = str(traced_model.graph)
    print("--- Traced Graph ---")
    print(graph_str.strip())

    scripted_model = torch.jit.script(model)

    # 7.2 Compare eager vs compiled / traced execution
    warmup_iters = 50
    test_iters = 500

    # Eager mode
    for _ in range(warmup_iters):
        _ = model(sample_input)
    eager_latencies = []
    for _ in range(test_iters):
        t0 = time.perf_counter()
        _ = model(sample_input)
        eager_latencies.append((time.perf_counter() - t0) * 1000.0)
    eager_median_ms = float(np.median(eager_latencies))

    # Traced mode
    for _ in range(warmup_iters):
        _ = traced_model(sample_input)
    traced_latencies = []
    for _ in range(test_iters):
        t0 = time.perf_counter()
        _ = traced_model(sample_input)
        traced_latencies.append((time.perf_counter() - t0) * 1000.0)
    traced_median_ms = float(np.median(traced_latencies))

    # Scripted mode
    for _ in range(warmup_iters):
        _ = scripted_model(sample_input)
    scripted_latencies = []
    for _ in range(test_iters):
        t0 = time.perf_counter()
        _ = scripted_model(sample_input)
        scripted_latencies.append((time.perf_counter() - t0) * 1000.0)
    scripted_median_ms = float(np.median(scripted_latencies))

    # Compile mode
    compile_supported = False
    compiled_median_ms = None
    try:
        compiled_model = torch.compile(model)
        for _ in range(warmup_iters):
            _ = compiled_model(sample_input)
        compiled_latencies = []
        for _ in range(test_iters):
            t0 = time.perf_counter()
            _ = compiled_model(sample_input)
            compiled_latencies.append((time.perf_counter() - t0) * 1000.0)
        compiled_median_ms = float(np.median(compiled_latencies))
        compile_supported = True
    except Exception as e:
        print(f"torch.compile not supported or failed: {e}")

    print(f"Eager median latency:    {eager_median_ms:.4f} ms")
    print(f"Traced median latency:   {traced_median_ms:.4f} ms")
    print(f"Scripted median latency: {scripted_median_ms:.4f} ms")
    if compile_supported:
        print(f"Compiled median latency: {compiled_median_ms:.4f} ms")

    # 7.3 Inspect tensor shape, dtype, contiguity
    print("\n--- Tensor Representation & Contiguity Inspection ---")
    tensor_c = torch.randn(2048, 2048, dtype=torch.float32)
    print(f"Contiguous tensor shape: {tensor_c.shape}, dtype: {tensor_c.dtype}, stride: {tensor_c.stride()}, is_contiguous: {tensor_c.is_contiguous()}")

    tensor_nc = tensor_c.t()
    print(f"Transposed view shape:   {tensor_nc.shape}, dtype: {tensor_nc.dtype}, stride: {tensor_nc.stride()}, is_contiguous: {tensor_nc.is_contiguous()}")

    # 7.4 Measure operation speed impact of non-contiguous view
    op_warmup = 20
    op_iters = 100

    for _ in range(op_warmup):
        _ = torch.sum(tensor_c, dim=1)
        _ = torch.sum(tensor_nc, dim=1)

    c_times = []
    for _ in range(op_iters):
        t0 = time.perf_counter()
        _ = torch.sum(tensor_c, dim=1)
        c_times.append((time.perf_counter() - t0) * 1000.0)
    c_median_ms = float(np.median(c_times))

    nc_times = []
    for _ in range(op_iters):
        t0 = time.perf_counter()
        _ = torch.sum(tensor_nc, dim=1)
        nc_times.append((time.perf_counter() - t0) * 1000.0)
    nc_median_ms = float(np.median(nc_times))

    tensor_made_c = tensor_nc.contiguous()
    made_c_times = []
    for _ in range(op_iters):
        t0 = time.perf_counter()
        _ = torch.sum(tensor_made_c, dim=1)
        made_c_times.append((time.perf_counter() - t0) * 1000.0)
    made_c_median_ms = float(np.median(made_c_times))

    slowdown = nc_median_ms / c_median_ms
    print(f"Contiguous sum median:     {c_median_ms:.4f} ms")
    print(f"Non-contiguous sum median: {nc_median_ms:.4f} ms ({slowdown:.2f}x slower due to non-sequential memory access)")
    print(f"After .contiguous() sum:   {made_c_median_ms:.4f} ms")

    results = {
        "graph_inspection": graph_str,
        "eager_median_ms": eager_median_ms,
        "traced_median_ms": traced_median_ms,
        "scripted_median_ms": scripted_median_ms,
        "compiled_median_ms": compiled_median_ms,
        "compile_supported": compile_supported,
        "contiguity": {
            "contiguous_stride": list(tensor_c.stride()),
            "contiguous_is_contiguous": tensor_c.is_contiguous(),
            "non_contiguous_stride": list(tensor_nc.stride()),
            "non_contiguous_is_contiguous": tensor_nc.is_contiguous(),
            "contiguous_op_ms": c_median_ms,
            "non_contiguous_op_ms": nc_median_ms,
            "made_contiguous_op_ms": made_c_median_ms,
            "slowdown_factor": slowdown
        }
    }

    with open(os.path.join(RESULTS_DIR, "pytorch_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# 8. Generate summary visualization plots
def generate_plots(train_res, pt_res):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Training loss curves
    axes[0, 0].plot(train_res["custom_curve"], label="Custom Autograd", color="#1f77b4", lw=2)
    axes[0, 0].plot(train_res["torch_curve"], label="PyTorch (Eager)", color="#ff7f0e", lw=2, linestyle="--")
    axes[0, 0].set_title("Training Loss on make_moons (50 Epochs)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Binary Cross Entropy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Training time comparison
    frameworks = ["Custom Autograd", "PyTorch"]
    medians = [train_res["custom_median_s"], train_res["torch_median_s"]]
    bars = axes[0, 1].bar(frameworks, medians, color=["#1f77b4", "#ff7f0e"], width=0.5)
    axes[0, 1].set_title(f"Median Training Time (PyTorch {train_res['speedup_ratio']:.1f}x faster)")
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h + 0.05 * max(medians), f"{h:.4f}s", ha="center")

    # Panel 3: PyTorch Execution Modes Latency
    modes = ["Eager", "Traced", "Scripted"]
    latencies = [pt_res["eager_median_ms"], pt_res["traced_median_ms"], pt_res["scripted_median_ms"]]
    bars3 = axes[1, 0].bar(modes, latencies, color=["#2ca02c", "#d62728", "#9467bd"], width=0.5)
    axes[1, 0].set_title("PyTorch Execution Modes Latency (Inference)")
    axes[1, 0].set_ylabel("Median Latency (ms)")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    for bar in bars3:
        h = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.4f}ms", ha="center")

    # Panel 4: Contiguous vs Non-contiguous memory access
    layouts = ["Contiguous", "Transposed (Non-contig)", "Post .contiguous()"]
    contig_data = pt_res["contiguity"]
    times = [contig_data["contiguous_op_ms"], contig_data["non_contiguous_op_ms"], contig_data["made_contiguous_op_ms"]]
    bars4 = axes[1, 1].bar(layouts, times, color=["#17becf", "#e377c2", "#bcbd22"], width=0.5)
    axes[1, 1].set_title(f"Memory Contiguity Impact on Sum Reduction ({contig_data['slowdown_factor']:.2f}x Slowdown)")
    axes[1, 1].set_ylabel("Median Time (ms)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    for bar in bars4:
        h = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.3f}ms", ha="center")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "benchmark_plots.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved summary figure to {plot_path}")


def main():
    verify_gradients()
    train_res = benchmark_make_moons()
    pt_res = inspect_and_benchmark_pytorch()
    generate_plots(train_res, pt_res)


if __name__ == "__main__":
    main()
