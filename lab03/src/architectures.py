import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

try:
    from thop import profile
except ImportError:
    profile = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Dataset loading and split (55k train, 5k val, 10k test)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_train_dataset, [55000, 5000], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. Three model architectures
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class PatchAttentionNet(nn.Module):
    def __init__(self, patch_size=7, d_model=64, nhead=4, dim_feedforward=128, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 1 * patch_size * patch_size
        num_patches = (28 // patch_size) ** 2

        self.projection = nn.Linear(patch_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Cut 28x28 image into 16 patches of 7x7
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, -1, C * p * p)

        tokens = self.projection(patches) + self.pos_embedding
        out = self.transformer_encoder(tokens)
        return self.fc(out.mean(dim=1))

models = {
    "MLP": MLP().to(device),
    "CNN": CNN().to(device),
    "Attention": PatchAttentionNet().to(device)
}

# 3. Model profiling: parameters, FLOPs, and latency
print("\n" + "="*50 + "\nPROFILING & HARDWARE COSTS\n" + "="*50)
profile_data = {}

for name, model in models.items():
    model.eval()
    dummy_single = torch.randn(1, 1, 28, 28).to(device)
    dummy_batch = torch.randn(128, 1, 28, 28).to(device)

    # Count parameters
    stats = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    total_params = stats.total_params

    # Measure FLOPs
    thop_flops = 0
    if profile is not None:
        try:
            flops_val, _ = profile(model, inputs=(dummy_single,), verbose=False)
            thop_flops = flops_val
        except Exception:
            thop_flops = -1

    # Time batch size 1 (100 runs)
    for _ in range(10):
        _ = model(dummy_single)
    latencies_b1 = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy_single)
        t1 = time.perf_counter()
        latencies_b1.append((t1 - t0) * 1000)
    median_lat_b1 = np.median(latencies_b1)

    # Time batch size 128 (20 runs)
    for _ in range(5):
        _ = model(dummy_batch)
    latencies_b128 = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model(dummy_batch)
        t1 = time.perf_counter()
        latencies_b128.append((t1 - t0) * 1000)
    median_lat_b128 = np.median(latencies_b128)

    profile_data[name] = {
        "params": total_params,
        "thop_flops": thop_flops,
        "lat_b1": median_lat_b1,
        "lat_b128": median_lat_b128
    }

    print(f"\n[{name}]")
    print(f"Total Parameters     : {total_params:,}")
    print(f"thop Measured FLOPs  : {int(thop_flops):,}")
    print(f"Latency (Batch 1)    : {median_lat_b1:.3f} ms")
    print(f"Latency (Batch 128)  : {median_lat_b128:.3f} ms")

# 4. Training (5 epochs with Adam)
print("\n" + "="*50 + "\nTRAINING MODELS (5 EPOCHS)\n" + "="*50)
criterion = nn.CrossEntropyLoss()

for name, model in models.items():
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"\nTraining {name}...")

    for epoch in range(1, 6):
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_b.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

        print(f"Epoch {epoch}/5 - Loss: {total_loss / total:.4f} | Train Acc: {(correct / total) * 100:.2f}%")

    # Final test evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = model(x_b)
            test_correct += (out.argmax(dim=1) == y_b).sum().item()
            test_total += y_b.size(0)

    final_test_acc = test_correct / test_total
    profile_data[name]["test_acc"] = final_test_acc
    print(f"--> {name} Final Test Accuracy: {final_test_acc * 100:.2f}%")

# 5. Save comparison plots
model_names = list(models.keys())
accuracies = [profile_data[m]["test_acc"] * 100 for m in model_names]
params = [profile_data[m]["params"] for m in model_names]
latencies = [profile_data[m]["lat_b1"] for m in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy vs Parameters
ax1.scatter(params, accuracies, color=["crimson", "dodgerblue", "darkorange"], s=100)
for i, txt in enumerate(model_names):
    ax1.annotate(txt, (params[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center')
ax1.set_xlabel("Number of Parameters")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_title("Accuracy vs. Parameter Count")
ax1.grid(True, linestyle="--", alpha=0.5)

# Accuracy vs Latency
ax2.scatter(latencies, accuracies, color=["crimson", "dodgerblue", "darkorange"], s=100)
for i, txt in enumerate(model_names):
    ax2.annotate(txt, (latencies[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center')

plot_path = os.path.join(RESULTS_DIR, "tradeoffs.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"\nTrade-off plots successfully saved to: {plot_path}")