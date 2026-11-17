"""Unstructured magnitude pruning, structured L2 filter pruning, and sparse serialization."""

from __future__ import annotations

import copy
import gzip
import io
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from lab07.src.models import (
        OversizedCNN,
        compute_macs_and_params,
        count_all_nonzero_parameters,
        count_parameters,
        count_prunable_weights,
        evaluate_model,
    )
except ImportError:
    from models import (
        OversizedCNN,
        compute_macs_and_params,
        count_all_nonzero_parameters,
        count_parameters,
        count_prunable_weights,
        evaluate_model,
    )


def get_prunable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return all Conv2d and Linear layers whose weight tensors are pruned."""
    layers: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


def build_magnitude_masks(
    model: nn.Module, sparsity: float, mode: str
) -> dict[str, torch.Tensor]:
    """Rank weights by |w| and build binary {0,1} masks globally or per layer."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"Sparsity must be in [0, 1), got {sparsity}")
    if mode not in ("global", "layerwise"):
        raise ValueError(f"Unknown pruning mode: {mode!r}")

    layers = get_prunable_layers(model)
    masks: dict[str, torch.Tensor] = {}

    if sparsity == 0.0:
        for name, mod in layers:
            masks[name] = torch.ones_like(mod.weight.data)
        return masks

    if mode == "layerwise":
        for name, mod in layers:
            w = mod.weight.data
            n = w.numel()
            k_prune = int(round(sparsity * n))
            flat_mask = torch.ones(n, dtype=w.dtype, device=w.device)
            if k_prune > 0:
                order = torch.argsort(w.abs().view(-1), descending=False)
                flat_mask[order[:k_prune]] = 0.0
            masks[name] = flat_mask.view_as(w)
        return masks

    # Global ranking by absolute value across all Conv2d and Linear weight tensors
    flat_abs = torch.cat([mod.weight.data.abs().view(-1) for _, mod in layers])
    total_n = flat_abs.numel()
    k_prune = int(round(sparsity * total_n))
    global_mask = torch.ones(total_n, dtype=flat_abs.dtype, device=flat_abs.device)
    if k_prune > 0:
        order = torch.argsort(flat_abs, descending=False)
        global_mask[order[:k_prune]] = 0.0

    offset = 0
    for name, mod in layers:
        n = mod.weight.data.numel()
        masks[name] = global_mask[offset : offset + n].view_as(mod.weight.data).clone()
        offset += n
    return masks


def apply_masks_(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """Apply binary masks in-place to pruned layers."""
    layer_dict = dict(get_prunable_layers(model))
    with torch.no_grad():
        for name, mask in masks.items():
            layer_dict[name].weight.data.mul_(mask)


def layer_sparsity_breakdown(model: nn.Module) -> dict[str, float]:
    """Compute zero fraction (%) per Conv2d/Linear weight tensor."""
    breakdown: dict[str, float] = {}
    for name, mod in get_prunable_layers(model):
        w = mod.weight.data
        zeros = int((w == 0).sum().item())
        breakdown[name] = 100.0 * zeros / w.numel()
    return breakdown


def finetune_with_fixed_mask(
    model: nn.Module,
    masks: dict[str, torch.Tensor] | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> list[dict[str, float]]:
    """Fine-tune a pruned model for `epochs` while keeping `masks` fixed (if provided)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    layer_dict = dict(get_prunable_layers(model))

    if masks is not None:
        apply_masks_(model, masks)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if masks is not None:
                with torch.no_grad():
                    for name, mask in masks.items():
                        grad = layer_dict[name].weight.grad
                        if grad is not None:
                            grad.mul_(mask)

            optimizer.step()

            if masks is not None:
                apply_masks_(model, masks)

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += bs

        val_metrics = evaluate_model(model, val_loader, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total,
                "train_acc": 100.0 * correct / total,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
        )

    if masks is not None:
        for name, mask in masks.items():
            w = layer_dict[name].weight.data
            if not torch.all(w[mask == 0] == 0):
                raise RuntimeError(f"Mask invariant violated in layer {name}")

    return history


def structured_prune_filters(
    model: OversizedCNN, filter_prune_ratio: float
) -> OversizedCNN:
    """Remove `filter_prune_ratio` of Conv2d filters ranked by L2 norm and rebuild dense CNN."""
    if not 0.0 <= filter_prune_ratio < 1.0:
        raise ValueError(f"filter_prune_ratio must be in [0, 1), got {filter_prune_ratio}")

    conv1: nn.Conv2d = model.features[0]  # type: ignore[assignment]
    conv2: nn.Conv2d = model.features[3]  # type: ignore[assignment]
    conv3: nn.Conv2d = model.features[6]  # type: ignore[assignment]
    fc1: nn.Linear = model.classifier[1]  # type: ignore[assignment]
    fc2: nn.Linear = model.classifier[3]  # type: ignore[assignment]

    def keep_top_filters(conv: nn.Conv2d, ratio: float) -> torch.Tensor:
        out_c = conv.out_channels
        k_keep = max(1, int(round(out_c * (1.0 - ratio))))
        norms = conv.weight.detach().view(out_c, -1).norm(p=2, dim=1)
        keep_idx = torch.topk(norms, k_keep, largest=True).indices
        return keep_idx.sort().values

    keep1 = keep_top_filters(conv1, filter_prune_ratio)
    keep2 = keep_top_filters(conv2, filter_prune_ratio)
    keep3 = keep_top_filters(conv3, filter_prune_ratio)

    new_channels = (int(keep1.numel()), int(keep2.numel()), int(keep3.numel()))
    pruned = OversizedCNN(
        channels=new_channels,
        fc_hidden=model.fc_hidden,
        num_classes=model.num_classes,
    )

    with torch.no_grad():
        p_conv1: nn.Conv2d = pruned.features[0]  # type: ignore[assignment]
        p_conv2: nn.Conv2d = pruned.features[3]  # type: ignore[assignment]
        p_conv3: nn.Conv2d = pruned.features[6]  # type: ignore[assignment]
        p_fc1: nn.Linear = pruned.classifier[1]  # type: ignore[assignment]
        p_fc2: nn.Linear = pruned.classifier[3]  # type: ignore[assignment]

        p_conv1.weight.copy_(conv1.weight[keep1])
        if conv1.bias is not None and p_conv1.bias is not None:
            p_conv1.bias.copy_(conv1.bias[keep1])

        p_conv2.weight.copy_(conv2.weight[keep2][:, keep1])
        if conv2.bias is not None and p_conv2.bias is not None:
            p_conv2.bias.copy_(conv2.bias[keep2])

        p_conv3.weight.copy_(conv3.weight[keep3][:, keep2])
        if conv3.bias is not None and p_conv3.bias is not None:
            p_conv3.bias.copy_(conv3.bias[keep3])

        # Feature map before Flatten is (B, C3, 3, 3) -> 9 elements per channel
        spatial_per_ch = 9
        flat_keep = torch.cat(
            [torch.arange(int(c) * spatial_per_ch, (int(c) + 1) * spatial_per_ch) for c in keep3]
        )
        p_fc1.weight.copy_(fc1.weight[:, flat_keep])
        if fc1.bias is not None and p_fc1.bias is not None:
            p_fc1.bias.copy_(fc1.bias)

        p_fc2.weight.copy_(fc2.weight)
        if fc2.bias is not None and p_fc2.bias is not None:
            p_fc2.bias.copy_(fc2.bias)

    return pruned


def measure_storage_formats(
    model: nn.Module, out_dir: Path, tag: str
) -> dict[str, Any]:
    """Compare on-disk size (bytes & KB) across dense and sparse representations."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dense PyTorch state_dict (.pt)
    dense_pt_path = out_dir / f"{tag}_dense.pt"
    torch.save(model.state_dict(), dense_pt_path)
    dense_pt_bytes = dense_pt_path.stat().st_size

    # 2. Gzip-compressed dense state_dict (.pt.gz)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    dense_gz_bytes = len(gzip.compress(buf.getvalue(), compresslevel=9))

    # 3. PyTorch native sparse_csr_tensor state dict (.pt) for weights + dense biases
    csr_torch_dict: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".weight") and tensor.ndim >= 2:
            mat = tensor.detach().cpu().reshape(tensor.shape[0], -1)
            csr_torch_dict[name] = {
                "shape": list(tensor.shape),
                "csr": mat.to_sparse_csr(),
            }
        else:
            csr_torch_dict[name] = tensor.detach().cpu()

    sparse_pt_path = out_dir / f"{tag}_csr.pt"
    torch.save(csr_torch_dict, sparse_pt_path)
    sparse_pt_bytes = sparse_pt_path.stat().st_size

    # 4. Raw SciPy CSR .npz (uncompressed) and .npz (compressed)
    npz_payload: dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        if name.endswith(".weight") and arr.ndim >= 2:
            mat = sp.csr_matrix(arr.reshape(arr.shape[0], -1).astype(np.float32))
            npz_payload[f"{name}__data"] = mat.data.astype(np.float32)
            npz_payload[f"{name}__indices"] = mat.indices.astype(np.int32)
            npz_payload[f"{name}__indptr"] = mat.indptr.astype(np.int32)
            npz_payload[f"{name}__shape"] = np.array(arr.shape, dtype=np.int32)
        else:
            npz_payload[name] = arr.astype(np.float32)

    csr_npz_path = out_dir / f"{tag}_csr.npz"
    np.savez(csr_npz_path, **npz_payload)
    csr_npz_bytes = csr_npz_path.stat().st_size

    csr_npz_comp_path = out_dir / f"{tag}_csr_compressed.npz"
    np.savez_compressed(csr_npz_comp_path, **npz_payload)
    csr_npz_comp_bytes = csr_npz_comp_path.stat().st_size

    nz_weights, tot_weights = count_prunable_weights(model)
    nz_all, tot_all = count_all_nonzero_parameters(model)
    actual_weight_sparsity = 1.0 - (nz_weights / tot_weights)

    return {
        "tag": tag,
        "total_params": tot_all,
        "nonzero_params": nz_all,
        "weight_sparsity_pct": 100.0 * actual_weight_sparsity,
        "dense_pt_bytes": dense_pt_bytes,
        "dense_gz_bytes": dense_gz_bytes,
        "sparse_csr_pt_bytes": sparse_pt_bytes,
        "sparse_csr_npz_bytes": csr_npz_bytes,
        "sparse_csr_npz_compressed_bytes": csr_npz_comp_bytes,
        "dense_pt_mb": dense_pt_bytes / (1024 * 1024),
        "sparse_csr_pt_mb": sparse_pt_bytes / (1024 * 1024),
        "sparse_csr_npz_mb": csr_npz_bytes / (1024 * 1024),
        "sparse_csr_npz_compressed_mb": csr_npz_comp_bytes / (1024 * 1024),
    }


def summarize_model_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    baseline_params: int | None = None,
) -> dict[str, Any]:
    """Evaluate accuracy, nonzero weights, parameter count, and theoretical/dense MACs."""
    val_res = evaluate_model(model, val_loader)
    test_res = evaluate_model(model, test_loader)
    total_params = count_parameters(model)
    nz_weights, tot_weights = count_prunable_weights(model)
    weight_sparsity = 1.0 - (nz_weights / tot_weights)
    dense_macs, _ = compute_macs_and_params(model)

    # For unstructured pruning on dense kernels, executed MACs remain dense_macs,
    # while theoretical nonzero MACs scale approximately with (1 - weight_sparsity).
    effective_param_sparsity = (
        1.0 - (total_params / baseline_params) if baseline_params else weight_sparsity
    )
    return {
        "val_loss": val_res["loss"],
        "val_acc": val_res["acc"],
        "test_loss": test_res["loss"],
        "test_acc": test_res["acc"],
        "total_params": total_params,
        "nonzero_weights": nz_weights,
        "total_prunable_weights": tot_weights,
        "weight_sparsity_pct": 100.0 * weight_sparsity,
        "structural_param_sparsity_pct": 100.0 * effective_param_sparsity,
        "dense_macs": dense_macs,
        "theoretical_nonzero_macs": dense_macs * (1.0 - weight_sparsity),
        "layer_sparsity_pct": layer_sparsity_breakdown(model),
    }
