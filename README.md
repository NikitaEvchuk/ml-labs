# ML Systems Engineering Labs

System benchmarks for Machine Learning Systems.

## Structure
- `lab01/`: Environment and first system measurements (Logistic Regression vs Random Forest on Breast Cancer Wisconsin).
- `lab02/`: Neural network from scratch in NumPy with analytical backpropagation and gradient checking.
- `lab03/`: Model architectures (MLP, CNN, Transformer) and efficiency benchmarking on FashionMNIST.
- `lab04/`: Data pipeline and quality (UCI Adult, schema validation, leakage-free transformation, feature engineering, data card).
- `lab05/`: Frameworks and automatic differentiation (scalar autograd engine, topological backprop, PyTorch autograd verification, eager vs traced/compiled execution, tensor contiguity benchmark).
- `lab06/`: YAML-driven CNN training pipeline, MLflow tracking, PyTorch profiling, DataLoader worker/batch-size/LR/optimizer/gradient-accumulation/resume studies, and individual project proposal.
- `lab07/`: Model compression via unstructured magnitude pruning (global and layer-wise), structured L2 filter pruning, dense vs. sparse storage/latency benchmarking, and knowledge distillation on FashionMNIST.
- `requirements.txt`: Pinned dependencies for reproducible runs.
