import sys
import importlib

packages = [
    "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory_profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm"
]

print(f"Python: {sys.version.split()[0]}")
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
