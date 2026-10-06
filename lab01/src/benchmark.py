import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

process = psutil.Process()
accuracy_records = []
metrics_summary = []

for name, clf in models.items():
    clf.fit(X_train, y_train)

    train_times = []
    peak_train_rss = 0
    for _ in range(5):
        start_mem = process.memory_info().rss
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t1 = time.perf_counter()
        end_mem = process.memory_info().rss
        train_times.append(t1 - t0)
        peak_train_rss = max(peak_train_rss, end_mem, start_mem)

    median_train_time_ms = np.median(train_times) * 1000

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_records.append({"model": name, "test_accuracy": f"{acc:.4f}"})

    single_sample = X_test[0:1]
    clf.predict(single_sample)

    inference_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        clf.predict(single_sample)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

    median_inference_lat_ms = np.median(inference_times) * 1000

    model_path = os.path.join(RESULTS_DIR, f"{name}.joblib")
    joblib.dump(clf, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    metrics_summary.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "Train Time (ms)": f"{median_train_time_ms:.2f}",
        "Inference Latency (ms)": f"{median_inference_lat_ms:.4f}",
        "Model Size (Bytes)": size_bytes,
        "Model Size (KB)": f"{size_kb:.2f}",
        "Peak RSS (MB)": f"{peak_train_rss / (1024 * 1024):.2f}"
    })

pd.DataFrame(accuracy_records).to_csv(
    os.path.join(RESULTS_DIR, "baseline_accuracy.csv"), index=False
)

summary_df = pd.DataFrame(metrics_summary)
print("\n=== SYSTEM BENCHMARK RESULTS ===")
print(summary_df.to_string(index=False))
