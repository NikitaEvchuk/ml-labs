# Lab 1: Environment and First System Measurements

## 1. Goal
The objective of this lab is to set up a clean, reproducible Python environment for ML engineering, train two baseline classifiers on the Breast Cancer Wisconsin dataset, and measure their system costs (training time, inference latency, memory footprint, and disk size). Finally, I evaluate whether these models fit into Cloud, Edge, Mobile, and TinyML hardware budgets.

## 2. Method
- I created an isolated virtual environment using Python 3.11.8 and installed the exact pinned dependencies from requirements.txt.
- I loaded the Breast Cancer Wisconsin diagnostic dataset from scikit-learn and split it into 70% train and 30% test sets with stratification and random_state=42.
- I trained two baseline models:
  - Logistic Regression (max_iter=1000, random_state=42)
  - Random Forest Classifier (n_estimators=100, random_state=42)
- To measure training time, I used a warm-up run followed by 5 timed runs, taking the median.
- For single-sample inference latency, I ran 1 warm-up call followed by 100 sequential predictions and recorded the median latency.
- Model sizes were measured after dumping them via joblib. Peak RAM (RSS) was recorded using psutil during execution.

## 3. Results

### Baseline Accuracy
| Model | Test Accuracy |
| :--- | :--- |
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |

### System Cost Measurements
| Model | Train Time (ms) | Inference Latency (ms) | Model Size (Bytes) | Model Size (KB) | Peak RSS (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LogisticRegression | 388.37 | 0.0509 | 1055 | 1.03 | 142.57 |
| RandomForest | 172.69 | 2.5639 | 290889 | 284.07 | 143.28 |

### Deployment Budget Assessment
Given budgets:
- **Cloud**: RAM >= 1 GB, Latency <= 100 ms, Size <= 500 MB
- **Edge**: RAM 256–1024 MB, Latency <= 50 ms, Size <= 50 MB
- **Mobile**: RAM 64–256 MB, Latency <= 20 ms, Size <= 10 MB
- **TinyML**: RAM <= 256 KB, Latency <= 10 ms, Size <= 100 KB

**Suitability:**
- **Logistic Regression**: Runs easily on **Cloud**, **Edge**, and **Mobile**. It is very fast (~0.05 ms) and tiny (~1 KB). However, for **TinyML**, standard Python and scikit-learn require tens of megabytes of RAM just to boot, which breaks the 256 KB limit. It could only run on TinyML if the weights are extracted and evaluated as a bare-metal C array.
- **Random Forest**: Runs without issues on **Cloud**, **Edge**, and **Mobile** (its ~2.56 ms latency is well under the 20 ms limit, and size is under 1 MB). It strictly **fails TinyML** because its file size (~284 KB) exceeds the 100 KB limit, and traversing 100 trees in RAM is too heavy for microcontrollers with <= 256 KB memory.

## 4. Conclusions
1. Random Forest turned out to be much heavier than Logistic Regression: it takes roughly 270 times more disk space and is about 50 times slower during inference, while giving slightly lower test accuracy on this specific split.
2. The single-sample inference time for Logistic Regression is almost instant (0.05 ms), showing that for simple linear models, the main latency overhead comes from Python function calls rather than matrix math.
3. TinyML has very strict hardware limits: tree ensemble models cannot fit onto tiny microcontrollers without heavy pruning or quantization, whereas a basic linear model could easily be rewritten in C and run on an embedded board.
