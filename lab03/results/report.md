### Written Explanation: System Trade-offs and Cost-Effectiveness (Reddi §§4.6–4.7)

1. **Most Cost-Effective Architecture: CNN**
   - The CNN achieves the highest classification accuracy (~90.8%) while utilizing the smallest footprint (~20.5k parameters)[cite: 10].
   - **Hardware Access Patterns (§4.6)**: Convolutional layers demonstrate high **spatial locality** and extensive **weight reuse**. Small 3x3 filter kernels are loaded into fast on-chip registers/SRAM and repeatedly applied across the spatial dimensions of feature maps. This yields high arithmetic intensity (FLOPs per byte transferred from DRAM), making CNNs optimal for spatial data under memory and compute constraints.

2. **MLP Limitations: Memory Bandwidth Bound (§4.6)**
   - The MLP contains the highest parameter count (~235k) due to dense unshared connections, yet achieves moderate accuracy (~87.5%)[cite: 10].
   - **Hardware Access Patterns (§4.6)**: Fully connected layers require large weight matrices to be streamed from main memory (DRAM) for every forward pass without data reuse across spatial positions. For small batch sizes (batch = 1), arithmetic intensity is extremely low, causing performance to become memory-bandwidth bound rather than compute bound.

3. **Attention Overhead on Small Edge Regimes (§4.7)**
   - The Patch-based Transformer attains the lowest accuracy (~85.85%) and the highest single-sample inference latency (~1.85 ms)[cite: 9, 10].
   - **Hardware Access Patterns (§4.7)**: Self-attention scales quadratically with patch count and involves multiple dynamic tensor permutations, sequential LayerNorms, and Softmax operations. On small 28x28 grayscale images without extensive pre-training, the model lacks convolutional inductive biases while introducing substantial memory fragmentation and non-contiguous memory access patterns, resulting in poor latency efficiency.

4. **FLOPs Discrepancy (Analytical vs. Measured)**
   - Analytical calculations use $1\text{ MAC} = 2\text{ FLOPs}$ (multiplication and accumulation)[cite: 8].
   - Automated profiling tools like `thop` report raw Multiply-Accumulate operations (MACs) and typically ignore activation layers (ReLU) and normalization overheads, leading to measured figures that appear roughly half of theoretical peak FLOP counts[cite: 8].