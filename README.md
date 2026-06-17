# tiled-attention-benchmark
An educational ML systems project that implements and visualizes **FlashAttention-inspired tiled attention** from scratch using **NumPy** and **Matplotlib**.

The project compares standard scaled dot-product attention with a tiled online-softmax implementation that avoids explicitly materializing the full `N × N` attention matrix. It includes numerical equivalence checks, estimated memory benchmarking, CPU timing, and visual explanations of block-wise attention computation.

> Inspired by: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al., 2022.

---

## Why This Project?

Transformer attention becomes expensive for long sequences because standard attention forms the full attention score matrix:

```text
S = QKᵀ / √d

For sequence length N, this creates an N × N matrix, causing memory usage to grow quadratically.

This project studies how FlashAttention-style tiled computation reduces memory pressure by processing attention block-by-block and using online softmax updates instead of storing the complete attention matrix.

The goal is to understand the systems idea behind memory-efficient attention in large language models.
