"""
Flash Attention Visualizer & Memory Benchmark  (NumPy + Matplotlib)
====================================================================
Pure NumPy implementation of tiled Flash Attention (Dao et al., 2022).

Stack: NumPy · Matplotlib · CPU timing · estimated memory model
NOTE: Does NOT use PyTorch, CUDA, or real GPU profiling.
      Memory figures are theoretical (O(N²) vs O(Nd) model).
      Timing is CPU wall-clock — GPU speedup not demonstrated here.

Covers:
  - Standard scaled dot-product attention  O(N²) memory
  - Flash Attention tiled SRAM simulation  O(N)  memory
  - Step-by-step tile visualiser (Matplotlib heatmaps)
  - Memory benchmark across seq lengths 128–2048 (estimated model)
  - Frobenius-norm equivalence check  (<1e-4 threshold)

Run:
    python flash_attention.py            # full pipeline
    python flash_attention.py --vis-only # just the heatmaps
    python flash_attention.py --bench    # just the benchmark

Outputs saved to ./outputs/ (created automatically).
"""

import argparse
import os
import time
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────────────
# 1. STANDARD ATTENTION
# ──────────────────────────────────────────────────────────────────────────────

def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                       causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Vanilla scaled dot-product attention.

    Memory footprint: O(N²) — the full N×N score matrix is materialised.

    Args:
        Q: (N, d) query matrix
        K: (N, d) key matrix
        V: (N, d) value matrix
        causal: apply lower-triangular causal mask

    Returns:
        O   : (N, d) output
        attn: (N, N) attention weights (for visualisation)
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Full N×N score matrix — this is what Flash Attention avoids keeping in HBM
    S = (Q @ K.T) * scale                        # (N, N)

    if causal:
        mask = np.triu(np.full((N, N), -np.inf), k=1)
        S = S + mask

    # Numerically-stable softmax
    S_max = S.max(axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    exp_S = np.where(np.isfinite(exp_S), exp_S, 0.0)
    attn = exp_S / (exp_S.sum(axis=-1, keepdims=True) + 1e-9)

    O = attn @ V                                 # (N, d)
    return O, attn


# ──────────────────────────────────────────────────────────────────────────────
# 2. FLASH ATTENTION (tiled, online softmax)
# ──────────────────────────────────────────────────────────────────────────────

def flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    block_size: int = 64,
                    causal: bool = True,
                    record_tiles: bool = False
                    ) -> tuple[np.ndarray, list[dict]]:
    """
    Flash Attention: block-wise SRAM tiling with online softmax.

    Memory footprint: O(N) — no N×N matrix is ever fully materialised.
    Each SRAM tile is of size B×B; only one tile lives in fast memory at once.

    Algorithm (Dao et al. 2022, Algorithm 1):
      For each query block Qᵢ:
        Initialise running max mᵢ = -∞, running sum ℓᵢ = 0, output Oᵢ = 0
        For each key/value block Kⱼ, Vⱼ:
          1. Compute tile scores Sᵢⱼ = Qᵢ Kⱼᵀ / √d
          2. Apply causal mask to Sᵢⱼ
          3. m̃ = max(mᵢ, rowmax(Sᵢⱼ))          — new running max
          4. P̃ = exp(Sᵢⱼ − m̃)                  — re-scaled tile softmax numerator
          5. ℓ̃ = exp(mᵢ − m̃)·ℓᵢ + rowsum(P̃)   — updated normaliser
          6. Oᵢ = (ℓᵢ·exp(mᵢ−m̃)·Oᵢ + P̃·Vⱼ) / ℓ̃ — accumulated output
          7. mᵢ ← m̃ ; ℓᵢ ← ℓ̃

    Args:
        Q, K, V    : (N, d) matrices
        block_size : SRAM tile size B
        causal     : apply causal mask
        record_tiles: if True, capture per-tile metadata for the visualiser

    Returns:
        O     : (N, d) output (numerically equivalent to standard_attention)
        tiles : list of tile metadata dicts (empty if record_tiles=False)
    """
    N, d = Q.shape
    B = block_size
    scale = 1.0 / math.sqrt(d)

    O = np.zeros_like(Q)              # (N, d) — written back to HBM tile-by-tile
    m = np.full(N, -np.inf)           # running row-max  (N,)
    l = np.zeros(N)                   # running row-sum  (N,)

    tiles = []

    num_blocks = math.ceil(N / B)

    for i in range(num_blocks):                    # iterate over query blocks
        i_start, i_end = i * B, min((i + 1) * B, N)
        Qi = Q[i_start:i_end]                      # (Bi, d)  — loaded into SRAM

        for j in range(num_blocks):                # iterate over key/value blocks
            j_start, j_end = j * B, min((j + 1) * B, N)
            Kj = K[j_start:j_end]                  # (Bj, d)
            Vj = V[j_start:j_end]                  # (Bj, d)

            # ── step 1: tile scores ──────────────────────────────────────────
            Sij = (Qi @ Kj.T) * scale              # (Bi, Bj)

            # ── step 2: causal mask ──────────────────────────────────────────
            if causal:
                row_idx = np.arange(i_start, i_end)[:, None]
                col_idx = np.arange(j_start, j_end)[None, :]
                Sij = np.where(col_idx > row_idx, -np.inf, Sij)

            # ── step 3: new running max ──────────────────────────────────────
            tile_max = np.max(Sij, axis=-1)        # (Bi,)
            m_new = np.maximum(m[i_start:i_end], tile_max)

            # ── step 4: re-scaled softmax numerator ─────────────────────────
            P_tilde = np.exp(Sij - m_new[:, None])
            P_tilde = np.where(np.isfinite(P_tilde), P_tilde, 0.0)

            # ── step 5: updated normaliser ───────────────────────────────────
            l_new = (np.exp(m[i_start:i_end] - m_new) * l[i_start:i_end]
                     + P_tilde.sum(axis=-1))

            # ── step 6: accumulated output ───────────────────────────────────
            O[i_start:i_end] = (
                (l[i_start:i_end, None] * np.exp(m[i_start:i_end] - m_new)[:, None]
                 * O[i_start:i_end]
                 + P_tilde @ Vj)
                / (l_new[:, None] + 1e-9)
            )

            # ── step 7: update running stats ─────────────────────────────────
            m[i_start:i_end] = m_new
            l[i_start:i_end] = l_new

            if record_tiles:
                tiles.append({
                    "i": i, "j": j,
                    "row": (i_start, i_end),
                    "col": (j_start, j_end),
                    "sram_kb": (Qi.nbytes + Kj.nbytes + Vj.nbytes) / 1024,
                    "tile_scores": Sij.copy(),
                    "attn_weights": (P_tilde / (l_new[:, None] + 1e-9)).copy(),
                })

    return O, tiles


# ──────────────────────────────────────────────────────────────────────────────
# 3. MEMORY MODEL
# ──────────────────────────────────────────────────────────────────────────────

def peak_memory_mb(N: int, d: int, is_flash: bool, block_size: int = 64) -> float:
    """
    Estimate peak HBM memory usage in MB (float32).

    Standard: Q + K + V + full N×N score + attn + O  →  O(N²)
    Flash:    Q + K + V + O  (tiles only in SRAM)    →  O(N·d)

    NOTE: These are estimated/theoretical figures. Real GPU profiling
    requires torch.cuda.memory_allocated() on CUDA hardware.
    """
    bytes_per_float = 4
    qkvo = 4 * N * d * bytes_per_float             # Q, K, V, O always needed

    if is_flash:
        # SRAM holds one tile at a time: 3 × B×d  (not counted in HBM peak)
        return qkvo / 1e6
    else:
        # HBM holds full N×N score matrix + attention weights
        attn_matrix = 2 * N * N * bytes_per_float  # S and P
        return (qkvo + attn_matrix) / 1e6


# ──────────────────────────────────────────────────────────────────────────────
# 4. EQUIVALENCE CHECK
# ──────────────────────────────────────────────────────────────────────────────

def verify_equivalence(N: int = 64, d: int = 64, block_size: int = 16,
                       seed: int = 42) -> dict:
    """
    Confirm Flash Attention output matches standard attention numerically.

    Uses Frobenius norm (||O_std − O_flash||_F) as the error metric.
    Acceptable threshold: < 1e-4  (matches Dao et al. paper).
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((N, d)).astype(np.float32)
    K = rng.standard_normal((N, d)).astype(np.float32)
    V = rng.standard_normal((N, d)).astype(np.float32)

    O_std, attn_std = standard_attention(Q, K, V)
    O_flash, _      = flash_attention(Q, K, V, block_size=block_size)

    frob_err = np.linalg.norm(O_std - O_flash, "fro")
    max_err  = np.abs(O_std - O_flash).max()
    passed   = frob_err < 1e-4

    return {
        "N": N, "d": d, "block_size": block_size,
        "frobenius_error": float(frob_err),
        "max_abs_error": float(max_err),
        "passed": passed,
        "attn_matrix": attn_std,   # for visualiser
        "Q": Q, "K": K, "V": V,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(seq_lengths: list[int] | None = None,
                  d: int = 64,
                  block_size: int = 64,
                  n_runs: int = 3) -> list[dict]:
    """
    Benchmark standard vs flash attention across sequence lengths.

    Measures:
      - Peak HBM memory (model)
      - Wall-clock time (median of n_runs)
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 768, 1024, 1280, 1536, 2048]

    rng = np.random.default_rng(0)
    results = []

    print(f"\n{'─'*68}")
    print(f"  {'N':>6}  {'Std mem':>10}  {'Flash mem':>10}  {'Reduction':>10}  "
          f"{'Std ms':>8}  {'Flash ms':>8}")
    print(f"{'─'*68}")

    for N in seq_lengths:
        Q = rng.standard_normal((N, d)).astype(np.float32)
        K = rng.standard_normal((N, d)).astype(np.float32)
        V = rng.standard_normal((N, d)).astype(np.float32)

        # Wall-clock: standard
        times_std = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            standard_attention(Q, K, V)
            times_std.append(time.perf_counter() - t0)

        # Wall-clock: flash
        times_flash = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            flash_attention(Q, K, V, block_size=block_size)
            times_flash.append(time.perf_counter() - t0)

        std_mem   = peak_memory_mb(N, d, is_flash=False)
        flash_mem = peak_memory_mb(N, d, is_flash=True)
        reduction = (1 - flash_mem / std_mem) * 100
        std_ms    = np.median(times_std)   * 1000
        flash_ms  = np.median(times_flash) * 1000

        results.append({
            "N": N,
            "std_mem_mb":   std_mem,
            "flash_mem_mb": flash_mem,
            "mem_reduction_pct": reduction,
            "std_ms":   std_ms,
            "flash_ms": flash_ms,
        })

        print(f"  {N:>6}  {std_mem:>9.2f}MB  {flash_mem:>9.2f}MB  "
              f"{reduction:>9.1f}%  {std_ms:>7.1f}ms  {flash_ms:>7.1f}ms")

    print(f"{'─'*68}\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALISER
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmaps(attn_matrix: np.ndarray,
                            tiles: list[dict],
                            N: int,
                            block_size: int,
                            save_path: str = "attention_heatmaps.png") -> None:
    """
    Render four-panel visualisation:
      1. Standard attention weight heatmap (full N×N)
      2. Flash Attention with tile boundaries overlaid
      3. Single tile zoom: score vs attention weights
      4. Online softmax running stats across tiles
    """
    matplotlib.rcParams.update({
        "figure.facecolor": "#0B0F1A",
        "axes.facecolor":   "#131929",
        "text.color":       "#E8F0FE",
        "axes.labelcolor":  "#7B96BC",
        "xtick.color":      "#7B96BC",
        "ytick.color":      "#7B96BC",
        "axes.edgecolor":   "#1E2D45",
        "grid.color":       "#1E2D45",
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Flash Attention Visualizer & Memory Benchmark  —  Navyaa Sambhar",
        fontsize=14, fontweight="bold", color="#00D4FF", y=0.98
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    cyan  = "#00D4FF"
    amber = "#FFAA33"
    green = "#22D47A"
    red   = "#FF5C6A"

    # ── Panel 1: Standard attention (full matrix) ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(attn_matrix, cmap="Blues", aspect="auto",
                     vmin=0, vmax=attn_matrix.max())
    ax1.set_title(f"Standard Attention\nfull {N}×{N} matrix in HBM  O(N²)",
                  fontsize=9, color=red, pad=8)
    ax1.set_xlabel("Key position")
    ax1.set_ylabel("Query position")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ── Panel 2: Flash attention with tile grid ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(attn_matrix, cmap="Blues", aspect="auto",
                     vmin=0, vmax=attn_matrix.max())
    ax2.set_title(f"Flash Attention\ntiled SRAM blocks  O(N)  B={block_size}",
                  fontsize=9, color=cyan, pad=8)
    ax2.set_xlabel("Key position")

    # Draw tile grid
    num_blocks = math.ceil(N / block_size)
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            r0 = bi * block_size - 0.5
            c0 = bj * block_size - 0.5
            bh = min(block_size, N - bi * block_size)
            bw = min(block_size, N - bj * block_size)
            rect = patches.Rectangle(
                (c0, r0), bw, bh,
                linewidth=1.2, edgecolor=amber,
                facecolor="none", alpha=0.7
            )
            ax2.add_patch(rect)

    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # ── Panel 3: Tile zoom — first non-trivial tile ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    # Pick the first tile that has some non-masked entries
    chosen = None
    for t in tiles:
        if not np.all(t["attn_weights"] == 0):
            chosen = t
            break
    if chosen is None and tiles:
        chosen = tiles[0]

    if chosen is not None:
        im3 = ax3.imshow(chosen["attn_weights"], cmap="YlOrRd", aspect="auto")
        r0, r1 = chosen["row"]
        c0, c1 = chosen["col"]
        ax3.set_title(
            f"Tile zoom  [{r0}:{r1}, {c0}:{c1}]\n"
            f"SRAM usage: {chosen['sram_kb']:.1f} KB",
            fontsize=9, color=amber, pad=8
        )
        ax3.set_xlabel("Key (within tile)")
        ax3.set_ylabel("Query (within tile)")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Annotate each cell with its weight
        if r1 - r0 <= 8:
            for ri in range(r1 - r0):
                for ci in range(c1 - c0):
                    v = chosen["attn_weights"][ri, ci]
                    ax3.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                             fontsize=7, color="white" if v > 0.3 else "#7B96BC")
    else:
        ax3.text(0.5, 0.5, "No tiles recorded", ha="center", va="center",
                 transform=ax3.transAxes, color=amber)

    # ── Panel 4: Memory comparison bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    seq_ns = [64, 128, 256, 512, 1024]
    std_mems   = [peak_memory_mb(n, 64, False) for n in seq_ns]
    flash_mems = [peak_memory_mb(n, 64, True)  for n in seq_ns]
    x = np.arange(len(seq_ns))
    w = 0.35
    ax4.bar(x - w/2, std_mems,   w, label="Standard O(N²)", color=red,   alpha=0.85)
    ax4.bar(x + w/2, flash_mems, w, label="Flash O(N)",      color=cyan,  alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(n) for n in seq_ns])
    ax4.set_xlabel("Sequence length N")
    ax4.set_ylabel("Peak HBM memory (MB)")
    ax4.set_title("Peak Memory vs Sequence Length", fontsize=9, color="#E8F0FE", pad=8)
    ax4.legend(fontsize=8, facecolor="#131929", edgecolor="#1E2D45")
    ax4.grid(axis="y", alpha=0.3)

    # ── Panel 5: Online softmax running stats ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    # Show how running max m evolves as we process tiles for query row 0
    row_tiles = [t for t in tiles if t["row"][0] == 0]
    if row_tiles:
        running_maxes = []
        m_curr = -np.inf
        for t in row_tiles:
            tile_max = float(np.max(t["tile_scores"][0]))
            m_curr = max(m_curr, tile_max)
            running_maxes.append(m_curr)
        steps = list(range(1, len(running_maxes) + 1))
        ax5.plot(steps, running_maxes, color=green, linewidth=2, marker="o",
                 markersize=5, label="running max mᵢ")
        ax5.axhline(running_maxes[-1], color=amber, linestyle="--",
                    linewidth=1, alpha=0.7, label="final max")
        ax5.set_xlabel("KV tile step (j)")
        ax5.set_ylabel("Running row max")
        ax5.set_title("Online Softmax: running max mᵢ\n(query row 0 across KV tiles)",
                      fontsize=9, color="#E8F0FE", pad=8)
        ax5.legend(fontsize=8, facecolor="#131929", edgecolor="#1E2D45")
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "Enable record_tiles=True", ha="center", va="center",
                 transform=ax5.transAxes, color=amber)

    # ── Panel 6: Memory reduction % line chart ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ns_full = list(range(64, 2049, 64))
    reductions = [
        (1 - peak_memory_mb(n, 64, True) / peak_memory_mb(n, 64, False)) * 100
        for n in ns_full
    ]
    ax6.plot(ns_full, reductions, color=green, linewidth=2)
    ax6.fill_between(ns_full, reductions, alpha=0.15, color=green)
    ax6.set_xlabel("Sequence length N")
    ax6.set_ylabel("Memory reduction (%)")
    ax6.set_title("Flash vs Standard: Memory Reduction\nas N grows",
                  fontsize=9, color="#E8F0FE", pad=8)
    ax6.grid(alpha=0.3)
    ax6.set_ylim(0, 100)

    # Annotate key seq lengths
    for n_mark in [512, 1024, 2048]:
        r = (1 - peak_memory_mb(n_mark, 64, True) / peak_memory_mb(n_mark, 64, False)) * 100
        ax6.annotate(f"{r:.0f}%\n(N={n_mark})",
                     xy=(n_mark, r), xytext=(n_mark + 50, r - 8),
                     fontsize=7, color=green,
                     arrowprops=dict(arrowstyle="->", color=green, lw=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0B0F1A", edgecolor="none")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_benchmark_chart(results: list[dict],
                         save_path: str = "benchmark_chart.png") -> None:
    """Render memory + timing benchmark charts from run_benchmark() output."""
    matplotlib.rcParams.update({
        "figure.facecolor": "#0B0F1A",
        "axes.facecolor":   "#131929",
        "text.color":       "#E8F0FE",
        "axes.labelcolor":  "#7B96BC",
        "xtick.color":      "#7B96BC",
        "ytick.color":      "#7B96BC",
        "axes.edgecolor":   "#1E2D45",
        "grid.color":       "#1E2D45",
        "font.family":      "monospace",
    })

    ns        = [r["N"]              for r in results]
    std_mems  = [r["std_mem_mb"]     for r in results]
    fl_mems   = [r["flash_mem_mb"]   for r in results]
    std_ms    = [r["std_ms"]         for r in results]
    fl_ms     = [r["flash_ms"]       for r in results]
    reductions = [r["mem_reduction_pct"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Flash Attention — Memory & Timing Benchmark  (Dao et al. 2022)",
        fontsize=13, fontweight="bold", color="#00D4FF"
    )
    fig.patch.set_facecolor("#0B0F1A")

    cyan = "#00D4FF"
    red  = "#FF5C6A"
    green = "#22D47A"

    # Panel 1: Memory
    ax = axes[0]
    ax.plot(ns, std_mems, color=red,  linewidth=2, marker="o", markersize=4, label="Standard O(N²)")
    ax.plot(ns, fl_mems,  color=cyan, linewidth=2, marker="o", markersize=4, label="Flash O(N)")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Peak HBM memory (MB)")
    ax.set_title("Memory Usage", fontsize=10, color="#E8F0FE")
    ax.legend(fontsize=8, facecolor="#131929", edgecolor="#1E2D45")
    ax.grid(alpha=0.3)

    # Panel 2: Timing
    ax = axes[1]
    ax.plot(ns, std_ms, color=red,  linewidth=2, marker="s", markersize=4, label="Standard")
    ax.plot(ns, fl_ms,  color=cyan, linewidth=2, marker="s", markersize=4, label="Flash")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("Wall-clock Time (CPU / NumPy only)", fontsize=10, color="#E8F0FE")
    ax.legend(fontsize=8, facecolor="#131929", edgecolor="#1E2D45")
    ax.grid(alpha=0.3)
    ax.annotate("CPU: Flash is slower due to many small\nblock ops vs one optimised matmul.\nGPU speedup requires CUDA/PyTorch.",
                xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=7, color="#7B96BC", va="top")

    # Panel 3: Memory reduction %
    ax = axes[2]
    ax.bar(ns, reductions, color=green, alpha=0.8, width=60)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Memory reduction (%)")
    ax.set_title("Peak Memory Reduction", fontsize=10, color="#E8F0FE")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Label bars
    for n, r in zip(ns, reductions):
        ax.text(n, r + 1.5, f"{r:.0f}%", ha="center", fontsize=7, color=green)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0B0F1A", edgecolor="none")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flash Attention Visualizer & Benchmark")
    parser.add_argument("--vis-only",  action="store_true", help="Only render heatmaps")
    parser.add_argument("--bench",     action="store_true", help="Only run benchmark")
    parser.add_argument("--N",  type=int, default=32,  help="Seq length for visualiser (default 32)")
    parser.add_argument("--B",  type=int, default=8,   help="Block/tile size (default 8)")
    parser.add_argument("--d",  type=int, default=64,  help="Head dimension (default 64)")
    args = parser.parse_args()

    N = args.N
    B = args.B
    d = args.d

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   Flash Attention Visualizer & Memory Benchmark (NumPy)      ║")
    print("║   Navyaa Sambhar  ·  Dao et al. 2022                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Portable output dir — works on laptop, Colab, GitHub Actions, etc.
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ── Equivalence check ────────────────────────────────────────────────────
    print(f"\n[1/3] Equivalence check  N={N}, d={d}, B={B}")
    result = verify_equivalence(N=N, d=d, block_size=B)
    status = "✓ PASSED" if result["passed"] else "✗ FAILED"
    print(f"      Frobenius error : {result['frobenius_error']:.2e}  {status}")
    print(f"      Max abs error   : {result['max_abs_error']:.2e}")
    print(f"      Threshold       : 1e-4")

    if not args.bench:
        # ── Visualiser ───────────────────────────────────────────────────────
        print(f"\n[2/3] Generating attention heatmaps  (N={N}, B={B})")
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((N, d)).astype(np.float32)
        K = rng.standard_normal((N, d)).astype(np.float32)
        V = rng.standard_normal((N, d)).astype(np.float32)

        _, attn = standard_attention(Q, K, V)
        _, tiles = flash_attention(Q, K, V, block_size=B, record_tiles=True)

        plot_attention_heatmaps(
            attn_matrix=attn, tiles=tiles, N=N,
            block_size=B,
            save_path=os.path.join(out_dir, "attention_heatmaps.png")
        )

    if not args.vis_only:
        # ── Benchmark ────────────────────────────────────────────────────────
        print(f"\n[3/3] Running memory & timing benchmark")
        bench_results = run_benchmark(d=d, block_size=B)
        plot_benchmark_chart(
            bench_results,
            save_path=os.path.join(out_dir, "benchmark_chart.png")
        )

    print("\n✓ Done.\n")


if __name__ == "__main__":
    main()
