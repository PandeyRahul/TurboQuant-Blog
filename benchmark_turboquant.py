"""Benchmark TurboQuant search speed and memory footprint.

Part A: the real RFP index  -> build time, on-disk size, single-query latency.
Part B: a scaling test       -> synthetic vectors at increasing N, comparing
                                TurboQuant against a plain NumPy float32
                                brute-force search on latency AND memory.

Part A needs Ollama running (it embeds the real docs); if it isn't, the script
skips it and still runs the synthetic scaling test, which needs no Ollama.
"""
import os
import time

import numpy as np
from turbovec import TurboQuantIndex

DIM = 1024          # bge-m3 embedding dimension
BIT_WIDTH = 4       # TurboQuant quantization (same as the project)
QUERIES = 100       # how many searches to time per measurement
K = 5               # top-k
SCALES = [1_000, 10_000, 100_000]   # corpus sizes for the scaling test


def time_search(search_fn, queries):
    """Return (median_ms, mean_ms) for running search_fn over each query."""
    timings = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    median = timings[len(timings) // 2]
    mean = sum(timings) / len(timings)
    return median, mean


def make_unit_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs


def brute_force_search(matrix, q, k):
    """Cosine top-k via NumPy (vectors are unit-norm, so dot == cosine)."""
    sims = matrix @ q[0]
    idx = np.argpartition(-sims, k)[:k]
    return idx[np.argsort(-sims[idx])]


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def part_a_real_index():
    print("=" * 60)
    print("PART A — Real RFP index")
    print("=" * 60)
    try:
        from langchain_ollama import OllamaEmbeddings
        from chunking import chunk_text

        with open("data/docs.txt", encoding="utf-8") as f:
            texts = chunk_text(f.read())

        embedder = OllamaEmbeddings(model="bge-m3:latest")

        t0 = time.perf_counter()
        vectors = np.array(embedder.embed_documents(texts), dtype=np.float32)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        index = TurboQuantIndex(dim=vectors.shape[1], bit_width=BIT_WIDTH)
        index.add(vectors)
        build_ms = (time.perf_counter() - t0) * 1000.0

        q = np.array([embedder.embed_query("How will proposals be evaluated?")],
                     dtype=np.float32)
        median, mean = time_search(lambda x: index.search(x, k=K), [q] * QUERIES)

        size = os.path.getsize("index/index.tq") if os.path.exists("index/index.tq") else None
        float32_bytes = vectors.nbytes

        print(f"Chunks indexed       : {len(texts)}")
        print(f"Embedding dim        : {vectors.shape[1]}")
        print(f"Embed all docs       : {embed_ms:.1f} ms")
        print(f"Build TurboQuant idx : {build_ms:.2f} ms")
        print(f"Search latency       : {median:.3f} ms median ({mean:.3f} ms mean) over {QUERIES} runs")
        print(f"Raw float32 in RAM   : {human_bytes(float32_bytes)}")
        if size:
            print(f"index.tq on disk     : {human_bytes(size)}")
    except Exception as e:
        print(f"(skipped — needs Ollama running and an embedder: {e})")
    print()


def part_b_scaling():
    print("=" * 60)
    print("PART B — Scaling test (synthetic vectors, no Ollama needed)")
    print("=" * 60)
    print(f"dim={DIM}, bit_width={BIT_WIDTH}, k={K}, queries={QUERIES}\n")
    header = f"{'N vectors':>12} | {'TurboQuant':>12} | {'NumPy f32':>12} | {'speedup':>8} | {'f32 RAM':>10} | {'TQ ~RAM':>10} | {'compress':>8}"
    print(header)
    print("-" * len(header))

    queries = [make_unit_vectors(1, DIM, seed=999 + i) for i in range(QUERIES)]

    for n in SCALES:
        corpus = make_unit_vectors(n, DIM, seed=n)

        index = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
        index.add(corpus)

        tq_median, _ = time_search(lambda x: index.search(x, k=K), queries)
        bf_median, _ = time_search(lambda x: brute_force_search(corpus, x, K), queries)

        f32_bytes = corpus.nbytes
        tq_bytes = int(n * DIM * BIT_WIDTH / 8)   # 4 bits per dimension
        speedup = bf_median / tq_median if tq_median else float("inf")
        compress = f32_bytes / tq_bytes if tq_bytes else float("inf")

        print(f"{n:>12,} | {tq_median:>9.3f} ms | {bf_median:>9.3f} ms | "
              f"{speedup:>7.1f}x | {human_bytes(f32_bytes):>10} | "
              f"{human_bytes(tq_bytes):>10} | {compress:>6.1f}x")
    print()


if __name__ == "__main__":
    part_a_real_index()
    part_b_scaling()
    print("Done. Copy these numbers into the blog draft.")
