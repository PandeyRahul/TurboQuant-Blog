# TurboQuant RAG — Local Document Q&A on a 4GB GPU

A small, fully **local** Retrieval-Augmented Generation (RAG) pipeline that answers
questions from your own documents — no cloud, no API keys. It uses a **4-bit
quantized TurboQuant vector index** for fast retrieval and **Gemma 3 (via Ollama)**
for generation, and it runs comfortably on a modest **Windows machine with a 4GB
GPU (tested on an RTX 3050)**.

This project started as Mac-only [MLX](https://github.com/ml-explore/mlx) code —
inspired by [*Gemma 4 12B + TurboQuant + MTP RAG*](https://medium.com/data-science-collective/gemma-4-12b-turboquant-mtp-rag-better-ocr-self-hosted-c2cc587bea10)
on the Data Science Collective — and was re-built to run cross-platform on Ollama.
It also ships with its own **speed benchmark** and a **RAGAS quality-evaluation
suite**, so the claims here are measured, not assumed.

---

## ✨ Highlights

- **100% local & private** — documents and inference never leave your machine.
- **Tiny memory footprint** — TurboQuant stores embeddings at 4 bits/dimension
  (≈ **8× smaller** than float32), which is what makes it fit a 4GB GPU.
- **Fast retrieval** — sub-millisecond search on small corpora; **~3× faster than
  NumPy brute force** at 100k vectors (see [Benchmarks](#-benchmarks)).
- **Cross-platform** — Windows / macOS / Linux via Ollama.
- **Strict, grounded answers** — the LLM answers *only* from retrieved context, or
  says `Not found in context`.
- **Measured quality** — bundled RAGAS suite (faithfulness, recall, precision,
  relevancy).

---

## 🧠 How it works

```
                          ┌─────────────────┐
   data/docs.txt  ──────► │  chunk_text()   │  section-based chunking
                          └────────┬────────┘
                                   │ chunks
                          ┌────────▼────────┐
                          │  bge-m3 (Ollama)│  embeddings (1024-dim)
                          └────────┬────────┘
                                   │ vectors (float32)
                          ┌────────▼────────┐
                          │ TurboQuantIndex │  4-bit quantized vector store
                          │  (index.tq)     │
                          └─────────────────┘

   question ─► embed ─► index.search(k=5) ─► top chunks ─► gemma3:4b ─► answer
```

1. **Chunking** — documents are split into **section-based chunks** by detecting
   numbered headings (`1.`, `2.1`, …). One shared `chunk_text()` is used at both
   build and query time so the index stays aligned with the text.
2. **Embedding** — each chunk becomes a 1024-dim vector via Ollama's `bge-m3`.
3. **Indexing** — vectors are stored in a `TurboQuantIndex` at `bit_width=4`.
4. **Retrieval** — the question is embedded and the top-`k` nearest chunks are
   fetched.
5. **Generation** — chunks + question go to `gemma3:4b` under a strict
   context-only system prompt.

---

## 📦 Requirements

- **Python** 3.10+
- **[Ollama](https://ollama.com)** installed and running
- ~4 GB of free VRAM (or RAM — Ollama offloads overflow to system memory)

### Python packages

```bash
pip install turbovec langchain-ollama ollama numpy ragas pandas
```

### Ollama models (pull once)

```bash
ollama pull gemma3:4b     # the LLM (3.3 GB, fits a 4GB GPU)
ollama pull bge-m3        # the embedding model
```

> **Tip:** On a 4GB GPU, stick to `gemma3:4b`. The 12B model (~8 GB) will run but
> spills into system RAM and gets slow.

---

## 🚀 Quick start

```bash
# 1. Put your documents in data/docs.txt (one source document, plain text)

# 2. Build the index (run once, or whenever docs change)
python indexer.py
#  -> ✅ Index built! (14 chunks -> index/index.tq)

# 3a. Ask a single question
python main.py

# 3b. ...or start the interactive loop
python rag.py
```

Example:

```
Question: How will proposals be evaluated?

💡 Answer:
- Technical Solution and Approach: 35%
- Relevant Experience and References: 25%
- Commercial Proposal and Value: 20%
- Team and Governance: 10%
- Implementation Plan and Risk Management: 10%
```

---

## 📁 Project structure

| File | Purpose |
|------|---------|
| `chunking.py` | Section-based chunking. **Single source of truth** — shared by builder and query side so indices stay aligned. |
| `indexer.py` | Builds the TurboQuant index from `data/docs.txt` (run once). |
| `main.py` | One-shot CLI: ask a single question and exit. |
| `rag.py` | Reusable `ask()` / `load_index()` functions + an interactive chat loop. |
| `benchmark_turboquant.py` | Measures index build time, search latency, memory, and a scaling test vs NumPy. |
| `eval_ragas.py` | RAGAS evaluation suite over the real pipeline (fully local judge). |
| `data/docs.txt` | Your source document(s). |
| `index/index.tq` | The generated TurboQuant index (created by `indexer.py`). |

---

## ⚡ Benchmarks

Run it yourself:

```bash
python benchmark_turboquant.py
```

**Real index** (14 RFP sections, 1024-dim, RTX 3050):

| Metric | Value |
|---|---|
| Build index | 158.7 ms |
| Search latency | **0.147 ms** (median over 100 runs) |
| Index on disk | 15.1 KB (vs 56 KB raw float32) |

**Scaling test** (synthetic vectors, TurboQuant vs NumPy float32 brute force):

| N vectors | TurboQuant | NumPy f32 | Speedup | Memory (f32 → TQ) | Compression |
|----------:|-----------:|----------:|--------:|-------------------|------------:|
| 1,000     | 0.167 ms   | 0.107 ms  | 0.6×    | 3.9 MB → 500 KB   | 8× |
| 10,000    | 0.507 ms   | 0.969 ms  | 1.9×    | 39.1 MB → 4.9 MB  | 8× |
| 100,000   | 4.835 ms   | 14.859 ms | **3.1×**| 390.6 MB → 48.8 MB| 8× |

> **Honest note:** at tiny corpus sizes TurboQuant's quantization overhead makes it
> slightly *slower* than NumPy. The win shows up as the corpus grows — both in speed
> and (consistently 8×) memory. The slowest part of the pipeline is actually the
> embedding model, not the search.

---

## ✅ Quality evaluation (RAGAS)

Run it yourself (fully local — Ollama is both pipeline and judge, no OpenAI key):

```bash
python eval_ragas.py
```

Results on the sample RFP dataset:

| Metric | Score | Meaning |
|---|------:|---------|
| **Faithfulness** | **1.00** | Answers are grounded in context — no hallucination. |
| **Context Recall** | **1.00** | Retrieval fetches everything the answer needs. |
| **Context Precision** | **0.86** | Relevant chunks are ranked at the top. |
| **Answer Relevancy** | 0.35* | *Misleadingly low — see note. |

> \* **Answer Relevancy** works by regenerating a question from the answer and
> comparing it to the original. Terse answers (e.g. a bare date) plus a small local
> judge model confuse this metric; the answers themselves are correct. For a more
> reliable reading, set `JUDGE_MODEL = "gemma3:12b"` at the top of `eval_ragas.py`.

---

## 🔧 Configuration

Model and path settings live at the top of the relevant files:

```python
EMBED_MODEL = "bge-m3:latest"   # embedding model (Ollama)
LLM_MODEL   = "gemma3:4b"       # generation model (Ollama)
DATA_PATH   = "data/docs.txt"
INDEX_PATH  = "index/index.tq"
```

- Change `k` (number of retrieved chunks) in `ask(..., k=5)` or `index.search(..., k=5)`.
- Lower `temperature` (e.g. `0.2`) for stricter, more deterministic answers.

---

## 🩹 Troubleshooting

**`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`**
A known RAGAS ↔ langchain-community version clash (RAGAS hardcodes a Vertex AI
import that newer langchain-community removed). It has nothing to do with this
project's local setup. `eval_ragas.py` includes a small compatibility shim at the
top that injects a stub module so the import succeeds. Alternatively, upgrade RAGAS
(`pip install -U ragas`).

**`TypeError: argument 'vectors': 'list' object cannot be cast as 'ndarray'`**
TurboQuant needs a NumPy array. Wrap embeddings in
`np.array(..., dtype=np.float32)` (already handled in `indexer.py`).

**Answers only return a heading / are incomplete**
Your chunks are probably too small. This project chunks by *section*; if your
documents don't use numbered headings, adjust the heuristics in `chunking.py`.

**Model is slow / out of memory**
You may be loading a model larger than your VRAM. Use `gemma3:4b` on a 4GB GPU.

---

## 🙏 Credits & further reading

- **Inspired by** [*Gemma 4 12B + TurboQuant + MTP RAG — Better OCR, Self-Hosted*](https://medium.com/data-science-collective/gemma-4-12b-turboquant-mtp-rag-better-ocr-self-hosted-c2cc587bea10) (Data Science Collective) — the article that kicked this off.
- [TurboQuant / turbovec](https://pypi.org/project/turbovec/) — quantized vector index
- [Ollama](https://ollama.com) — local model runtime
- [Gemma 3](https://ai.google.dev/gemma) — Google's open models
- [RAGAS](https://docs.ragas.io) — RAG evaluation framework
- [bge-m3](https://huggingface.co/BAAI/bge-m3) — embedding model

---

## 📄 License

MIT — do whatever you like, attribution appreciated.
