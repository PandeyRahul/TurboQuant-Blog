import os

import numpy as np
from turbovec import TurboQuantIndex
from langchain_ollama import OllamaEmbeddings

from chunking import chunk_text

EMBED_MODEL = "bge-m3:latest"
DATA_PATH = "data/docs.txt"
INDEX_PATH = "index/index.tq"

embedder = OllamaEmbeddings(model=EMBED_MODEL)

# Section-based chunks (see chunking.py). The query-time code reloads texts
# with the same chunk_text() so search indices line up with the texts.
with open(DATA_PATH, encoding="utf-8") as f:
    texts = chunk_text(f.read())

vectors = np.array(embedder.embed_documents(texts), dtype=np.float32)

index = TurboQuantIndex(dim=vectors.shape[1], bit_width=4)
index.add(vectors)

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
index.write(INDEX_PATH)

print(f"✅ Index built successfully! ({len(texts)} chunks -> {INDEX_PATH})")