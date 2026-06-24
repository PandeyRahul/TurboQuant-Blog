import os
import re
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import ollama
from turbovec import TurboQuantIndex
from langchain_ollama import OllamaEmbeddings

from chunking import chunk_text

EMBED_MODEL = "bge-m3:latest"
LLM_MODEL = "gemma3:4b"
DATA_PATH = "data/docs.txt"
INDEX_PATH = "index/index.tq"

embedder = OllamaEmbeddings(model=EMBED_MODEL)
index = TurboQuantIndex.load(INDEX_PATH)

# Load the same chunks as indexer.py (shared chunk_text), so the indices
# returned by index.search() map back to the correct text.
with open(DATA_PATH, encoding="utf-8") as f:
    texts = chunk_text(f.read())

query = input("Question: ").strip()

q_vec = np.array([embedder.embed_query(query)], dtype=np.float32)
_, indices = index.search(q_vec, k=5)

retrieved = [texts[i] for i in indices[0]]
retrieved = list(dict.fromkeys(retrieved))  # dedupe, preserve order

# strip URLs from context
context = "\n".join(re.sub(r'https?://\S+', '', c).strip() for c in retrieved)

print("\n🔍 Retrieved Context:\n")
for i, chunk in enumerate(retrieved, 1):
    print(f"{i}. {chunk}")

messages = [
    {
        "role": "system",
        "content": "You are a strict assistant.\n\nRules:\n1. ONLY use the provided context.\n2. DO NOT use prior knowledge.\n3. If the answer is not fully in the context, say exactly: \"Not found in context\"."
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    }
]

response = ollama.chat(
    model=LLM_MODEL,
    messages=messages,
    options={"temperature": 0.7, "num_predict": 512},
)

answer = response["message"]["content"].strip()

print("\n💡 Answer:\n")
print(answer)
