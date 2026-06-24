import os
import re
import functools
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


def clean_text(text):
    """Strip URLs from a retrieved chunk before putting it in the prompt."""
    return re.sub(r'https?://\S+', '', text).strip()


@functools.lru_cache(maxsize=1)
def load_index(index_path=INDEX_PATH, data_path=DATA_PATH):
    index = TurboQuantIndex.load(index_path)
    with open(data_path, encoding="utf-8") as f:
        texts = chunk_text(f.read())
    return index, texts


def ask(query, index, texts, k=5):
    q_vec = np.array([embedder.embed_query(query)], dtype=np.float32)
    _, indices = index.search(q_vec, k=k)
    retrieved = [texts[i] for i in indices[0]]
    retrieved = list(dict.fromkeys(retrieved))  # dedupe, preserve order
    context = "\n".join(clean_text(c) for c in retrieved)

    messages = [
        {
            "role": "system",
            "content": "STRICT MODE:\n- Answer ONLY from context\n- No external knowledge\n- If missing: say 'Not found in context'"
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0.7, "num_predict": 512},
    )
    answer = response["message"]["content"].strip()

    return answer, retrieved


if __name__ == "__main__":
    index, texts = load_index()
    print("RAG ready. Ask a question (Ctrl+C to quit).\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        answer, retrieved = ask(query, index, texts)
        print("\n🔍 Retrieved Context:\n")
        for i, chunk in enumerate(retrieved, 1):
            print(f"{i}. {chunk}")
        print("\n💡 Answer:\n")
        print(answer, "\n")
