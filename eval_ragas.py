"""RAGAS evaluation suite for the local TurboQuant + Ollama RAG pipeline.

Runs the *real* pipeline (rag.ask) over a curated test set drawn from
data/docs.txt, then scores it with RAGAS. Everything runs locally: Ollama
serves both the pipeline LLM and the RAGAS judge (LLM + embeddings), so no
OpenAI key is required.

Metrics:
  - faithfulness        : answer grounded in retrieved context (no hallucination)
  - answer_relevancy    : answer actually addresses the question
  - context_precision   : relevant chunks ranked at the top (needs reference)
  - context_recall      : retrieval fetched all the answer needs (needs reference)
"""
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# --- Compatibility shim (ragas <-> langchain-community) --------------------
# Some ragas builds hardcode `from langchain_community.chat_models.vertexai
# import ChatVertexAI` at import time, but newer langchain-community removed
# that submodule, which crashes ragas before our code runs. This has nothing
# to do with our local Ollama setup, so we inject a stub module to satisfy the
# import. Harmless if the real module exists or ragas is already patched.
import sys
import types

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    import langchain_community.chat_models as _chat_models

    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder; never instantiated in a local run
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
    _chat_models.vertexai = _stub
# ---------------------------------------------------------------------------

from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from rag import ask, load_index, LLM_MODEL, EMBED_MODEL

# A small judge model can be flaky at the structured output RAGAS expects.
# Override with a stronger local model if you have one pulled, e.g. "gemma3:12b".
JUDGE_MODEL = LLM_MODEL

# Curated Q&A drawn from data/docs.txt. `ground_truth` is the reference answer
# used by context_precision / context_recall.
TEST_SET = [
    {
        "question": "Who issued this RFP and what is its reference number?",
        "ground_truth": "Meridian Housing Association (MHA) issued the RFP. The reference is MHA-2026-DTP-001.",
    },
    {
        "question": "What is the budget for the programme?",
        "ground_truth": "An approved budget envelope of £2.8M over three years, inclusive of implementation, licensing, data migration, training, and first-year support.",
    },
    {
        "question": "How will proposals be evaluated?",
        "ground_truth": "Technical Solution and Approach 35%, Relevant Experience and References 25%, Commercial Proposal and Value 20%, Team and Governance 10%, and Implementation Plan and Risk Management 10%.",
    },
    {
        "question": "What is the proposal submission deadline?",
        "ground_truth": "Proposals must be submitted by 17:00 on 14 May 2026.",
    },
    {
        "question": "What must the digital resident portal support?",
        "ground_truth": "A self-service portal letting residents report repairs, track progress, manage payments, and update personal details; mobile-first design for iOS and Android; WCAG 2.1 AA accessibility; and multilingual support for the six most common languages spoken by MHA residents.",
    },
    {
        "question": "What go-live support is required?",
        "ground_truth": "Go-live support including a hypercare period of no less than 8 weeks post-launch.",
    },
]


def build_dataset():
    index, texts = load_index()
    print(f"Loaded index with {len(texts)} chunks.\n")

    samples = []
    for item in TEST_SET:
        answer, retrieved = ask(item["question"], index, texts, k=5)
        print(f"Q: {item['question']}\nA: {answer}\n")
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=answer,
                retrieved_contexts=retrieved,
                reference=item["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def main():
    dataset = build_dataset()

    # temperature=0 makes the judge as deterministic as possible.
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

    # answer_relevancy needs embeddings; the rest get the LLM via evaluate(llm=).
    # context_precision/recall use the `reference` field in each sample.
    metrics = [
        Faithfulness(),
        ResponseRelevancy(embeddings=evaluator_emb),       # answer relevancy
        LLMContextPrecisionWithReference(),                # context precision
        LLMContextRecall(),                                # context recall
    ]

    # Local generation is slow: give each judge call a long timeout and keep
    # concurrency at 1 so we don't overwhelm the Ollama server.
    run_config = RunConfig(timeout=600, max_workers=1)

    print("Running RAGAS evaluation (this is slow on a local judge)...\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=run_config,
    )

    print("\n===== RAGAS SCORES (aggregate, 0-1, higher is better) =====")
    print(result)

    try:
        df = result.to_pandas()
        print("\n===== Per-question breakdown =====")
        print(df.to_string())
        df.to_csv("ragas_results.csv", index=False)
        print("\nSaved per-question scores to ragas_results.csv")
    except Exception as e:
        print(f"\n(Per-question breakdown unavailable: {e})")


if __name__ == "__main__":
    main()
