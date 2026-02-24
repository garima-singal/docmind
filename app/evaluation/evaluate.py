# Evaluates the RAG pipeline using RAGAS metrics — faithfulness, context precision and context recall

import sys
import os
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from datasets import Dataset
from ragas import evaluate
from openai import OpenAI
from ragas.llms import llm_factory

from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import retrieve_chunks
from app.retrieval.reranker import rerank_chunks
from app.generation.llm_chain import answer_question
from config import OPENAI_API_KEY


# ---------------------------------------------------------------------------
# Test dataset — 10 question/answer pairs based on test_document.pdf
# ---------------------------------------------------------------------------

TEST_DATASET = [
    {
        "question": "What is Artificial Intelligence?",
        "ground_truth": "Artificial Intelligence refers to the simulation of human intelligence in machines programmed to think, learn, and problem-solve."
    },
    {
        "question": "What are the three types of Artificial Intelligence?",
        "ground_truth": "The three types are Narrow AI, General AI, and Super AI."
    },
    {
        "question": "What is Machine Learning?",
        "ground_truth": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."
    },
    {
        "question": "What are the three types of machine learning?",
        "ground_truth": "The three types are supervised learning, unsupervised learning, and reinforcement learning."
    },
    {
        "question": "What is a Large Language Model?",
        "ground_truth": "Large Language Models are AI models trained on massive amounts of text data to understand and generate human-like text."
    },
    {
        "question": "What is the Transformer architecture?",
        "ground_truth": "The Transformer architecture uses a self-attention mechanism that allows the model to weigh the importance of different words relative to each other."
    },
    {
        "question": "What is RAG?",
        "ground_truth": "RAG stands for Retrieval Augmented Generation. It combines large language models with external knowledge retrieval to reduce hallucinations."
    },
    {
        "question": "What is chunking in RAG?",
        "ground_truth": "Chunking is the process of splitting documents into smaller pieces before embedding them."
    },
    {
        "question": "What is a vector database?",
        "ground_truth": "A vector database stores high-dimensional vectors and supports semantic similarity search using approximate nearest neighbor algorithms."
    },
    {
        "question": "What is reranking in RAG?",
        "ground_truth": "Reranking uses a cross-encoder model to reorder retrieved chunks by relevance after initial retrieval."
    }
]


# ---------------------------------------------------------------------------
# Helper — run the full RAG pipeline for a single question
# ---------------------------------------------------------------------------

def run_pipeline_for_question(question: str) -> dict:
    result = answer_question(question)
    answer = result["answer"]

    query_embedding = embed_query(question)
    retrieved = retrieve_chunks(query_embedding, top_k=10)
    reranked = rerank_chunks(question, retrieved, top_k=3)
    contexts = [chunk["text"] for chunk in reranked]

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation():
    print("🚀 Starting RAGAS evaluation...")
    print(f"📋 Evaluating {len(TEST_DATASET)} questions\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # step 1 — collect pipeline outputs
    for i, item in enumerate(TEST_DATASET):
        print(f"⏳ Q{i+1}: {item['question'][:60]}...")
        try:
            result = run_pipeline_for_question(item["question"])
            questions.append(result["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(item["ground_truth"])
            print(f"   ✅ Done")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # step 2 — build RAGAS dataset
    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })

    # step 3 — configure LLM using llm_factory (modern RAGAS API)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    llm = llm_factory("gpt-4o-mini", client=openai_client)

    # step 4 — instantiate metrics using collections (requires llm_factory)
    from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall
    metrics = [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm)
    ]

    # step 5 — run evaluation
    print("\n📊 Running RAGAS metrics (this takes 3-5 minutes)...")
    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics
    )

    # step 6 — print results dynamically (handles any column naming)
    df = results.to_pandas()

    print("\n" + "="*60)
    print("📈 RAGAS EVALUATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))

    # step 7 — print averages for numeric columns only
    print("\n" + "="*60)
    print("📊 AVERAGE SCORES")
    print("="*60)
    skip_cols = {"user_input", "response", "retrieved_contexts", "reference",
                 "question", "answer", "contexts", "ground_truth"}
    for col in df.columns:
        if col not in skip_cols:
            try:
                avg = df[col].mean()
                print(f"  {col:<25} {avg:.4f}")
            except Exception:
                pass
    print("="*60)

    # step 8 — save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to app/evaluation/evaluation_results.csv")

    return df


if __name__ == "__main__":
    run_evaluation()