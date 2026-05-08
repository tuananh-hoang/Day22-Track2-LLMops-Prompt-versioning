"""
Step 3 — RAGAS Evaluation
===========================
TASKS:
  1. Run all 50 QA pairs through BOTH prompt versions
  2. Build EvaluationDataset from SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics:
       faithfulness, answer_relevancy, context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json

⏰ NOTE: ~20-30 minutes total. Start early!

BONUS:
  - Analysis comment explaining why V1 or V2 scores higher (see end of main)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# ── 1. Environment setup ────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from config import (
    LANGSMITH_API_KEY, LANGSMITH_PROJECT,
    OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_LLM_MODEL, EMBEDDING_MODEL,
    enable_tracing,
)
enable_tracing()

# ── 2. Imports ───────────────────────────────────────────────────────────────
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from qa_pairs import QA_PAIRS


# ── 3. Prompt templates (identical to Step 2) ─────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant specializing in machine learning and AI.\n"
    "Answer the user's question using ONLY the provided context.\n"
    "Keep your answer concise (2-4 sentences).\n"
    "If the context does not contain the answer, say: "
    "'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor with deep knowledge in machine learning, "
    "deep learning, and AI systems. Provide a structured, accurate, and "
    "comprehensive answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully and identify key facts.\n"
    "2. Write a clear, well-organized answer (3-5 sentences).\n"
    "3. Include specific technical details from the context.\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 4. Build vectorstore ─────────────────────────────────────────────────────
def build_vectorstore() -> FAISS:
    """
    Load data/knowledge_base.txt, split into 500-token chunks with 50-token
    overlap, embed with OpenAI, and index with FAISS.
    """
    kb_path = Path(__file__).parent / "data" / "knowledge_base.txt"
    text = kb_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    embed = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    vectorstore = FAISS.from_texts(chunks, embed)
    print(f"📚 FAISS index built: {len(chunks)} chunks")
    return vectorstore


# ── 5. Run RAG and capture outputs + contexts ─────────────────────────────
@traceable(name="ragas-rag-query", tags=["evaluation", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run one RAG inference and return both the generated answer and the
    list of retrieved passage strings.

    IMPORTANT: contexts returned as list[str] (not joined), because RAGAS
    needs individual passage strings to compute context_recall and
    context_precision correctly.

    Returns:
        {"answer": str, "contexts": list[str]}
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # list of passage strings
    ctx_str  = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": ctx_str, "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore: FAISS, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the specified prompt version.

    Args:
        vectorstore:    Prebuilt FAISS index
        prompt_version: "v1" or "v2"

    Returns:
        list of dicts with keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    model = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\n🤖 Running 50 questions with prompt {prompt_version} …")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, model, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Convert RAG result dicts into a RAGAS EvaluationDataset.

    SingleTurnSample fields:
      user_input         → the question
      response           → the generated answer
      retrieved_contexts → list[str] of retrieved passages
      reference          → ground-truth answer (required for context_recall)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate 50 RAG outputs with all 4 RAGAS metrics.

    RAGAS evaluation internally calls the LLM many times (claim extraction,
    entailment checking, synthetic question generation) — expect ~20 min for
    50 samples × 4 metrics.

    Returns:
        {metric_name: mean_score_float}
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} …")

    dataset = build_ragas_dataset(rag_results)

    # RAGAS needs its own LLM and embedding model instances
    llm_eval = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    emb_eval = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # result[metric_name] → list of per-sample floats; take mean
    scores = {}
    metric_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    for key in metric_keys:
        raw_values = result[key]
        valid = [v for v in raw_values if v is not None and not np.isnan(v)]
        scores[key] = float(np.mean(valid)) if valid else 0.0

    # Print per-metric results
    print(f"\n  Results for prompt {version}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        bonus_star = " 🌟" if k == "faithfulness" and v >= 0.9 else ""
        print(f"  {k:30s}: {v:.4f}{star}{bonus_star}")

    return scores


# ── 8. Analysis helper (BONUS) ───────────────────────────────────────────────
def generate_analysis(v1_scores: dict, v2_scores: dict) -> str:
    """
    Generate a brief technical analysis explaining why one prompt version
    outperforms the other on each metric. (Bonus +2 pts criterion.)

    Analysis logic:
    - Faithfulness: V2's instruction to "include specific technical details
      from the context" encourages closer grounding, reducing hallucination.
    - Answer relevancy: V2's structured instructions produce more on-topic
      answers; V1's brevity can omit key aspects, lowering relevancy.
    - Context recall: Both versions use the same retriever (k=3), so recall
      differences reflect how well the prompt encourages using all retrieved
      passages rather than cherry-picking.
    - Context precision: Shorter, more focused V1 answers may cite fewer
      distracting chunks, potentially improving precision.
    """
    lines = ["## V1 vs V2 Analysis\n"]
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 >= s2 else "V2"
        diff   = abs(s1 - s2)
        lines.append(f"### {metric}")
        lines.append(f"V1={s1:.4f}  V2={s2:.4f}  → {winner} wins by {diff:.4f}")
        if metric == "faithfulness":
            if winner == "V2":
                lines.append(
                    "V2 instructs the model to include specific technical details "
                    "from the context, which reduces hallucination and increases "
                    "the fraction of answer claims that are entailed by the retrieved "
                    "passages. V1's brevity constraint sometimes causes the model to "
                    "generalise beyond what the context strictly supports."
                )
            else:
                lines.append(
                    "V1's strict conciseness constraint limits the model to only the "
                    "most directly supported claims, keeping faithfulness high. V2's "
                    "longer answers introduce a higher risk of incorporating claims "
                    "not explicitly stated in the retrieved context."
                )
        elif metric == "answer_relevancy":
            if winner == "V2":
                lines.append(
                    "V2's step-by-step instructions produce more complete, on-topic "
                    "answers, which RAGAS rewards with higher relevancy because "
                    "synthetic questions generated from them align more closely "
                    "with the original query."
                )
            else:
                lines.append(
                    "V1's concise answers stay tightly focused on the core question, "
                    "which RAGAS scores highly for relevancy since there is less "
                    "off-topic elaboration."
                )
        elif metric == "context_recall":
            lines.append(
                "Context recall is driven primarily by the retriever (k=3) and the "
                "quality of the knowledge base rather than the prompt. Small "
                f"differences here reflect whether {winner}'s prompt better "
                "exploits all three retrieved passages."
            )
        elif metric == "context_precision":
            lines.append(
                "Context precision measures whether the retrieved chunks are "
                "necessary for the answer. Both prompts use the same retriever, "
                f"so {winner}'s answer style likely refers to fewer irrelevant chunks."
            )
        lines.append("")
    return "\n".join(lines)


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # Build shared vectorstore
    vectorstore = build_vectorstore()

    # Collect RAG outputs for both prompt versions
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Run RAGAS evaluation on both
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # Print comparison table
    print("\n" + "=" * 60)
    print("  V1 vs V2 Comparison")
    print("=" * 60)
    print(f"  {'Metric':<30}  {'V1':>8}  {'V2':>8}  {'Winner'}")
    print("  " + "-" * 56)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 >= s2 else "← V2"
        flag    = " ⭐" if metric == "faithfulness" and max(s1, s2) >= 0.8 else ""
        print(f"  {metric:<30}  {s1:>8.4f}  {s2:>8.4f}  {winner}{flag}")

    # Check faithfulness target
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    print()
    if best_faith >= 0.9:
        print(f"🌟 BONUS: faithfulness = {best_faith:.4f} ≥ 0.9 for at least one version!")
    if best_faith >= 0.8:
        print(f"✅ Target met: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"⚠️  Below target ({best_faith:.4f} < 0.8). "
              "Try reducing chunk_size or adjusting prompts.")

    # BONUS: generate analysis
    analysis = generate_analysis(v1_scores, v2_scores)
    print("\n" + analysis)

    # Save JSON report to data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met":       best_faith >= 0.8,
        "bonus_target_met": best_faith >= 0.9,
        "best_faithfulness": best_faith,
        "analysis": analysis,
    }
    report_path = Path(__file__).parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n💾 Saved {report_path}")


if __name__ == "__main__":
    main()