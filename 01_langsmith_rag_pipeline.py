"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASKS:
  1. Load knowledge base, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable → every call creates a LangSmith trace
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm ≥ 50 traces appear.
"""

import os
import sys
from pathlib import Path

# ── 1. Environment setup ────────────────────────────────────────────────────
# IMPORTANT: tracing env vars MUST be set before importing LangChain modules
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from config import (
    LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_ENDPOINT,
    OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_LLM_MODEL, EMBEDDING_MODEL,
    enable_tracing,
)
enable_tracing()   # sets LANGCHAIN_TRACING_V2=true + keys before any LC import

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=DEFAULT_LLM_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore() -> FAISS:
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.

    Pipeline:
      a) Read data/knowledge_base.txt
      b) Split with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) FAISS.from_texts(chunks, embeddings) → build index
      d) Return the vectorstore
    """
    kb_path = Path(__file__).parent / "data" / "knowledge_base.txt"
    text = kb_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f"📚 Knowledge base split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"🗂️  FAISS index built with {len(chunks)} vectors")
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful AI assistant specializing in machine learning and AI.\n"
            "Answer the user's question using ONLY the information in the context below.\n"
            "Be accurate and concise. If the context does not contain the answer, "
            "say: 'I don't have enough information in the provided context.'\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Chain structure:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Returns: (chain, retriever)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input / output / latency to LangSmith.

    Args:
        chain:    The LCEL RAG chain built by build_rag_chain()
        question: Natural language question string

    Returns:
        Generated answer string
    """
    return chain.invoke(question)


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)
    print(f"  Project : {LANGSMITH_PROJECT}")
    print(f"  Model   : {DEFAULT_LLM_MODEL}")
    print()

    # Build vectorstore and chain
    vectorstore = build_vectorstore()
    chain, _retriever = build_rag_chain(vectorstore)

    # Import all 50 questions
    from qa_pairs import SAMPLE_QUESTIONS

    # Run all 50 questions
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        q_preview = question[:60].ljust(60)
        a_preview = answer[:90].replace("\n", " ")
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {q_preview}")
        print(f"          A: {a_preview}\n")

    print("─" * 60)
    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{LANGSMITH_PROJECT}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()