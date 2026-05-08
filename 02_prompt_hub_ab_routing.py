"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASKS:
  1. Write two distinct system prompts (V1: concise, V2: structured/detailed)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import hashlib
from pathlib import Path

# ── 1. Environment / imports ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from config import (
    LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_ENDPOINT,
    OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_LLM_MODEL, EMBEDDING_MODEL,
    enable_tracing,
)
enable_tracing()   # must happen before any LangChain import

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────

# V1 — Concise: 2-4 sentences, direct answers
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

# V2 — Structured: expert-level, 3-5 sentences with explicit reasoning
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

# Hub prompt names — must be globally unique in your LangSmith org
PROMPT_V1_NAME = "rag-prompt-v1"
PROMPT_V2_NAME = "rag-prompt-v2"


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client: Client) -> None:
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    Uses client.push_prompt(name, object=ChatPromptTemplate, description=...).
    Handles the case where the prompt already exists (update semantics).
    """
    for name, template, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 – concise 2-4 sentence answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 – structured expert 3-5 sentence answers"),
    ]:
        try:
            url = client.push_prompt(name, object=template, description=desc)
            print(f"✅ Pushed '{name}' → {url}")
        except Exception as exc:
            print(f"⚠️  Could not push '{name}': {exc}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Falls back to locally defined templates if the Hub is unreachable.

    Returns: {PROMPT_V1_NAME: ChatPromptTemplate, PROMPT_V2_NAME: ChatPromptTemplate}
    """
    prompts = {}
    for name, local_fallback in [
        (PROMPT_V1_NAME, PROMPT_V1),
        (PROMPT_V2_NAME, PROMPT_V2),
    ]:
        try:
            pulled = client.pull_prompt(name)
            prompts[name] = pulled
            print(f"↓ Pulled '{name}' from Hub")
        except Exception:
            prompts[name] = local_fallback
            print(f"ℹ️  Using local fallback for '{name}'")
    return prompts


# ── 5. Deterministic A/B routing ────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to V1 or V2 based on the MD5 hash of request_id.

    Routing rule:
      even MD5 integer → PROMPT_V1_NAME
      odd  MD5 integer → PROMPT_V2_NAME

    Property: same request_id ALWAYS maps to the same version (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (shared helper) ───────────────────────────────────
def build_vectorstore() -> FAISS:
    """
    Load data/knowledge_base.txt, chunk with overlap, embed, index with FAISS.
    Identical implementation to Step 1 (shared helper for DRY code).
    """
    kb_path = Path(__file__).parent / "data" / "knowledge_base.txt"
    text = kb_path.read_text(encoding="utf-8")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
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


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run one RAG inference using the specified prompt version.

    Steps:
      a) Retrieve top-3 documents with retriever.invoke(question)
      b) Concatenate page_content into a single context string
      c) Run (prompt | llm | StrOutputParser()).invoke(...)
      d) Return dict with question, answer, and version label

    The @traceable decorator captures this run as a LangSmith trace tagged
    with the prompt version, enabling per-version filtering in the UI.
    """
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # Initialise LangSmith client
    client = Client(api_key=LANGSMITH_API_KEY)

    # Push both prompts to the Hub
    print("\n📤 Pushing prompts to Prompt Hub …")
    push_prompts_to_hub(client)

    # Pull them back (verifies round-trip; falls back to local if needed)
    print("\n📥 Pulling prompts from Prompt Hub …")
    prompts = pull_prompts_from_hub(client)

    # Build vectorstore, retriever, and LLM
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    model       = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )

    # Run all 50 questions through the A/B router
    from qa_pairs import SAMPLE_QUESTIONS
    v1_count = v2_count = 0
    print("\n🔀 Running A/B routing across 50 questions …\n")

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, model, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:58]}…")

    # Routing summary
    print("\n" + "─" * 60)
    print(f"📊 Routing summary: V1={v1_count} queries | V2={v2_count} queries")
    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{LANGSMITH_PROJECT}'")
    print("   Open https://smith.langchain.com → Prompt Hub to view versions.")


if __name__ == "__main__":
    main()